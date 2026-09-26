use axum::{
    extract::State,
    http::StatusCode,
    Json,
};

use std::sync::Arc;
use uuid::Uuid;

use crate::{
    models::{
        ChatCompletionRequest,
        ChatCompletionResponse,
        Choice,
    },
    routing::resolve_model,
    state::AppState,
};

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<
    Json<ChatCompletionResponse>,
    (StatusCode, String),
> {
    tracing::info!(
        model = %request.model,
        messages = request.messages.len(),
        "received chat completion request"
    );

    // Resolve the requested model name.
    //
    // The request can contain either:
    //   - a logical model, e.g. "qwen-small"
    //   - a category, e.g. "chat"
    //
    // The resolver determines the logical model,
    // provider, and concrete provider model.
    let resolved = resolve_model(
        &request.model,
        &state.models,
        &state.categories,
    )
    .map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            error,
        )
    })?;

    tracing::info!(
        requested_model = %resolved.requested_name,
        logical_model = %resolved.logical_model,
        provider = %resolved.provider,
        provider_model = %resolved.provider_model,
        "model resolved"
    );

    // Find the provider implementation.
    let provider = state
        .providers
        .get(&resolved.provider)
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Configured provider not found: {}",
                    resolved.provider
                ),
            )
        })?;

    // Create the provider-specific request.
    //
    // The provider receives the actual model name,
    // while the client only knows the logical model/category.
    let provider_request = ChatCompletionRequest {
        model: resolved.provider_model.clone(),
        messages: request.messages,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        stream: request.stream,
    };

    // Send the request to the selected provider.
    let message = provider
        .chat(&provider_request)
        .await
        .map_err(|error| {
            tracing::error!(
                ?error,
                requested_model = %resolved.requested_name,
                logical_model = %resolved.logical_model,
                provider = %resolved.provider,
                provider_model = %resolved.provider_model,
                "Provider request failed"
            );

            (
                StatusCode::BAD_GATEWAY,
                format!(
                    "Provider request failed: {:?}",
                    error
                ),
            )
        })?;

    // Return an OpenAI-compatible response.
    //
    // We return the model name requested by the client,
    // rather than exposing the provider-specific model.
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        model: request.model,
        choices: vec![
            Choice {
                index: 0,
                message,
                finish_reason: "stop".to_string(),
            }
        ],
    };

    Ok(Json(response))
}
