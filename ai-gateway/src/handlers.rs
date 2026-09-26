use axum::{
    extract::State,
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
    state::AppState,
};

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {

    tracing::info!(
        model = %request.model,
        messages = request.messages.len(),
        "received chat completion request"
    );

    // Split "provider/model"
    let (provider_name, model_name) = match request.model.split_once('/') {
        Some((provider, model)) => (
            provider.to_string(),
            model.to_string(),
        ),

        None => (
            "mock".to_string(),
            request.model.clone(),
        ),
    };

    // Find provider
    let provider = state
        .providers
        .get(&provider_name)
        .expect("provider not found");

    // Create a new request with only the model name.
    let provider_request = ChatCompletionRequest {
        model: model_name.clone(),
        messages: request.messages,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        stream: request.stream,
    };

    // Send request to provider
    let message = provider
        .chat(&provider_request)
        .await
        .expect("provider request failed");

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        model: model_name,

        choices: vec![
            Choice {
                index: 0,
                message,
                finish_reason: "stop".to_string(),
            }
        ],
    };

    Json(response)
}
