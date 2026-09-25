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
    provider::Provider,
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

    let message = state
        .provider
        .chat(&request)
        .await
        .expect("provider request failed");

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

    Json(response)
}
