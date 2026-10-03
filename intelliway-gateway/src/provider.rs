use crate::models::{ChatCompletionRequest, ChatMessage};

#[async_trait::async_trait]
/// Interface implemented by every chat completion backend.
pub trait Provider: Send + Sync {
    /// Sends a chat request to the provider and returns the assistant message.
    async fn chat(&self, request: &ChatCompletionRequest) -> Result<ChatMessage, ProviderError>;
}

#[derive(Debug)]
/// Errors that can occur while communicating with a provider.
pub enum ProviderError {
    /// The provider could not be reached or is unavailable.
    Unavailable,
    /// The provider rejected the request due to rate limiting.
    RateLimited,
    /// The request was invalid for the provider.
    InvalidRequest,
    /// The provider returned an internal or otherwise unexpected failure.
    Internal,
}

/// In-process provider that returns a deterministic response for testing.
pub struct MockProvider;

#[async_trait::async_trait]
impl Provider for MockProvider {
    async fn chat(&self, request: &ChatCompletionRequest) -> Result<ChatMessage, ProviderError> {
        Ok(ChatMessage {
            role: "assistant".to_string(),
            content: format!("Mock response for model '{}'", request.model),
        })
    }
}
