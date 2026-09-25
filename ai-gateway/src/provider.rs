use crate::models::{ChatCompletionRequest, ChatMessage};

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    async fn chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatMessage, ProviderError>;
}

#[derive(Debug)]
pub enum ProviderError {
    Unavailable,
    RateLimited,
    InvalidRequest,
    Internal,
}

pub struct MockProvider;

#[async_trait::async_trait]
impl Provider for MockProvider {
    async fn chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatMessage, ProviderError> {
        Ok(ChatMessage {
            role: "assistant".to_string(),
            content: format!(
                "Mock response for model '{}'",
                request.model
            ),
        })
    }
}
