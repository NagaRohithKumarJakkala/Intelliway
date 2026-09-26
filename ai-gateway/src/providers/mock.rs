use async_trait::async_trait;

use crate::models::{ChatCompletionRequest, ChatMessage};
use crate::provider::{Provider, ProviderError};

pub struct MockProvider;

#[async_trait]
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
