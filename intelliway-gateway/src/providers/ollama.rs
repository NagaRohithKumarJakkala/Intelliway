use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::models::{ChatCompletionRequest, ChatMessage};
use crate::provider::{Provider, ProviderError};

/// Provider implementation backed by an Ollama HTTP server.
pub struct OllamaProvider {
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }
}

// --------------------------------------------------
// Request sent to Ollama
// --------------------------------------------------

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

// --------------------------------------------------
// Message used by Ollama
// --------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

// --------------------------------------------------
// Response received from Ollama
// --------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
}

// --------------------------------------------------
// Provider implementation
// --------------------------------------------------

#[async_trait]
impl Provider for OllamaProvider {
    async fn chat(&self, request: &ChatCompletionRequest) -> Result<ChatMessage, ProviderError> {
        let ollama_request = OllamaRequest {
            model: request.model.clone(),

            messages: request
                .messages
                .iter()
                .map(|message| OllamaMessage {
                    role: message.role.clone(),
                    content: message.content.clone(),
                })
                .collect(),

            stream: false,
        };

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|_| ProviderError::Unavailable)?;

        if !response.status().is_success() {
            return Err(ProviderError::Internal);
        }

        let response: OllamaResponse =
            response.json().await.map_err(|_| ProviderError::Internal)?;

        Ok(ChatMessage {
            role: response.message.role,
            content: response.message.content,
        })
    }
}
