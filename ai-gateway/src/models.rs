use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}
