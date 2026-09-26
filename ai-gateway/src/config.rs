use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub providers: HashMap<String, ProviderConfig>,
    pub models: HashMap<String, ModelConfig>,
    pub categories: HashMap<String, CategoryConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    #[serde(rename = "mock")]
    Mock,

    #[serde(rename = "ollama")]
    Ollama {
        base_url: String,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CategoryConfig {
    pub models: Vec<String>,
}
