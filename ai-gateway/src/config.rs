use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
/// Complete gateway configuration loaded from TOML.
pub struct Config {
    /// Configured provider instances keyed by provider name.
    pub providers: HashMap<String, ProviderConfig>,
    /// Logical models and their provider mappings.
    pub models: HashMap<String, ModelConfig>,
    /// Model categories that clients may request.
    pub categories: HashMap<String, CategoryConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
/// Configuration for a provider implementation.
pub enum ProviderConfig {
    #[serde(rename = "mock")]
    /// An in-process provider used for testing and local development.
    Mock,

    #[serde(rename = "ollama")]
    /// An Ollama server identified by its base URL.
    Ollama {
        /// Base URL of the Ollama server.
        base_url: String,
    },
}

#[derive(Debug, Deserialize, Clone)]
/// Maps a logical model name to a provider model.
pub struct ModelConfig {
    /// Name of the registered provider instance.
    pub provider: String,
    /// Model name understood by that provider.
    pub model: String,
}

#[derive(Debug, Deserialize, Clone)]
/// Ordered list of logical models belonging to a category.
pub struct CategoryConfig {
    /// Models considered for this category, in resolution order.
    pub models: Vec<String>,
}
