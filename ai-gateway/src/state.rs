use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{CategoryConfig, ModelConfig};
use crate::registry::ProviderRegistry;

/// Immutable application data shared by all HTTP request handlers.
pub struct AppState {
    /// Registered provider implementations.
    pub providers: Arc<ProviderRegistry>,
    /// Configured logical models keyed by name.
    pub models: Arc<HashMap<String, ModelConfig>>,
    /// Configured model categories keyed by name.
    pub categories: Arc<HashMap<String, CategoryConfig>>,
}
