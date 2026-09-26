use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{CategoryConfig, ModelConfig};
use crate::registry::ProviderRegistry;

pub struct AppState {
    pub providers: Arc<ProviderRegistry>,
    pub models: Arc<HashMap<String, ModelConfig>>,
    pub categories: Arc<HashMap<String, CategoryConfig>>,
}
