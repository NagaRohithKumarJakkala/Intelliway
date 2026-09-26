use std::collections::HashMap;
use std::sync::Arc;

use crate::provider::Provider;

pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        provider: Arc<dyn Provider>,
    ) {
        self.providers.insert(name.into(), provider);
    }

    pub fn get(
        &self,
        name: &str,
    ) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }
}
