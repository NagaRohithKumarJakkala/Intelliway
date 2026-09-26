use std::sync::Arc;

use crate::registry::ProviderRegistry;

pub struct AppState {
    pub providers: Arc<ProviderRegistry>,
}
