use std::sync::Arc;

use crate::provider::Provider;

pub struct AppState {
    pub provider: Arc<dyn Provider>,
}
