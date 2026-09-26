use axum::{
    routing::{get, post},
    Router,
};

use ai_gateway::{
    config::{Config, ProviderConfig},
    handlers,
    providers::mock::MockProvider,
    providers::ollama::OllamaProvider,
    registry::ProviderRegistry,
    state::AppState,
};

use std::{
    fs,
    net::SocketAddr,
    sync::Arc,
};

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // --------------------------------------------------
    // Load configuration
    // --------------------------------------------------

    let config_path =
        format!("{}/config.toml", env!("CARGO_MANIFEST_DIR"));

    let config_text = fs::read_to_string(&config_path)
        .expect("Failed to read config.toml");

    let config: Config = toml::from_str(&config_text)
        .expect("Failed to parse config.toml");

    // --------------------------------------------------
    // Create provider registry
    // --------------------------------------------------

    let mut registry = ProviderRegistry::new();

    for (name, provider_config) in config.providers {
        match provider_config {
            ProviderConfig::Mock => {
                tracing::info!(
                    provider = %name,
                    "Registering mock provider"
                );

                registry.register(
                    name,
                    Arc::new(MockProvider),
                );
            }

            ProviderConfig::Ollama { base_url } => {
                tracing::info!(
                    provider = %name,
                    base_url = %base_url,
                    "Registering Ollama provider"
                );

                registry.register(
                    name,
                    Arc::new(
                        OllamaProvider::new(base_url)
                    ),
                );
            }
        }
    }

    // --------------------------------------------------
    // Create application state
    // --------------------------------------------------

    let state = Arc::new(AppState {
        providers: Arc::new(registry),
        models: Arc::new(config.models),
        categories: Arc::new(config.categories),
    });

    // --------------------------------------------------
    // Create Axum router
    // --------------------------------------------------

    let app = Router::new()
        .route(
            "/health",
            get(health),
        )
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions),
        )
        .with_state(state);

    // --------------------------------------------------
    // Start server
    // --------------------------------------------------

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    tracing::info!(
        "AI Gateway listening on {}",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind server");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}

// --------------------------------------------------
// Health endpoint
// --------------------------------------------------

async fn health() -> &'static str {
    "OK"
}
