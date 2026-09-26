mod handlers;
mod models;
mod provider;
mod providers;
mod registry;
mod state;

use axum::{
    routing::{get, post},
    Router,
};

use std::net::SocketAddr;
use std::sync::Arc;

use providers::mock::MockProvider;
use providers::ollama::OllamaProvider;
use registry::ProviderRegistry;
use state::AppState;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // --------------------------------------------------
    // Create provider registry
    // --------------------------------------------------

    let mut registry = ProviderRegistry::new();

    // Register Mock provider
    registry.register(
        "mock",
        Arc::new(MockProvider),
    );

    // Register Ollama provider
    registry.register(
        "ollama",
        Arc::new(
            OllamaProvider::new(
                "http://localhost:11434".to_string()
            )
        ),
    );

    // --------------------------------------------------
    // Create application state
    // --------------------------------------------------

    let state = Arc::new(AppState {
        providers: Arc::new(registry),
    });

    // --------------------------------------------------
    // Create Axum router
    // --------------------------------------------------

    let app = Router::new()
        // Health check
        .route(
            "/health",
            get(health),
        )

        // OpenAI-compatible chat endpoint
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions),
        )

        // Make provider registry available to handlers
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
