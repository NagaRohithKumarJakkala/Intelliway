mod handlers;
mod models;
mod provider;
mod state;

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;

use provider::MockProvider;
use state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let provider = MockProvider;

    let state = Arc::new(AppState {
        provider: Arc::new(provider),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions),
        )
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    tracing::info!("AI Gateway listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap();

    axum::serve(listener, app)
        .await
        .unwrap();
}

async fn health() -> &'static str {
    "OK"
}
