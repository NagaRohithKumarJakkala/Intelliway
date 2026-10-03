//! Core types and services for the Intelliway AI gateway.

/// Configuration loading and deserialization types.
pub mod config;
/// HTTP request handlers exposed by the gateway.
pub mod handlers;
/// OpenAI-compatible chat completion models.
pub mod models;
/// Provider abstraction and provider errors.
pub mod provider;
/// Built-in provider implementations.
pub mod providers;
/// Registry used to look up configured providers.
pub mod registry;
/// Logical model and category resolution.
pub mod routing;
/// Shared application state passed to request handlers.
pub mod state;
