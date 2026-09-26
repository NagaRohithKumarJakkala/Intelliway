use std::collections::HashMap;

use ai_gateway::{
    config::{CategoryConfig, ModelConfig},
    routing::resolve_model,
};

mod category_resolution;
mod direct_logical_model;

fn test_models() -> HashMap<String, ModelConfig> {
    HashMap::from([
        (
            "qwen-small".to_string(),
            ModelConfig {
                provider: "ollama".to_string(),
                model: "qwen2.5:1.5b".to_string(),
            },
        ),
        (
            "qwen-coder".to_string(),
            ModelConfig {
                provider: "ollama".to_string(),
                model: "qwen2.5-coder".to_string(),
            },
        ),
    ])
}

fn test_categories() -> HashMap<String, CategoryConfig> {
    HashMap::from([
        (
            "chat".to_string(),
            CategoryConfig {
                models: vec!["qwen-small".to_string()],
            },
        ),
        (
            "coding".to_string(),
            CategoryConfig {
                models: vec!["qwen-coder".to_string()],
            },
        ),
    ])
}

#[test]
fn rejects_unknown_model_or_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model("unknown", &models, &categories);

    assert!(result.is_err());

    assert_eq!(result.unwrap_err(), "Unknown model or category: unknown");
}

#[test]
fn preserves_requested_name() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model("chat", &models, &categories).unwrap();

    // What the client requested.
    assert_eq!(result.requested_name, "chat");

    // What the gateway actually selected.
    assert_eq!(result.logical_model, "qwen-small");
}
