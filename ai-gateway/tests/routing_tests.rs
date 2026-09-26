use std::collections::HashMap;

use ai_gateway::{
    config::{CategoryConfig, ModelConfig},
    routing::resolve_model,
};

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
                models: vec![
                    "qwen-small".to_string(),
                ],
            },
        ),
        (
            "coding".to_string(),
            CategoryConfig {
                models: vec![
                    "qwen-coder".to_string(),
                ],
            },
        ),
    ])
}

#[test]
fn resolves_direct_logical_model() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model(
        "qwen-small",
        &models,
        &categories,
    )
    .unwrap();

    assert_eq!(
        result.requested_name,
        "qwen-small"
    );

    assert_eq!(
        result.logical_model,
        "qwen-small"
    );

    assert_eq!(
        result.provider,
        "ollama"
    );

    assert_eq!(
        result.provider_model,
        "qwen2.5:1.5b"
    );
}

#[test]
fn resolves_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model(
        "chat",
        &models,
        &categories,
    )
    .unwrap();

    assert_eq!(
        result.requested_name,
        "chat"
    );

    assert_eq!(
        result.logical_model,
        "qwen-small"
    );

    assert_eq!(
        result.provider,
        "ollama"
    );

    assert_eq!(
        result.provider_model,
        "qwen2.5:1.5b"
    );
}

#[test]
fn resolves_coding_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model(
        "coding",
        &models,
        &categories,
    )
    .unwrap();

    assert_eq!(
        result.requested_name,
        "coding"
    );

    assert_eq!(
        result.logical_model,
        "qwen-coder"
    );

    assert_eq!(
        result.provider,
        "ollama"
    );

    assert_eq!(
        result.provider_model,
        "qwen2.5-coder"
    );
}

#[test]
fn rejects_unknown_model_or_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model(
        "unknown",
        &models,
        &categories,
    );

    assert!(result.is_err());

    assert_eq!(
        result.unwrap_err(),
        "Unknown model or category: unknown"
    );
}

#[test]
fn rejects_empty_category() {
    let models = test_models();

    let categories = HashMap::from([
        (
            "empty".to_string(),
            CategoryConfig {
                models: vec![],
            },
        ),
    ]);

    let result = resolve_model(
        "empty",
        &models,
        &categories,
    );

    assert!(result.is_err());

    assert_eq!(
        result.unwrap_err(),
        "Category 'empty' has no models"
    );
}

#[test]
fn rejects_category_with_unknown_model() {
    let models = test_models();

    let categories = HashMap::from([
        (
            "broken".to_string(),
            CategoryConfig {
                models: vec![
                    "does-not-exist".to_string(),
                ],
            },
        ),
    ]);

    let result = resolve_model(
        "broken",
        &models,
        &categories,
    );

    assert!(result.is_err());

    assert_eq!(
        result.unwrap_err(),
        "Category 'broken' references unknown model 'does-not-exist'"
    );
}

#[test]
fn preserves_requested_name() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model(
        "chat",
        &models,
        &categories,
    )
    .unwrap();

    // What the client requested.
    assert_eq!(
        result.requested_name,
        "chat"
    );

    // What the gateway actually selected.
    assert_eq!(
        result.logical_model,
        "qwen-small"
    );
}
