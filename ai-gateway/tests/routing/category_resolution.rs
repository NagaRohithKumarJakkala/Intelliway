use std::collections::HashMap;

use super::{test_categories, test_models};
use ai_gateway::{config::CategoryConfig, routing::resolve_model};

#[test]
fn resolves_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model("chat", &models, &categories).unwrap();

    assert_eq!(result.requested_name, "chat");

    assert_eq!(result.logical_model, "qwen-small");

    assert_eq!(result.provider, "ollama");

    assert_eq!(result.provider_model, "qwen2.5:1.5b");
}

#[test]
fn rejects_category_with_unknown_model() {
    let models = test_models();

    let categories = HashMap::from([(
        "broken".to_string(),
        CategoryConfig {
            models: vec!["does-not-exist".to_string()],
        },
    )]);

    let result = resolve_model("broken", &models, &categories);

    assert!(result.is_err());

    assert_eq!(
        result.unwrap_err(),
        "Category 'broken' references unknown model 'does-not-exist'"
    );
}

#[test]
fn rejects_empty_category() {
    let models = test_models();

    let categories = HashMap::from([("empty".to_string(), CategoryConfig { models: vec![] })]);

    let result = resolve_model("empty", &models, &categories);

    assert!(result.is_err());

    assert_eq!(result.unwrap_err(), "Category 'empty' has no models");
}

#[test]
fn resolves_coding_category() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model("coding", &models, &categories).unwrap();

    assert_eq!(result.requested_name, "coding");

    assert_eq!(result.logical_model, "qwen-coder");

    assert_eq!(result.provider, "ollama");

    assert_eq!(result.provider_model, "qwen2.5-coder");
}
