use super::{test_categories, test_models};
use ai_gateway::routing::resolve_model;

#[test]
fn resolves_direct_logical_model() {
    let models = test_models();
    let categories = test_categories();

    let result = resolve_model("qwen-small", &models, &categories).unwrap();

    assert_eq!(result.requested_name, "qwen-small");

    assert_eq!(result.logical_model, "qwen-small");

    assert_eq!(result.provider, "ollama");

    assert_eq!(result.provider_model, "qwen2.5:1.5b");
}
