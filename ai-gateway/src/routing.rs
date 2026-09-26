use std::collections::HashMap;

use crate::config::{CategoryConfig, ModelConfig};

#[derive(Debug, Clone)]
/// The concrete provider mapping selected for a client model request.
pub struct ResolvedModel {
    /// Model or category name supplied by the client.
    pub requested_name: String,
    /// Logical model selected from a category, or the requested model itself.
    pub logical_model: String,
    /// Registered provider instance that should handle the request.
    pub provider: String,
    /// Provider-specific model name to send downstream.
    pub provider_model: String,
}

impl ResolvedModel {
    /// Builds a resolved model from a logical model configuration.
    pub fn from_model(requested_name: String, logical_model: String, config: &ModelConfig) -> Self {
        Self {
            requested_name,
            logical_model,
            provider: config.provider.clone(),
            provider_model: config.model.clone(),
        }
    }
}

/// Resolves a client model or category name to a configured provider model.
///
/// Categories use their first configured model. An error is returned when the
/// requested name is unknown, a category is empty, or a category references an
/// unknown model.
pub fn resolve_model(
    requested_name: &str,
    models: &HashMap<String, ModelConfig>,
    categories: &HashMap<String, CategoryConfig>,
) -> Result<ResolvedModel, String> {
    // Direct logical-model request.
    if let Some(model_config) = models.get(requested_name) {
        return Ok(ResolvedModel::from_model(
            requested_name.to_string(),
            requested_name.to_string(),
            model_config,
        ));
    }

    // Category request.
    if let Some(category) = categories.get(requested_name) {
        let logical_model = category
            .models
            .first()
            .ok_or_else(|| format!("Category '{}' has no models", requested_name))?;

        let model_config = models.get(logical_model).ok_or_else(|| {
            format!(
                "Category '{}' references unknown model '{}'",
                requested_name, logical_model
            )
        })?;

        return Ok(ResolvedModel::from_model(
            requested_name.to_string(),
            logical_model.clone(),
            model_config,
        ));
    }

    Err(format!("Unknown model or category: {}", requested_name))
}
