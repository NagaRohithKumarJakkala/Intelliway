use std::collections::HashMap;

use crate::config::{CategoryConfig, ModelConfig};

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub requested_name: String,
    pub logical_model: String,
    pub provider: String,
    pub provider_model: String,
}

impl ResolvedModel {
    pub fn from_model(
        requested_name: String,
        logical_model: String,
        config: &ModelConfig,
    ) -> Self {
        Self {
            requested_name,
            logical_model,
            provider: config.provider.clone(),
            provider_model: config.model.clone(),
        }
    }
}

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
            .ok_or_else(|| {
                format!("Category '{}' has no models", requested_name)
            })?;

        let model_config = models
            .get(logical_model)
            .ok_or_else(|| {
                format!(
                    "Category '{}' references unknown model '{}'",
                    requested_name,
                    logical_model
                )
            })?;

        return Ok(ResolvedModel::from_model(
            requested_name.to_string(),
            logical_model.clone(),
            model_config,
        ));
    }

    Err(format!(
        "Unknown model or category: {}",
        requested_name
    ))
}
