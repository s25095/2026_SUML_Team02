"""Helpers for mapping transformed model columns back to source features."""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from car_price_prediction import config


def required_pipeline_step(pipeline: Any, step_name: str) -> Any:
    """Return a required sklearn pipeline step or raise a clear artifact error."""

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            "Trained model artifact must be a scikit-learn Pipeline with named steps."
        )

    try:
        return pipeline.named_steps[step_name]
    except KeyError as error:
        raise ValueError(
            f"Trained model pipeline is missing required step: {step_name}"
        ) from error


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names produced by the fitted preprocessing step."""

    preprocessor = required_pipeline_step(pipeline, "preprocessor")
    return [str(feature) for feature in preprocessor.get_feature_names_out()]


def source_feature_name(transformed_feature_name: str) -> str:
    """Map a transformed numeric/one-hot feature back to its source column."""

    if "__" not in transformed_feature_name:
        return transformed_feature_name

    transformer_name, feature_name = transformed_feature_name.split("__", maxsplit=1)
    if transformer_name == "numeric":
        return feature_name

    if transformer_name == "categorical":
        for column in config.CATEGORICAL_FEATURE_COLUMNS:
            if feature_name == column or feature_name.startswith(f"{column}_"):
                return column

    raise ValueError(f"Unknown transformed feature prefix: {transformer_name}")
