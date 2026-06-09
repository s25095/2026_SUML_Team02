from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from car_price_prediction import config
from car_price_prediction.model.feature_names import (
    required_pipeline_step,
    source_feature_name,
    transformed_feature_names,
)
from car_price_prediction.schemas import (
    CarFeatures,
    PredictionExplanationItem,
    PredictionResponse,
)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, Any]


FEATURE_DISPLAY_NAMES = {
    "Condition": "Stan",
    "Vehicle_brand": "Marka",
    config.VEHICLE_AGE_COLUMN: "Wiek auta",
    "Mileage_km": "Przebieg",
    "Power_HP": "Moc",
    "Displacement_cm3": "Pojemnosc silnika",
    "Fuel_type": "Paliwo",
    "Drive": "Naped",
    "Transmission": "Skrzynia",
    "Type": "Typ nadwozia",
    "Doors_number": "Liczba drzwi",
}


def features_to_frame(
    features: CarFeatures,
    vehicle_age_reference_year: int = config.MAX_PRODUCTION_YEAR,
) -> pd.DataFrame:
    payload = features.model_dump(by_alias=True)
    production_year = payload.pop(config.PRODUCTION_YEAR_COLUMN)
    payload[config.VEHICLE_AGE_COLUMN] = vehicle_age_reference_year - production_year
    return pd.DataFrame([payload], columns=config.FEATURE_COLUMNS)


def model_input_frame(pipeline: Any, frame: pd.DataFrame) -> pd.DataFrame:
    feature_names = transformed_feature_names(pipeline)
    preprocessor = required_pipeline_step(pipeline, "preprocessor")
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(np.asarray(transformed, dtype=float), columns=feature_names)


def is_lightgbm_bundle(bundle: ModelBundle) -> bool:
    model = required_pipeline_step(bundle.model, "model")
    selected_model = str(bundle.metadata.get("selected_model", "")).lower()
    model_class = model.__class__.__name__.lower()
    return selected_model == "lightgbm" or model_class == "lgbmregressor"


def format_integer(value: object) -> str:
    return f"{int(round(float(value))):,}".replace(",", " ")


def feature_display_value(
    source_feature: str,
    frame: pd.DataFrame,
    features: CarFeatures,
) -> str:
    value = frame.iloc[0][source_feature]
    if source_feature == config.VEHICLE_AGE_COLUMN:
        return f"{features.production_year} ({format_integer(value)} lat)"
    if source_feature == "Mileage_km":
        return f"{format_integer(value)} km"
    if source_feature == "Power_HP":
        return f"{format_integer(value)} KM"
    if source_feature == "Displacement_cm3":
        return f"{format_integer(value)} cm3"
    if source_feature == "Doors_number":
        return format_integer(value)
    return str(value)


def contribution_direction(
    contribution_pln: float,
) -> str:
    if contribution_pln > 0:
        return "increases"
    if contribution_pln < 0:
        return "decreases"
    return "neutral"


def lightgbm_prediction_explanations(
    bundle: ModelBundle,
    transformed_frame: pd.DataFrame,
    frame: pd.DataFrame,
    features: CarFeatures,
) -> (
    tuple[float, list[PredictionExplanationItem]]
    | tuple[None, list[PredictionExplanationItem]]
):
    if not is_lightgbm_bundle(bundle):
        return None, []

    model = required_pipeline_step(bundle.model, "model")
    raw_contributions = model.predict(transformed_frame, pred_contrib=True)
    if hasattr(raw_contributions, "toarray"):
        raw_contributions = raw_contributions.toarray()
    contributions = np.asarray(raw_contributions, dtype=float)
    if contributions.ndim == 1:
        contributions = contributions.reshape(1, -1)

    feature_names = [str(column) for column in transformed_frame.columns]
    row_contributions = contributions[0]
    if len(row_contributions) != len(feature_names) + 1:
        raise ValueError(
            "LightGBM contribution count does not match transformed feature count: "
            f"{len(row_contributions)} != {len(feature_names) + 1}"
        )

    base_value = float(row_contributions[-1])
    grouped_contributions: dict[str, float] = {}
    for feature_name, contribution in zip(
        feature_names,
        row_contributions[:-1],
        strict=True,
    ):
        source_feature = source_feature_name(feature_name)
        grouped_contributions[source_feature] = grouped_contributions.get(
            source_feature,
            0.0,
        ) + float(contribution)

    explanations = []
    for source_feature, contribution in sorted(
        grouped_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        contribution_pln = round(contribution, 2)
        explanations.append(
            PredictionExplanationItem(
                feature_name=source_feature,
                display_name=FEATURE_DISPLAY_NAMES.get(source_feature, source_feature),
                feature_value=feature_display_value(source_feature, frame, features),
                contribution_pln=contribution_pln,
                direction=contribution_direction(contribution_pln),
            )
        )
    return round(base_value, 2), explanations


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model is missing. Run `uv run train-model` first."
        )

    model = joblib.load(config.MODEL_PATH)
    metadata: dict[str, Any] = {}
    if config.MODEL_METADATA_PATH.exists():
        with config.MODEL_METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

    return ModelBundle(model=model, metadata=metadata)


def model_available() -> bool:
    return config.MODEL_PATH.exists()


def warm_model_bundle() -> bool:
    if not model_available():
        return False

    load_model_bundle()
    return True


def predict_price(features: CarFeatures) -> PredictionResponse:
    bundle = load_model_bundle()
    vehicle_age_reference_year = bundle.metadata.get(
        "vehicle_age_reference_year",
        config.MAX_PRODUCTION_YEAR,
    )
    frame = features_to_frame(
        features,
        vehicle_age_reference_year=int(vehicle_age_reference_year),
    )
    transformed_frame = model_input_frame(bundle.model, frame)
    model = required_pipeline_step(bundle.model, "model")
    prediction = model.predict(transformed_frame)
    predicted_price = max(0.0, float(prediction[0]))
    base_value, explanations = lightgbm_prediction_explanations(
        bundle,
        transformed_frame,
        frame,
        features,
    )

    return PredictionResponse(
        predicted_price_pln=round(predicted_price, 2),
        model_name=bundle.metadata.get("selected_model", "unknown"),
        model_version=bundle.metadata.get("trained_at_utc", "unknown"),
        vehicle_age_reference_year=int(vehicle_age_reference_year),
        base_value_pln=base_value,
        explanation_method="lightgbm_pred_contrib" if explanations else None,
        explanations=explanations,
        features=features,
    )
