from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from car_price_prediction import config
from car_price_prediction.schemas import CarFeatures, PredictionResponse


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, Any]


def features_to_frame(
    features: CarFeatures,
    vehicle_age_reference_year: int = config.MAX_PRODUCTION_YEAR,
) -> pd.DataFrame:
    payload = features.model_dump(by_alias=True)
    production_year = payload.pop(config.PRODUCTION_YEAR_COLUMN)
    payload[config.VEHICLE_AGE_COLUMN] = vehicle_age_reference_year - production_year
    return pd.DataFrame([payload], columns=config.FEATURE_COLUMNS)


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
    prediction = bundle.model.predict(frame)
    predicted_price = max(0.0, float(prediction[0]))

    return PredictionResponse(
        predicted_price_pln=round(predicted_price, 2),
        model_name=bundle.metadata.get("selected_model", "unknown"),
        model_version=bundle.metadata.get("trained_at_utc", "unknown"),
        vehicle_age_reference_year=int(vehicle_age_reference_year),
        features=features,
    )
