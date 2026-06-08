from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from car_price_prediction import config


class CarFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    condition: str = Field(..., alias="Condition", min_length=1)
    vehicle_brand: str = Field(..., alias="Vehicle_brand", min_length=1)
    production_year: int = Field(
        ...,
        alias="Production_year",
        ge=config.MIN_PRODUCTION_YEAR,
        le=config.MAX_PRODUCTION_YEAR,
    )
    mileage_km: int = Field(..., alias="Mileage_km", ge=0)
    power_hp: int = Field(..., alias="Power_HP", gt=0)
    displacement_cm3: int = Field(..., alias="Displacement_cm3", gt=0)
    fuel_type: str = Field(..., alias="Fuel_type", min_length=1)
    drive: str = Field(..., alias="Drive", min_length=1)
    transmission: str = Field(..., alias="Transmission", min_length=1)
    body_type: str = Field(..., alias="Type", min_length=1)
    doors_number: int = Field(..., alias="Doors_number", ge=1, le=6)


class PredictionResponse(BaseModel):
    predicted_price_pln: float
    model_name: str
    model_version: str
    features: dict[str, Any]


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, Any]


def features_to_frame(features: CarFeatures) -> pd.DataFrame:
    payload = features.model_dump(by_alias=True)
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


def predict_price(features: CarFeatures) -> PredictionResponse:
    bundle = load_model_bundle()
    frame = features_to_frame(features)
    prediction = bundle.model.predict(frame)
    predicted_price = max(0.0, float(prediction[0]))

    return PredictionResponse(
        predicted_price_pln=round(predicted_price, 2),
        model_name=bundle.metadata.get("selected_model", "unknown"),
        model_version=bundle.metadata.get("trained_at_utc", "unknown"),
        features=features.model_dump(by_alias=True),
    )
