"""Pydantic schemas used by JSON API, HTML forms and prediction responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from car_price_prediction import config


EXAMPLE_FEATURES: dict[str, Any] = {
    "Condition": "Used",
    "Vehicle_brand": "Toyota",
    "Production_year": 2018,
    "Mileage_km": 120000,
    "Power_HP": 150,
    "Displacement_cm3": 1998,
    "Fuel_type": "Gasoline",
    "Drive": "Front wheels",
    "Transmission": "Manual",
    "Type": "SUV",
    "Doors_number": 5,
}


class CarFeatures(BaseModel):
    """Validated source-level car features accepted by the prediction API."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_FEATURES},
    )

    condition: str = Field(
        ...,
        alias="Condition",
        min_length=1,
        description="Vehicle condition from the source listing vocabulary.",
    )
    vehicle_brand: str = Field(
        ...,
        alias="Vehicle_brand",
        min_length=1,
        description="Vehicle brand or manufacturer.",
    )
    production_year: int = Field(
        ...,
        alias="Production_year",
        ge=config.MIN_PRODUCTION_YEAR,
        le=config.MAX_PRODUCTION_YEAR,
        description="Calendar year when the vehicle was produced.",
    )
    mileage_km: int = Field(
        ...,
        alias="Mileage_km",
        ge=config.MIN_MILEAGE_KM,
        le=config.MAX_MILEAGE_KM,
        description="Vehicle mileage in kilometers.",
    )
    power_hp: int = Field(
        ...,
        alias="Power_HP",
        ge=config.MIN_POWER_HP,
        le=config.MAX_POWER_HP,
        description="Engine power in horsepower.",
    )
    displacement_cm3: int = Field(
        ...,
        alias="Displacement_cm3",
        ge=config.MIN_DISPLACEMENT_CM3,
        le=config.MAX_DISPLACEMENT_CM3,
        description="Engine displacement in cubic centimeters.",
    )
    fuel_type: str = Field(
        ...,
        alias="Fuel_type",
        min_length=1,
        description="Fuel type from the source listing vocabulary.",
    )
    drive: str = Field(
        ...,
        alias="Drive",
        min_length=1,
        description="Drive type from the source listing vocabulary.",
    )
    transmission: str = Field(
        ...,
        alias="Transmission",
        min_length=1,
        description="Transmission type from the source listing vocabulary.",
    )
    body_type: str = Field(
        ...,
        alias="Type",
        min_length=1,
        description="Vehicle body type from the source listing vocabulary.",
    )
    doors_number: int = Field(
        ...,
        alias="Doors_number",
        ge=config.MIN_DOORS_NUMBER,
        le=config.MAX_DOORS_NUMBER,
        description="Number of vehicle doors.",
    )


class PredictionExplanationItem(BaseModel):
    """Single local LightGBM contribution for one source feature."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_name": "Mileage_km",
                "display_name": "Przebieg",
                "feature_value": "120 000 km",
                "contribution_pln": -8200.0,
                "direction": "decreases",
            }
        }
    )

    feature_name: str = Field(
        ...,
        description="Model input feature represented by this local contribution.",
    )
    display_name: str = Field(
        ...,
        description="Human-readable feature name for UI display.",
    )
    feature_value: str = Field(
        ...,
        description="Feature value used for this prediction, formatted for display.",
    )
    contribution_pln: float = Field(
        ...,
        description=(
            "Local LightGBM SHAP contribution in PLN relative to the model base value."
        ),
    )
    direction: Literal["increases", "decreases", "neutral"] = Field(
        ...,
        description="Whether this feature contribution raises or lowers the prediction.",
    )


class PredictionResponse(BaseModel):
    """Prediction payload returned by JSON API and rendered by the HTML app."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price_pln": 52000.0,
                "currency": "PLN",
                "model_name": "LightGBM",
                "model_version": "2026-06-09T10:00:00Z",
                "vehicle_age_reference_year": 2021,
                "base_value_pln": 63000.0,
                "explanation_method": "lightgbm_pred_contrib",
                "explanations": [
                    {
                        "feature_name": "Mileage_km",
                        "display_name": "Przebieg",
                        "feature_value": "120 000 km",
                        "contribution_pln": -8200.0,
                        "direction": "decreases",
                    }
                ],
                "features": EXAMPLE_FEATURES,
            }
        }
    )

    predicted_price_pln: float = Field(
        ...,
        ge=0,
        description="Predicted vehicle offer price in PLN.",
    )
    currency: Literal["PLN"] = Field(
        default="PLN",
        description="Currency of the predicted price.",
    )
    model_name: str = Field(..., description="Name of the selected trained model.")
    model_version: str = Field(
        ...,
        description="Training timestamp or model version stored in metadata.",
    )
    vehicle_age_reference_year: int = Field(
        ...,
        description=(
            "Reference year used to transform Production_year into "
            "Vehicle_age_years."
        ),
    )
    base_value_pln: float | None = Field(
        default=None,
        description=(
            "LightGBM expected/base prediction value in PLN before local "
            "feature contributions are added."
        ),
    )
    explanation_method: Literal["lightgbm_pred_contrib"] | None = Field(
        default=None,
        description="Method used to calculate local feature contributions.",
    )
    explanations: list[PredictionExplanationItem] = Field(
        default_factory=list,
        description="Local feature contributions for this specific prediction.",
    )
    features: CarFeatures = Field(
        ...,
        description="Validated input features used for the prediction.",
    )


class HealthResponse(BaseModel):
    """Health-check payload exposing model artifact availability."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_available": True,
                "model_path": "models/car_price_model.joblib",
            }
        }
    )

    status: Literal["ok"] = Field(..., description="Application health status.")
    model_available: bool = Field(
        ...,
        description="Whether the trained model artifact exists on disk.",
    )
    model_path: str = Field(
        ...,
        description="Repository-relative path to the expected model artifact.",
    )
