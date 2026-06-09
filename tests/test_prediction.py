from __future__ import annotations

import pytest
import pandas as pd
from lightgbm import LGBMRegressor
from pydantic import ValidationError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from car_price_prediction import config
from car_price_prediction.model.feature_names import source_feature_name
from car_price_prediction.model import predict as prediction_service
from car_price_prediction.model.predict import (
    ModelBundle,
    features_to_frame,
    is_lightgbm_bundle,
)
from car_price_prediction.model.train import ModelCandidate, train_final_pipeline
from car_price_prediction.schemas import CarFeatures


def valid_features() -> CarFeatures:
    return CarFeatures(
        Condition="Used",
        Vehicle_brand="Toyota",
        Production_year=2018,
        Mileage_km=120000,
        Power_HP=150,
        Displacement_cm3=1998,
        Fuel_type="Gasoline",
        Drive="Front wheels",
        Transmission="Manual",
        Type="SUV",
        Doors_number=5,
    )


def lightgbm_training_frame() -> pd.DataFrame:
    rows = []
    brands = [
        "Toyota",
        "Skoda",
        "Ford",
        "BMW",
        "Kia",
        "Opel",
        *[f"Brand_{index}" for index in range(30)],
    ]
    brand_bonus = {brand: index * 700 for index, brand in enumerate(brands)}
    for index in range(72):
        brand = brands[index % len(brands)]
        year = 2008 + index % 12
        mileage = 25000 + index * 7200
        power = 90 + index * 4
        automatic_bonus = 7000 if index % 3 == 0 else 0
        rows.append(
            {
                "Price": (
                    14000
                    + (year - 2008) * 3800
                    + power * 260
                    - mileage * 0.06
                    + brand_bonus[brand]
                    + automatic_bonus
                ),
                "Condition": "Used",
                "Vehicle_brand": brand,
                "Production_year": year,
                "Mileage_km": mileage,
                "Power_HP": power,
                "Displacement_cm3": 1400 + index * 45,
                "Fuel_type": "Gasoline" if index % 2 else "Diesel",
                "Drive": "Front wheels",
                "Transmission": "Automatic" if index % 3 == 0 else "Manual",
                "Type": "SUV" if index % 2 else "Combi",
                "Doors_number": 5 if index % 4 else 3,
            }
        )
    return pd.DataFrame(rows)


def test_features_to_frame_uses_model_column_order():
    frame = features_to_frame(valid_features(), vehicle_age_reference_year=2021)

    assert list(frame.columns) == [
        "Condition",
        "Vehicle_brand",
        "Vehicle_age_years",
        "Mileage_km",
        "Power_HP",
        "Displacement_cm3",
        "Fuel_type",
        "Drive",
        "Transmission",
        "Type",
        "Doors_number",
    ]
    assert frame.loc[0, "Vehicle_brand"] == "Toyota"
    assert frame.loc[0, "Vehicle_age_years"] == 3


def test_predict_price_includes_lightgbm_local_explanations(monkeypatch):
    candidate = ModelCandidate(
        "lightgbm",
        LGBMRegressor(
            n_estimators=30,
            learning_rate=0.1,
            num_leaves=7,
            min_child_samples=1,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
    )
    pipeline = train_final_pipeline(
        lightgbm_training_frame(),
        selected_model_name="lightgbm",
        candidates=[candidate],
    )
    bundle = ModelBundle(
        model=pipeline,
        metadata={
            "selected_model": "lightgbm",
            "trained_at_utc": "test-version",
            "vehicle_age_reference_year": 2019,
        },
    )
    monkeypatch.setattr(prediction_service, "load_model_bundle", lambda: bundle)

    response = prediction_service.predict_price(valid_features())

    assert response.explanation_method == "lightgbm_pred_contrib"
    assert response.base_value_pln is not None
    assert {item.feature_name for item in response.explanations} == set(
        config.FEATURE_COLUMNS
    )
    assert response.explanations == sorted(
        response.explanations,
        key=lambda item: abs(item.contribution_pln),
        reverse=True,
    )
    assert response.explanations[0].display_name
    assert {item.direction for item in response.explanations} <= {
        "increases",
        "decreases",
        "neutral",
    }

    explained_price = response.base_value_pln + sum(
        item.contribution_pln for item in response.explanations
    )
    assert response.predicted_price_pln == pytest.approx(explained_price, abs=1.0)


def test_is_lightgbm_bundle_rejects_missing_model_step():
    bundle = ModelBundle(
        model=Pipeline([("preprocessor", FunctionTransformer())]),
        metadata={"selected_model": "lightgbm"},
    )

    with pytest.raises(ValueError, match="missing required step: model"):
        is_lightgbm_bundle(bundle)


def test_predict_price_rejects_non_pipeline_artifact(monkeypatch):
    monkeypatch.setattr(
        prediction_service,
        "load_model_bundle",
        lambda: ModelBundle(model=object(), metadata={}),
    )

    with pytest.raises(ValueError, match="scikit-learn Pipeline"):
        prediction_service.predict_price(valid_features())


def test_source_feature_name_rejects_unknown_transformer_prefix():
    with pytest.raises(ValueError, match="Unknown transformed feature prefix"):
        source_feature_name("custom__Some_feature")


def test_car_features_rejects_unrealistic_mileage():
    payload = valid_features().model_dump(by_alias=True)
    payload["Mileage_km"] = config.MAX_MILEAGE_KM + 1

    with pytest.raises(ValidationError, match="Mileage_km"):
        CarFeatures(**payload)


@pytest.mark.parametrize("power_hp", [config.MIN_POWER_HP - 1, config.MAX_POWER_HP + 1])
def test_car_features_rejects_unreliable_power(power_hp: int):
    payload = valid_features().model_dump(by_alias=True)
    payload["Power_HP"] = power_hp

    with pytest.raises(ValidationError, match="Power_HP"):
        CarFeatures(**payload)


@pytest.mark.parametrize(
    "displacement_cm3",
    [config.MIN_DISPLACEMENT_CM3 - 1, config.MAX_DISPLACEMENT_CM3 + 1],
)
def test_car_features_rejects_unreliable_displacement(displacement_cm3: int):
    payload = valid_features().model_dump(by_alias=True)
    payload["Displacement_cm3"] = displacement_cm3

    with pytest.raises(ValidationError, match="Displacement_cm3"):
        CarFeatures(**payload)


@pytest.mark.parametrize(
    "doors_number",
    [config.MIN_DOORS_NUMBER - 1, config.MAX_DOORS_NUMBER + 1],
)
def test_car_features_rejects_unreliable_doors(doors_number: int):
    payload = valid_features().model_dump(by_alias=True)
    payload["Doors_number"] = doors_number

    with pytest.raises(ValidationError, match="Doors_number"):
        CarFeatures(**payload)
