from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from car_price_prediction.data.preprocessing import preprocess_data
from car_price_prediction.model.train import (
    ModelCandidate,
    aggregate_feature_importance,
    evaluate_candidates,
    model_feature_importance,
    train_and_save_model,
    train_final_pipeline,
)


def training_frame() -> pd.DataFrame:
    rows = []
    brands = ["Toyota", "Skoda", "Ford", "BMW", "Kia", "Opel"]
    for index in range(18):
        year = 2010 + index % 10
        mileage = 30000 + index * 8500
        power = 90 + index * 5
        rows.append(
            {
                "Price": 18000 + (year - 2010) * 3000 + power * 160 - mileage * 0.05,
                "Condition": "Used",
                "Vehicle_brand": brands[index % len(brands)],
                "Production_year": year,
                "Mileage_km": mileage,
                "Power_HP": power,
                "Displacement_cm3": 1400 + index * 60,
                "Fuel_type": "Gasoline" if index % 2 else "Diesel",
                "Drive": "Front wheels",
                "Transmission": "Manual" if index % 3 else "Automatic",
                "Type": "SUV" if index % 2 else "Combi",
                "Doors_number": 5,
            }
        )
    return pd.DataFrame(rows)


def raw_training_frame_with_missing_categories() -> pd.DataFrame:
    data = training_frame()
    data.insert(0, "Currency", "PLN")
    data.loc[0, "Drive"] = " "
    data.loc[1, "Fuel_type"] = None
    return data


def test_evaluate_candidates_accepts_direct_preprocessed_data():
    candidates = [
        ModelCandidate("dummy_median", DummyRegressor(strategy="median")),
        ModelCandidate("ridge", Ridge(alpha=1.0)),
    ]

    data = preprocess_data(raw_training_frame_with_missing_categories())
    results = evaluate_candidates(data, candidates=candidates)

    assert {result.name for result in results} == {"dummy_median", "ridge"}


def test_evaluate_candidates_accepts_pandas_na_categoricals():
    candidates = [
        ModelCandidate("dummy_median", DummyRegressor(strategy="median")),
        ModelCandidate("ridge", Ridge(alpha=1.0)),
    ]
    data = training_frame()
    data.loc[0, "Drive"] = pd.NA
    data.loc[1, "Fuel_type"] = pd.NA

    results = evaluate_candidates(data, candidates=candidates)

    assert {result.name for result in results} == {"dummy_median", "ridge"}


def test_train_and_save_model_creates_artifacts(tmp_path):
    candidates = [
        ModelCandidate("dummy_median", DummyRegressor(strategy="median")),
        ModelCandidate("ridge", Ridge(alpha=1.0)),
    ]

    selected = train_and_save_model(
        training_frame(),
        candidates=candidates,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
        metrics_path=tmp_path / "metrics.json",
        feature_options_path=tmp_path / "feature_options.json",
    )

    assert selected in {"dummy_median", "ridge"}
    assert (tmp_path / "model.joblib").exists()

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    feature_options = json.loads((tmp_path / "feature_options.json").read_text())

    assert metadata["selected_model"] == selected
    assert metadata["feature_columns"]
    assert "Vehicle_age_years" in metadata["feature_columns"]
    assert "Production_year" not in metadata["feature_columns"]
    assert metadata["vehicle_age_reference_year"] == 2019
    assert metadata["feature_options_path"] == str(tmp_path / "feature_options.json")
    assert len(metrics["models"]) == 2
    assert feature_options["fields"]["Vehicle_brand"]
    assert feature_options["fields"]["Doors_number"] == ["5"]


def test_feature_importance_maps_transformed_features_to_source_columns():
    candidates = [ModelCandidate("ridge", Ridge(alpha=1.0))]
    pipeline = train_final_pipeline(
        training_frame(),
        selected_model_name="ridge",
        candidates=candidates,
    )

    transformed_importance = model_feature_importance(pipeline)
    source_importance = aggregate_feature_importance(pipeline)

    assert not transformed_importance.empty
    assert not source_importance.empty
    assert "Vehicle_age_years" in set(source_importance["source_feature"])
    assert "Vehicle_brand" in set(source_importance["source_feature"])
    assert transformed_importance["importance_type"].eq("absolute_coefficient").all()
    assert source_importance["importance_share"].sum() == pytest.approx(1.0)
