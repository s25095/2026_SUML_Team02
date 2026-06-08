from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from car_price_prediction.model.train import ModelCandidate, train_and_save_model


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
    )

    assert selected in {"dummy_median", "ridge"}
    assert (tmp_path / "model.joblib").exists()

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert metadata["selected_model"] == selected
    assert metadata["feature_columns"]
    assert len(metrics["models"]) == 2
