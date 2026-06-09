from __future__ import annotations

import json

import pandas as pd
import pytest

from car_price_prediction.app.forms import build_form_fields
from car_price_prediction.feature_options import (
    build_feature_options,
    load_feature_options,
    save_feature_options,
)


def feature_options_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Condition": "Used",
                "Vehicle_brand": "Toyota",
                "Fuel_type": "Gasoline",
                "Drive": "Front wheels",
                "Transmission": "Manual",
                "Type": "SUV",
                "Doors_number": 5.0,
            },
            {
                "Condition": "Used",
                "Vehicle_brand": "Toyota",
                "Fuel_type": "Gasoline",
                "Drive": "Front wheels",
                "Transmission": "Manual",
                "Type": "SUV",
                "Doors_number": 5.0,
            },
            {
                "Condition": "New",
                "Vehicle_brand": "Honda",
                "Fuel_type": "Electric",
                "Drive": "Rear wheels",
                "Transmission": "Automatic",
                "Type": "sedan",
                "Doors_number": 4.0,
            },
        ]
    )


def test_build_feature_options_uses_training_data_values():
    artifact = build_feature_options(feature_options_frame())

    assert artifact["source"] == "processed_training_data"
    assert artifact["fields"]["Vehicle_brand"] == ["Toyota", "Honda"]
    assert artifact["fields"]["Doors_number"] == ["4", "5"]


def test_save_and_load_feature_options_round_trip(tmp_path):
    output_path = tmp_path / "feature_options.json"

    save_feature_options(feature_options_frame(), output_path)
    options = load_feature_options(output_path)

    assert options["Condition"] == ["Used", "New"]
    assert options["Fuel_type"] == ["Gasoline", "Electric"]


def test_load_feature_options_requires_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="Feature options artifact is missing"):
        load_feature_options(tmp_path / "missing.json")


def test_load_feature_options_rejects_invalid_artifact(tmp_path):
    output_path = tmp_path / "feature_options.json"
    output_path.write_text(json.dumps({"fields": {"Condition": ["Used"]}}))

    with pytest.raises(ValueError, match="Missing feature options"):
        load_feature_options(output_path)


def test_build_form_fields_uses_loaded_select_options():
    fields = build_form_fields(
        {
            "Condition": ["Used"],
            "Vehicle_brand": ["Toyota", "Honda"],
            "Fuel_type": ["Gasoline", "Diesel"],
            "Drive": ["Front wheels"],
            "Transmission": ["Manual"],
            "Type": ["SUV"],
            "Doors_number": ["5"],
        }
    )

    brand_field = next(field for field in fields if field["name"] == "Vehicle_brand")
    fuel_field = next(field for field in fields if field["name"] == "Fuel_type")
    doors_field = next(field for field in fields if field["name"] == "Doors_number")

    assert brand_field["options"] == [
        {"value": "Toyota", "label": "Toyota"},
        {"value": "Honda", "label": "Honda"},
    ]
    assert {"value": "Diesel", "label": "Diesel"} in fuel_field["options"]
    assert doors_field["options"] == [{"value": "5", "label": "5 drzwi"}]


def test_build_form_fields_rejects_empty_select_options():
    with pytest.raises(ValueError, match="No options available for field: Condition"):
        build_form_fields(
            {
                "Condition": [],
                "Vehicle_brand": ["Toyota"],
                "Fuel_type": ["Gasoline"],
                "Drive": ["Front wheels"],
                "Transmission": ["Manual"],
                "Type": ["SUV"],
                "Doors_number": ["5"],
            }
        )


def test_build_form_fields_rejects_missing_select_options():
    with pytest.raises(KeyError, match="Missing options for select field: Drive"):
        build_form_fields(
            {
                "Condition": ["Used"],
                "Vehicle_brand": ["Toyota"],
                "Fuel_type": ["Gasoline"],
                "Transmission": ["Manual"],
                "Type": ["SUV"],
                "Doors_number": ["5"],
            }
        )
