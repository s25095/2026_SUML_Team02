"""Build and load categorical form options from the processed training data."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from car_price_prediction import config


SELECT_FEATURE_COLUMNS = [*config.CATEGORICAL_FEATURE_COLUMNS, "Doors_number"]

DISPLAY_ORDER: dict[str, list[str]] = {
    "Condition": ["Used", "New"],
    "Fuel_type": [
        "Gasoline",
        "Diesel",
        "Gasoline + LPG",
        "Hybrid",
        "Electric",
        "Gasoline + CNG",
        "Hydrogen",
        "Ethanol",
    ],
    "Drive": [
        "Front wheels",
        "Rear wheels",
        "4x4 (permanent)",
        "4x4 (attached automatically)",
        "4x4 (attached manually)",
    ],
    "Transmission": ["Manual", "Automatic"],
    "Type": [
        "SUV",
        "station_wagon",
        "sedan",
        "compact",
        "city_cars",
        "minivan",
        "coupe",
        "small_cars",
        "convertible",
    ],
    "Doors_number": ["1", "2", "3", "4", "5", "6"],
}


def normalize_option_value(field_name: str, value: object) -> str | None:
    """Normalize raw dataset values into stable strings used by HTML selects."""

    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"nan", "none", "<na>"}:
        return None

    if field_name == "Doors_number":
        try:
            return str(int(float(normalized)))
        except ValueError:
            return None

    return normalized


def ordered_values(field_name: str, counts: Counter[str]) -> list[str]:
    """Order options by curated display order, then by training frequency."""

    display_order = DISPLAY_ORDER.get(field_name)
    if display_order:
        ordered = [value for value in display_order if value in counts]
        remaining = sorted(
            (value for value in counts if value not in display_order),
            key=lambda value: (-counts[value], value.casefold()),
        )
        return [*ordered, *remaining]

    return sorted(
        counts,
        key=lambda value: (-counts[value], value.casefold()),
    )


def build_feature_options(data: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable dropdown artifact from processed training data."""

    missing_columns = sorted(set(SELECT_FEATURE_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Missing columns required for feature options: "
            f"{', '.join(missing_columns)}"
        )

    fields: dict[str, list[str]] = {}
    for field_name in SELECT_FEATURE_COLUMNS:
        counts: Counter[str] = Counter()
        for value in data[field_name]:
            normalized = normalize_option_value(field_name, value)
            if normalized is not None:
                counts[normalized] += 1

        if not counts:
            raise ValueError(f"No available options for field: {field_name}")

        fields[field_name] = ordered_values(field_name, counts)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "processed_training_data",
        "fields": fields,
    }


def save_feature_options(
    data: pd.DataFrame,
    output_path: Path = config.FEATURE_OPTIONS_PATH,
) -> dict[str, Any]:
    """Persist dropdown options used by the app at startup."""

    artifact = build_feature_options(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(artifact, output_file, ensure_ascii=False, indent=2)
    return artifact


def load_feature_options(
    input_path: Path = config.FEATURE_OPTIONS_PATH,
) -> dict[str, list[str]]:
    """Load and validate the required dropdown options artifact."""

    if not input_path.exists():
        raise FileNotFoundError(
            "Feature options artifact is missing. Run `uv run build-model` first."
        )

    with input_path.open("r", encoding="utf-8") as input_file:
        artifact = json.load(input_file)

    fields = artifact.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"Invalid feature options artifact: {input_path}")

    options: dict[str, list[str]] = {}
    for field_name in SELECT_FEATURE_COLUMNS:
        raw_values = fields.get(field_name)
        if not isinstance(raw_values, list):
            raise ValueError(f"Missing feature options for {field_name}: {input_path}")

        values = [
            value
            for raw_value in raw_values
            if (value := normalize_option_value(field_name, raw_value)) is not None
        ]
        if not values:
            raise ValueError(f"Empty feature options for {field_name}: {input_path}")
        options[field_name] = values

    return options
