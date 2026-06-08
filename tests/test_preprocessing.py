from __future__ import annotations

import pandas as pd
import pytest

from car_price_prediction.data.preprocessing import preprocess_data


def valid_row(**overrides):
    row = {
        "Currency": "PLN",
        "Price": "55000",
        "Condition": " Used ",
        "Vehicle_brand": "Toyota",
        "Production_year": "2018",
        "Mileage_km": "120000",
        "Power_HP": "150",
        "Displacement_cm3": "1998",
        "Fuel_type": "Gasoline",
        "Drive": "Front wheels",
        "Transmission": "Manual",
        "Type": "SUV",
        "Doors_number": "5",
    }
    row.update(overrides)
    return row


def test_preprocess_filters_currency_and_invalid_rows():
    data = pd.DataFrame(
        [
            valid_row(),
            valid_row(Currency="EUR"),
            valid_row(Price="-1"),
            valid_row(Production_year="1800"),
            valid_row(Doors_number="9"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 1
    assert "Currency" not in processed.columns
    assert processed.loc[0, "Price"] == 55000
    assert processed.loc[0, "Condition"] == "Used"


def test_preprocess_reports_missing_columns():
    data = pd.DataFrame([valid_row()])
    data = data.drop(columns=["Power_HP"])

    with pytest.raises(ValueError, match="Power_HP"):
        preprocess_data(data)
