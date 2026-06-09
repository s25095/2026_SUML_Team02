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
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 1
    assert "Currency" not in processed.columns
    assert "Vehicle_age_years" in processed.columns
    assert processed.loc[0, "Price"] == 55000
    assert processed.loc[0, "Condition"] == "Used"
    assert processed.loc[0, "Vehicle_age_years"] == 0


def test_preprocess_nullifies_unreliable_mileage():
    data = pd.DataFrame(
        [
            valid_row(Price="56000", Mileage_km="1111111111"),
            valid_row(Price="57000", Mileage_km="-1"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 2
    assert processed["Mileage_km"].isna().all()
    assert processed["Price"].tolist() == [56000, 57000]


def test_preprocess_nullifies_unreliable_power():
    data = pd.DataFrame(
        [
            valid_row(Price="56000", Power_HP="1"),
            valid_row(Price="57000", Power_HP="901"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 2
    assert processed["Power_HP"].isna().all()
    assert processed["Price"].tolist() == [56000, 57000]


def test_preprocess_nullifies_unreliable_displacement():
    data = pd.DataFrame(
        [
            valid_row(Price="56000", Displacement_cm3="0"),
            valid_row(Price="57000", Displacement_cm3="9001"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 2
    assert processed["Displacement_cm3"].isna().all()
    assert processed["Price"].tolist() == [56000, 57000]


def test_preprocess_nullifies_unreliable_doors():
    data = pd.DataFrame(
        [
            valid_row(Price="56000", Doors_number="7"),
            valid_row(Price="57000", Doors_number="55"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 2
    assert processed["Doors_number"].isna().all()
    assert processed["Price"].tolist() == [56000, 57000]


def test_preprocess_removes_recent_low_price_rows():
    data = pd.DataFrame(
        [
            valid_row(Price="999", Production_year="2021"),
            valid_row(Price="999", Production_year="2010"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 1
    assert processed.loc[0, "Price"] == 999
    assert processed.loc[0, "Production_year"] == 2010
    assert processed.loc[0, "Vehicle_age_years"] == 11


def test_preprocess_removes_extreme_high_price_rows():
    data = pd.DataFrame(
        [
            valid_row(Price="3000000"),
            valid_row(Price="3000001"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 1
    assert processed.loc[0, "Price"] == 3000000


def test_preprocess_drops_exact_duplicate_rows_after_cleaning():
    data = pd.DataFrame(
        [
            valid_row(Price="56000"),
            valid_row(Price="56000"),
            valid_row(Price="57000"),
        ]
    )

    processed = preprocess_data(data)

    assert len(processed) == 2
    assert processed["Price"].tolist() == [56000, 57000]


def test_preprocess_uses_sklearn_compatible_missing_values():
    data = pd.DataFrame([valid_row(Drive=" ", Fuel_type=None)])

    processed = preprocess_data(data)

    assert pd.isna(processed.loc[0, "Drive"])
    assert pd.isna(processed.loc[0, "Fuel_type"])
    assert processed.loc[0, "Drive"] is not pd.NA
    assert processed.loc[0, "Fuel_type"] is not pd.NA


def test_preprocess_reports_missing_columns():
    data = pd.DataFrame([valid_row()])
    data = data.drop(columns=["Power_HP"])

    with pytest.raises(ValueError, match="Power_HP"):
        preprocess_data(data)
