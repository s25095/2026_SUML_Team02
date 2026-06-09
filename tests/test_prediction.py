from __future__ import annotations

import pytest
from pydantic import ValidationError

from car_price_prediction import config
from car_price_prediction.model.predict import CarFeatures, features_to_frame


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
