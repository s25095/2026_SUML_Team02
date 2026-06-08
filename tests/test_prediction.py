from __future__ import annotations

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
    frame = features_to_frame(valid_features())

    assert list(frame.columns) == [
        "Condition",
        "Vehicle_brand",
        "Production_year",
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
