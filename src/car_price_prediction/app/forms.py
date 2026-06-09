from __future__ import annotations

from functools import lru_cache
from typing import Any

from car_price_prediction import config
from car_price_prediction.feature_options import load_feature_options


DEFAULT_VALUES: dict[str, Any] = {
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

OPTION_LABELS: dict[str, dict[str, str]] = {
    "Condition": {
        "Used": "Uzywany",
        "New": "Nowy",
    },
    "Fuel_type": {
        "Gasoline": "Benzyna",
        "Diesel": "Diesel",
        "Gasoline + LPG": "Benzyna + LPG",
        "Hybrid": "Hybryda",
        "Electric": "Elektryczny",
        "Gasoline + CNG": "Benzyna + CNG",
        "Hydrogen": "Wodor",
        "Ethanol": "Etanol",
    },
    "Drive": {
        "Front wheels": "Przedni",
        "Rear wheels": "Tylny",
        "4x4 (permanent)": "4x4 staly",
        "4x4 (attached automatically)": "4x4 automatyczny",
        "4x4 (attached manually)": "4x4 dolaczany recznie",
    },
    "Transmission": {
        "Manual": "Manualna",
        "Automatic": "Automatyczna",
    },
    "Type": {
        "station_wagon": "Kombi",
        "SUV": "SUV",
        "sedan": "Sedan",
        "compact": "Kompakt",
        "city_cars": "Auto miejskie",
        "minivan": "Minivan",
        "coupe": "Coupe",
        "small_cars": "Male auto",
        "convertible": "Kabriolet",
    },
    "Doors_number": {
        "1": "1 drzwi",
        "2": "2 drzwi",
        "3": "3 drzwi",
        "4": "4 drzwi",
        "5": "5 drzwi",
        "6": "6 drzwi",
    },
}


def option(field_name: str, value: str) -> dict[str, str]:
    return {
        "value": value,
        "label": OPTION_LABELS.get(field_name, {}).get(value, value),
    }


def default_value(field_name: str, options: list[str]) -> str:
    if not options:
        raise ValueError(f"No options available for field: {field_name}")

    preferred_value = str(DEFAULT_VALUES[field_name])
    if preferred_value in options:
        return preferred_value

    return options[0]


def select_field(
    name: str,
    label: str,
    options_by_field: dict[str, list[str]],
) -> dict[str, Any]:
    if name not in options_by_field:
        raise KeyError(f"Missing options for select field: {name}")

    values = options_by_field[name]
    return {
        "name": name,
        "label": label,
        "default": default_value(name, values),
        "options": [option(name, value) for value in values],
    }


def number_field(
    name: str,
    label: str,
    placeholder: str,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "type": "number",
        "placeholder": placeholder,
        "default": DEFAULT_VALUES[name],
        "min": minimum,
        "max": maximum,
    }


def build_form_fields(
    options_by_field: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    options = options_by_field or load_feature_options()

    return [
        select_field("Condition", "Stan", options),
        select_field("Vehicle_brand", "Marka", options),
        number_field(
            "Production_year",
            "Rok produkcji",
            "2018",
            config.MIN_PRODUCTION_YEAR,
            config.MAX_PRODUCTION_YEAR,
        ),
        number_field(
            "Mileage_km",
            "Przebieg (km)",
            "120000",
            config.MIN_MILEAGE_KM,
            config.MAX_MILEAGE_KM,
        ),
        number_field(
            "Power_HP",
            "Moc (KM)",
            "150",
            config.MIN_POWER_HP,
            config.MAX_POWER_HP,
        ),
        number_field(
            "Displacement_cm3",
            "Pojemnosc (cm3)",
            "1998",
            config.MIN_DISPLACEMENT_CM3,
            config.MAX_DISPLACEMENT_CM3,
        ),
        select_field("Fuel_type", "Paliwo", options),
        select_field("Drive", "Naped", options),
        select_field("Transmission", "Skrzynia", options),
        select_field("Type", "Typ nadwozia", options),
        select_field("Doors_number", "Liczba drzwi", options),
    ]


@lru_cache(maxsize=1)
def form_fields() -> list[dict[str, Any]]:
    return build_form_fields()
