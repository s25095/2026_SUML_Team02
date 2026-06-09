from __future__ import annotations

from typing import Any

from car_price_prediction import config


FORM_FIELDS: list[dict[str, Any]] = [
    {
        "name": "Condition",
        "label": "Stan",
        "type": "text",
        "placeholder": "Used",
    },
    {
        "name": "Vehicle_brand",
        "label": "Marka",
        "type": "text",
        "placeholder": "Toyota",
    },
    {
        "name": "Production_year",
        "label": "Rok produkcji",
        "type": "number",
        "placeholder": "2018",
        "min": config.MIN_PRODUCTION_YEAR,
        "max": config.MAX_PRODUCTION_YEAR,
    },
    {
        "name": "Mileage_km",
        "label": "Przebieg (km)",
        "type": "number",
        "placeholder": "125000",
        "min": config.MIN_MILEAGE_KM,
        "max": config.MAX_MILEAGE_KM,
    },
    {
        "name": "Power_HP",
        "label": "Moc (KM)",
        "type": "number",
        "placeholder": "150",
        "min": config.MIN_POWER_HP,
        "max": config.MAX_POWER_HP,
    },
    {
        "name": "Displacement_cm3",
        "label": "Pojemnosc (cm3)",
        "type": "number",
        "placeholder": "1998",
        "min": config.MIN_DISPLACEMENT_CM3,
        "max": config.MAX_DISPLACEMENT_CM3,
    },
    {
        "name": "Fuel_type",
        "label": "Paliwo",
        "type": "text",
        "placeholder": "Gasoline",
    },
    {
        "name": "Drive",
        "label": "Naped",
        "type": "text",
        "placeholder": "Front wheels",
    },
    {
        "name": "Transmission",
        "label": "Skrzynia",
        "type": "text",
        "placeholder": "Manual",
    },
    {
        "name": "Type",
        "label": "Typ nadwozia",
        "type": "text",
        "placeholder": "SUV",
    },
    {
        "name": "Doors_number",
        "label": "Liczba drzwi",
        "type": "number",
        "placeholder": "5",
        "min": config.MIN_DOORS_NUMBER,
        "max": config.MAX_DOORS_NUMBER,
    },
]
