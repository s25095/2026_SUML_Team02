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
        "min": 0,
    },
    {
        "name": "Power_HP",
        "label": "Moc (KM)",
        "type": "number",
        "placeholder": "150",
        "min": 1,
    },
    {
        "name": "Displacement_cm3",
        "label": "Pojemnosc (cm3)",
        "type": "number",
        "placeholder": "1998",
        "min": 1,
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
        "min": 1,
        "max": 6,
    },
]
