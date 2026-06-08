from __future__ import annotations

from fastapi.testclient import TestClient

from car_price_prediction.app import main
from car_price_prediction.model.predict import PredictionResponse


client = TestClient(main.app)


def api_payload():
    return {
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_available" in response.json()


def test_index_renders_form():
    response = client.get("/")

    assert response.status_code == 200
    assert "Wycena auta" in response.text
    assert 'name="Vehicle_brand"' in response.text


def test_api_predict_uses_prediction_service(monkeypatch):
    def fake_predict_price(features):
        return PredictionResponse(
            predicted_price_pln=52000.0,
            model_name="test_model",
            model_version="test-version",
            features=features.model_dump(by_alias=True),
        )

    monkeypatch.setattr(main.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/api/predict", json=api_payload())

    assert response.status_code == 200
    assert response.json()["predicted_price_pln"] == 52000.0
    assert response.json()["model_name"] == "test_model"


def test_form_predict_uses_prediction_service(monkeypatch):
    def fake_predict_price(features):
        return PredictionResponse(
            predicted_price_pln=52000.0,
            model_name="test_model",
            model_version="test-version",
            features=features.model_dump(by_alias=True),
        )

    monkeypatch.setattr(main.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/predict", data=api_payload())

    assert response.status_code == 200
    assert "52 000 PLN" in response.text
