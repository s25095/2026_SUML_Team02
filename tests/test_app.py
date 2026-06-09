from __future__ import annotations

from fastapi.testclient import TestClient

from car_price_prediction.app import api
from car_price_prediction.app import main
from car_price_prediction.schemas import PredictionResponse


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
    assert 'action="/form/predict"' in response.text


def test_startup_warms_model_cache_when_model_exists(monkeypatch):
    calls = []

    def fake_warm_model_bundle():
        calls.append("warm")
        return True

    monkeypatch.setattr(
        main.prediction_service,
        "warm_model_bundle",
        fake_warm_model_bundle,
    )

    with TestClient(main.app):
        pass

    assert calls == ["warm"]


def test_api_predict_uses_prediction_service(monkeypatch):
    def fake_predict_price(features):
        return PredictionResponse(
            predicted_price_pln=52000.0,
            model_name="test_model",
            model_version="test-version",
            vehicle_age_reference_year=2021,
            features=features,
        )

    monkeypatch.setattr(api.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/api/predict", json=api_payload())

    assert response.status_code == 200
    assert response.json()["predicted_price_pln"] == 52000.0
    assert response.json()["currency"] == "PLN"
    assert response.json()["model_name"] == "test_model"
    assert response.json()["vehicle_age_reference_year"] == 2021
    assert response.json()["features"]["Production_year"] == 2018


def test_api_predict_rejects_unknown_fields():
    payload = api_payload()
    payload["Unknown_feature"] = "bad"

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_api_predict_returns_503_when_model_is_missing(monkeypatch):
    def fake_predict_price(_features):
        raise FileNotFoundError("Trained model is missing.")

    monkeypatch.setattr(api.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/api/predict", json=api_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Trained model is missing."


def test_form_predict_uses_prediction_service(monkeypatch):
    def fake_predict_price(features):
        return PredictionResponse(
            predicted_price_pln=52000.0,
            model_name="test_model",
            model_version="test-version",
            vehicle_age_reference_year=2021,
            features=features,
        )

    monkeypatch.setattr(main.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/form/predict", data=api_payload())

    assert response.status_code == 200
    assert "52 000 PLN" in response.text
