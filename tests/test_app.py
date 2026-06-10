"""Tests for FastAPI web and API behavior."""

from __future__ import annotations

# pylint: disable=missing-function-docstring

from fastapi.testclient import TestClient

from car_price_prediction.app import api
from car_price_prediction.app import main
from car_price_prediction.schemas import PredictionExplanationItem, PredictionResponse


client = TestClient(main.app)


def stub_form_fields():
    return [
        {
            "name": "Condition",
            "label": "Stan",
            "default": "Used",
            "options": [
                {"value": "Used", "label": "Uzywany"},
                {"value": "New", "label": "Nowy"},
            ],
        },
        {
            "name": "Vehicle_brand",
            "label": "Marka",
            "default": "Toyota",
            "options": [
                {"value": "Toyota", "label": "Toyota"},
                {"value": "Honda", "label": "Honda"},
            ],
        },
        {
            "name": "Production_year",
            "label": "Rok produkcji",
            "type": "number",
            "placeholder": "2018",
            "default": 2018,
        },
    ]


def api_payload():
    # pylint: disable=duplicate-code
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


def test_index_renders_form(monkeypatch):
    monkeypatch.setattr(main, "form_fields", stub_form_fields)

    response = client.get("/")

    assert response.status_code == 200
    assert "Wycena auta" in response.text
    assert 'name="Vehicle_brand"' in response.text
    assert 'action="/form/predict"' in response.text
    assert '<select name="Condition"' in response.text
    assert '<select name="Vehicle_brand"' in response.text
    assert 'value="Toyota"' in response.text
    assert "selected" in response.text
    assert 'value="2018"' in response.text


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
    monkeypatch.setattr(main, "form_fields", stub_form_fields)

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
            base_value_pln=50000.0,
            explanation_method="lightgbm_pred_contrib",
            explanations=[
                PredictionExplanationItem(
                    feature_name="Vehicle_brand",
                    display_name="Marka",
                    feature_value="Toyota",
                    contribution_pln=2000.0,
                    direction="increases",
                )
            ],
            features=features,
        )

    monkeypatch.setattr(api.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/api/predict", json=api_payload())

    assert response.status_code == 200
    assert response.json()["predicted_price_pln"] == 52000.0
    assert response.json()["currency"] == "PLN"
    assert response.json()["model_name"] == "test_model"
    assert response.json()["vehicle_age_reference_year"] == 2021
    assert response.json()["explanation_method"] == "lightgbm_pred_contrib"
    assert response.json()["explanations"][0]["feature_name"] == "Vehicle_brand"
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
    monkeypatch.setattr(main, "form_fields", stub_form_fields)

    def fake_predict_price(features):
        return PredictionResponse(
            predicted_price_pln=52000.0,
            model_name="test_model",
            model_version="test-version",
            vehicle_age_reference_year=2021,
            base_value_pln=50000.0,
            explanation_method="lightgbm_pred_contrib",
            explanations=[
                PredictionExplanationItem(
                    feature_name="Vehicle_brand",
                    display_name="Marka",
                    feature_value="Toyota",
                    contribution_pln=2000.0,
                    direction="increases",
                )
            ],
            features=features,
        )

    monkeypatch.setattr(main.prediction_service, "predict_price", fake_predict_price)

    response = client.post("/form/predict", data=api_payload())

    assert response.status_code == 200
    assert "52 000 PLN" in response.text
    assert "Wplyw parametrow" in response.text
    assert "Marka" in response.text
    assert "+2 000 PLN" in response.text


def test_form_predict_get_redirects_to_index():
    response = client.get("/form/predict", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
