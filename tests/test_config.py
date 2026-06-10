"""Tests for runtime settings and derived config values."""

from __future__ import annotations

# pylint: disable=missing-function-docstring

import pytest

from car_price_prediction.config import AppSettings, infer_vehicle_age_reference_year


def test_settings_reads_dotenv_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "KAGGLE_API_TOKEN=test-token",
                "KAGGLE_DATASET_ID=owner/custom-dataset",
                "APP_PORT=8010",
                "APP_RELOAD=false",
                "RANDOM_STATE=123",
                "TEST_SIZE=0.25",
            ]
        ),
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_file)

    assert settings.kaggle_api_token is not None
    assert settings.kaggle_api_token.get_secret_value() == "test-token"
    assert settings.kaggle_dataset_id == "owner/custom-dataset"
    assert settings.app_port == 8010
    assert settings.app_reload is False
    assert settings.random_state == 123
    assert settings.test_size == 0.25


def test_settings_supports_kaggle_access_token():
    settings = AppSettings(kaggle_api_token="test-token")

    assert settings.has_kaggle_credentials()
    assert settings.kaggle_env() == {"KAGGLE_API_TOKEN": "test-token"}


def test_infer_vehicle_age_reference_year_ignores_invalid_values():
    assert infer_vehicle_age_reference_year([None, "not-a-year", 2019, 2021]) == 2021


def test_infer_vehicle_age_reference_year_uses_default_when_requested():
    assert infer_vehicle_age_reference_year(["bad"], default=2026) == 2026


def test_infer_vehicle_age_reference_year_raises_without_valid_values():
    with pytest.raises(ValueError, match="Cannot infer"):
        infer_vehicle_age_reference_year(["bad"])
