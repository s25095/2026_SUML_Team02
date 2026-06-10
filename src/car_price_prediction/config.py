"""Central project configuration and shared column/path constants."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import math
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"


class AppSettings(BaseSettings):
    """Environment-backed runtime settings loaded from `.env` and OS env vars."""

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_reload: bool = True

    kaggle_dataset_id: str = "bartoszpieniak/poland-cars-for-sale-dataset"
    kaggle_api_token: SecretStr | None = None

    raw_data_filename: str = "Car_sale_ads.csv"
    processed_data_filename: str = "car_prices_clean.csv"
    model_filename: str = "car_price_model.joblib"
    model_metadata_filename: str = "model_metadata.json"
    training_metrics_filename: str = "training_metrics.json"
    feature_options_filename: str = "feature_options.json"

    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)

    def kaggle_env(self) -> dict[str, str]:
        """Return environment variables expected by the Kaggle Python client."""

        env: dict[str, str] = {}
        if self.kaggle_api_token:
            env["KAGGLE_API_TOKEN"] = self.kaggle_api_token.get_secret_value()
        return env

    def kaggle_access_token_files(self) -> tuple[Path, Path]:
        """Return default token-file locations supported by Kaggle access tokens."""

        kaggle_dir = Path.home() / ".kaggle"
        return kaggle_dir / "access_token", kaggle_dir / "access_token.txt"

    def has_kaggle_credentials(self) -> bool:
        """Check whether Kaggle credentials are available without exposing secrets."""

        return bool(self.kaggle_api_token) or any(
            path.exists() for path in self.kaggle_access_token_files()
        )


settings = AppSettings()

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
LOGS_DIR = REPO_ROOT / "logs"
LOG_FILE_PATH = LOGS_DIR / "app.log"

APP_HOST = settings.app_host
APP_PORT = settings.app_port
APP_RELOAD = settings.app_reload

KAGGLE_DATASET_ID = settings.kaggle_dataset_id
RAW_DATA_FILENAME = settings.raw_data_filename
PROCESSED_DATA_FILENAME = settings.processed_data_filename

RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
MODEL_PATH = MODELS_DIR / settings.model_filename
MODEL_METADATA_PATH = MODELS_DIR / settings.model_metadata_filename
TRAINING_METRICS_PATH = MODELS_DIR / settings.training_metrics_filename
FEATURE_OPTIONS_PATH = MODELS_DIR / settings.feature_options_filename

TARGET_COLUMN = "Price"
CURRENCY_COLUMN = "Currency"
CURRENCY_TO_KEEP = "PLN"
PRODUCTION_YEAR_COLUMN = "Production_year"
VEHICLE_AGE_COLUMN = "Vehicle_age_years"

SOURCE_FEATURE_COLUMNS = [
    "Condition",
    "Vehicle_brand",
    PRODUCTION_YEAR_COLUMN,
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Fuel_type",
    "Drive",
    "Transmission",
    "Type",
    "Doors_number",
]

FEATURE_COLUMNS = [
    "Condition",
    "Vehicle_brand",
    VEHICLE_AGE_COLUMN,
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Fuel_type",
    "Drive",
    "Transmission",
    "Type",
    "Doors_number",
]

SOURCE_MODEL_COLUMNS = [TARGET_COLUMN, *SOURCE_FEATURE_COLUMNS]
MODEL_COLUMNS = [TARGET_COLUMN, *FEATURE_COLUMNS]
PROCESSED_COLUMNS = [TARGET_COLUMN, PRODUCTION_YEAR_COLUMN, *FEATURE_COLUMNS]

DEFAULT_FEATURE_VALUES: dict[str, object] = {
    "Condition": "Used",
    "Vehicle_brand": "Toyota",
    PRODUCTION_YEAR_COLUMN: 2018,
    "Mileage_km": 120000,
    "Power_HP": 150,
    "Displacement_cm3": 1998,
    "Fuel_type": "Gasoline",
    "Drive": "Front wheels",
    "Transmission": "Manual",
    "Type": "SUV",
    "Doors_number": 5,
}

NUMERIC_FEATURE_COLUMNS = [
    VEHICLE_AGE_COLUMN,
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Doors_number",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "Condition",
    "Vehicle_brand",
    "Fuel_type",
    "Drive",
    "Transmission",
    "Type",
]

NUMERIC_SOURCE_COLUMNS = [
    TARGET_COLUMN,
    PRODUCTION_YEAR_COLUMN,
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Doors_number",
]

MIN_PRODUCTION_YEAR = 1900
MAX_PRODUCTION_YEAR = date.today().year
MIN_MILEAGE_KM = 0
MAX_MILEAGE_KM = 1_000_000
MIN_POWER_HP = 2
MAX_POWER_HP = 900
MIN_DISPLACEMENT_CM3 = 1
MAX_DISPLACEMENT_CM3 = 9_000
MIN_DOORS_NUMBER = 1
MAX_DOORS_NUMBER = 6
RECENT_CAR_MAX_AGE_YEARS = 5
MIN_RECENT_CAR_PRICE_PLN = 1_000
MAX_TARGET_PRICE_PLN = 3_000_000
RANDOM_STATE = settings.random_state
TEST_SIZE = settings.test_size


def infer_vehicle_age_reference_year(
    production_years: Iterable[object],
    default: int | None = None,
) -> int:
    """Infer the reference year used to convert production year into car age."""

    valid_years: list[int] = []
    for value in production_years:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(numeric_value):
            continue

        year = int(numeric_value)
        if MIN_PRODUCTION_YEAR <= year <= MAX_PRODUCTION_YEAR:
            valid_years.append(year)

    if valid_years:
        return max(valid_years)

    if default is not None:
        return default

    raise ValueError("Cannot infer vehicle age reference year from the dataset.")


def ensure_project_directories() -> None:
    """Create all runtime directories used by data, model, notebook and log flows."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
