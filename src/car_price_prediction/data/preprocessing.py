from __future__ import annotations

from pathlib import Path

import pandas as pd

from car_price_prediction import config


def validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")


def load_raw_data(input_path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    return pd.read_csv(input_path)


def filter_pln_offers(data: pd.DataFrame) -> pd.DataFrame:
    return data[data[config.CURRENCY_COLUMN] == config.CURRENCY_TO_KEEP].copy()


def select_model_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, config.MODEL_COLUMNS].copy()


def clean_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()

    for column in config.NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")

    return cleaned_data


def clean_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()

    for column in config.CATEGORICAL_FEATURE_COLUMNS:
        cleaned_data[column] = cleaned_data[column].astype("string").str.strip()
        cleaned_data[column] = cleaned_data[column].replace("", pd.NA)

    return cleaned_data


def remove_invalid_rows(data: pd.DataFrame) -> pd.DataFrame:
    valid_rows = data[config.TARGET_COLUMN].notna()
    valid_rows &= data[config.TARGET_COLUMN] > 0
    valid_rows &= data["Production_year"] >= config.MIN_PRODUCTION_YEAR
    valid_rows &= data["Production_year"] <= config.MAX_PRODUCTION_YEAR
    valid_rows &= data["Mileage_km"].isna() | (data["Mileage_km"] >= 0)
    valid_rows &= data["Power_HP"].isna() | (data["Power_HP"] > 0)
    valid_rows &= data["Displacement_cm3"].isna() | (
        data["Displacement_cm3"] > 0
    )
    valid_rows &= data["Doors_number"].isna() | (
        (data["Doors_number"] >= 1) & (data["Doors_number"] <= 6)
    )

    return data.loc[valid_rows].copy()


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = [config.CURRENCY_COLUMN, *config.MODEL_COLUMNS]
    validate_columns(data, required_columns)

    processed_data = filter_pln_offers(data)
    processed_data = select_model_columns(processed_data)
    processed_data = clean_numeric_columns(processed_data)
    processed_data = clean_categorical_columns(processed_data)
    processed_data = remove_invalid_rows(processed_data)

    return processed_data.reset_index(drop=True)


def save_processed_data(
    data: pd.DataFrame,
    output_path: Path = config.PROCESSED_DATA_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main() -> None:
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data)

    print(
        "Saved processed data: "
        f"{config.PROCESSED_DATA_PATH} ({len(processed_data)} rows)"
    )


if __name__ == "__main__":
    main()
