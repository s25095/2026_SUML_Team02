from __future__ import annotations

from pathlib import Path

import numpy as np
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
    return data.loc[:, config.SOURCE_MODEL_COLUMNS].copy()


def clean_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()

    for column in config.NUMERIC_SOURCE_COLUMNS:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")

    return cleaned_data


def add_vehicle_age_feature(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    reference_year = config.infer_vehicle_age_reference_year(
        cleaned_data[config.PRODUCTION_YEAR_COLUMN]
    )
    cleaned_data[config.VEHICLE_AGE_COLUMN] = (
        reference_year - cleaned_data[config.PRODUCTION_YEAR_COLUMN]
    )
    return cleaned_data


def nullify_unreliable_mileage(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    valid_mileage = cleaned_data["Mileage_km"].between(
        config.MIN_MILEAGE_KM,
        config.MAX_MILEAGE_KM,
    )
    unreliable_mileage = ~cleaned_data["Mileage_km"].isna() & ~valid_mileage
    cleaned_data.loc[unreliable_mileage, "Mileage_km"] = np.nan
    return cleaned_data


def nullify_unreliable_power(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    valid_power = cleaned_data["Power_HP"].between(
        config.MIN_POWER_HP,
        config.MAX_POWER_HP,
    )
    unreliable_power = ~cleaned_data["Power_HP"].isna() & ~valid_power
    cleaned_data.loc[unreliable_power, "Power_HP"] = np.nan
    return cleaned_data


def nullify_unreliable_displacement(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    valid_displacement = cleaned_data["Displacement_cm3"].between(
        config.MIN_DISPLACEMENT_CM3,
        config.MAX_DISPLACEMENT_CM3,
    )
    unreliable_displacement = (
        ~cleaned_data["Displacement_cm3"].isna() & ~valid_displacement
    )
    cleaned_data.loc[unreliable_displacement, "Displacement_cm3"] = np.nan
    return cleaned_data


def nullify_unreliable_doors(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    valid_doors = cleaned_data["Doors_number"].between(
        config.MIN_DOORS_NUMBER,
        config.MAX_DOORS_NUMBER,
    )
    unreliable_doors = ~cleaned_data["Doors_number"].isna() & ~valid_doors
    cleaned_data.loc[unreliable_doors, "Doors_number"] = np.nan
    return cleaned_data


def nullify_recent_low_prices(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    recent_low_price = (
        cleaned_data[config.VEHICLE_AGE_COLUMN].notna()
        & (cleaned_data[config.VEHICLE_AGE_COLUMN] < config.RECENT_CAR_MAX_AGE_YEARS)
        & cleaned_data[config.TARGET_COLUMN].notna()
        & (cleaned_data[config.TARGET_COLUMN] < config.MIN_RECENT_CAR_PRICE_PLN)
    )
    cleaned_data.loc[recent_low_price, config.TARGET_COLUMN] = np.nan
    return cleaned_data


def clean_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()

    for column in config.CATEGORICAL_FEATURE_COLUMNS:
        values = cleaned_data[column].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA)
        cleaned_data[column] = pd.Series(
            values.to_numpy(dtype=object, na_value=np.nan),
            index=cleaned_data.index,
        )

    return cleaned_data


def remove_invalid_rows(data: pd.DataFrame) -> pd.DataFrame:
    valid_rows = data[config.TARGET_COLUMN].notna()
    valid_rows &= data[config.TARGET_COLUMN] > 0
    valid_rows &= data[config.TARGET_COLUMN] <= config.MAX_TARGET_PRICE_PLN
    valid_rows &= data[config.PRODUCTION_YEAR_COLUMN] >= config.MIN_PRODUCTION_YEAR
    valid_rows &= data[config.PRODUCTION_YEAR_COLUMN] <= config.MAX_PRODUCTION_YEAR
    valid_rows &= data[config.VEHICLE_AGE_COLUMN] >= 0
    valid_rows &= data["Power_HP"].isna() | (data["Power_HP"] > 0)

    return data.loc[valid_rows].copy()


def drop_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates(subset=config.PROCESSED_COLUMNS).copy()


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = [config.CURRENCY_COLUMN, *config.SOURCE_MODEL_COLUMNS]
    validate_columns(data, required_columns)

    processed_data = filter_pln_offers(data)
    processed_data = select_model_columns(processed_data)
    processed_data = clean_numeric_columns(processed_data)
    processed_data = add_vehicle_age_feature(processed_data)
    processed_data = nullify_unreliable_mileage(processed_data)
    processed_data = nullify_unreliable_power(processed_data)
    processed_data = nullify_unreliable_displacement(processed_data)
    processed_data = nullify_unreliable_doors(processed_data)
    processed_data = nullify_recent_low_prices(processed_data)
    processed_data = clean_categorical_columns(processed_data)
    processed_data = remove_invalid_rows(processed_data)
    processed_data = processed_data.loc[:, config.PROCESSED_COLUMNS]
    processed_data = drop_duplicate_rows(processed_data)

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
