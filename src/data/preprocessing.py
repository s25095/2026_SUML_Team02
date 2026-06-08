from datetime import date
from pathlib import Path
import pandas as pd


# Ścieżki do surowych i przetworzonych danych
RAW_DATA_PATH = Path("data/raw/Car_sale_ads.csv")
PROCESSED_DATA_PATH = Path("data/processed/car_prices_clean.csv")

# Nazwa docelowej kolumny i waluta do zachowania
TARGET_COLUMN = "Price"
CURRENCY_COLUMN = "Currency"
CURRENCY_TO_KEEP = "PLN"

# Kolumny, które będą przekazywane jako input do modelu
INPUT_COLUMNS = [
    "Condition",
    "Vehicle_brand",
    "Production_year",
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Fuel_type",
    "Drive",
    "Transmission",
    "Type",
    "Doors_number",
]

# Lista kolumn, które będą używane w modelu
MODEL_COLUMNS = [TARGET_COLUMN, *INPUT_COLUMNS]

# Kolumny numeryczne
NUMERIC_COLUMNS = [
    "Price",
    "Production_year",
    "Mileage_km",
    "Power_HP",
    "Displacement_cm3",
    "Doors_number",
]

# Kolumny kategoryczne
CATEGORICAL_COLUMNS = [
    "Condition",
    "Vehicle_brand",
    "Fuel_type",
    "Drive",
    "Transmission",
    "Type",
]

# Minimalny wiek auta i maksymalny do kontroli błędów w danych
MIN_PRODUCTION_YEAR = 1900
MAX_PRODUCTION_YEAR = date.today().year

# Sprawdza czy podane dane zawierają wymagane kolumny, 
# jeśli nie to wyrzuca błąd z brakującymi kolumnami
def validate_columns(data, required_columns):
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")

# Wczytuje surowe dane z pliku CSV, 
# jeśli plik nie istnieje to wyrzuca błąd
def load_raw_data(input_path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    return pd.read_csv(input_path)


# Filtruje oferty z cennikiem w PLN
def filter_pln_offers(data):
    return data[data[CURRENCY_COLUMN] == CURRENCY_TO_KEEP].copy()

# Wybiera tylko kolumny potrzebne do modelu
def select_model_columns(data):
    return data.loc[:, MODEL_COLUMNS].copy()

# Konwertuje kolumny numeryczne na liczby, 
# błędne wartości zamienia na NaN
def clean_numeric_columns(data):
    cleaned_data = data.copy()

    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

    return cleaned_data


# Czyści kolumny kategoryczne, 
# usuwa białe znaki i spacje,
# zamienia puste wartości na NaN
def clean_categorical_columns(data):
    cleaned_data = data.copy()

    for column in CATEGORICAL_COLUMNS:
        cleaned_data[column] = cleaned_data[column].astype("string").str.strip()
        cleaned_data[column] = cleaned_data[column].replace("", pd.NA)

    return cleaned_data

# Usuwa wiersze z nieprawidłowymi wartościami i błędami w danych
def remove_invalid_rows(data):
    # Puste lub ujemne ceny
    valid_rows = data[TARGET_COLUMN].notna()
    valid_rows &= data[TARGET_COLUMN] > 0

    # Nierealistyczne lata produkcji
    valid_rows &= data["Production_year"].between(
        MIN_PRODUCTION_YEAR,
        MAX_PRODUCTION_YEAR,
        inclusive="both",
    )

    # Ujemny przebieg, ujemna moc, ujemna pojemność
    valid_rows &= data["Mileage_km"].isna() | (data["Mileage_km"] >= 0)
    valid_rows &= data["Power_HP"].isna() | (data["Power_HP"] > 0)
    valid_rows &= (
        data["Displacement_cm3"].isna() | (data["Displacement_cm3"] > 0)
    )

    # Nierealistyczna liczba drzwi
    valid_rows &= data["Doors_number"].isna() | (
        data["Doors_number"].between(1, 6, inclusive="both")
    )

    return data.loc[valid_rows].copy()

# Główna funkcja przetwarzania danych
def preprocess_data(data):

    # Walidacja wymaganych kolumn
    required_columns = [CURRENCY_COLUMN, *MODEL_COLUMNS]
    validate_columns(data, required_columns)

    # Filtracja, wybór kolumn, czyszczenie danych i usuwanie błędnych wierszy
    processed_data = filter_pln_offers(data)
    processed_data = select_model_columns(processed_data)
    processed_data = clean_numeric_columns(processed_data)
    processed_data = clean_categorical_columns(processed_data)
    processed_data = remove_invalid_rows(processed_data)

    return processed_data.reset_index(drop=True)

# Zapisuje przetworzone dane do pliku CSV
def save_processed_data(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main():
    raw_data = load_raw_data(RAW_DATA_PATH)
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data, PROCESSED_DATA_PATH)

    print(
        "Saved processed data: "
        f"{PROCESSED_DATA_PATH} ({len(processed_data)} rows)"
    )


if __name__ == "__main__":
    main()
