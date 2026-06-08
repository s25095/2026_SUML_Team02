from pathlib import Path
import json

import pandas as pd
from autogluon.tabular import TabularPredictor


PROCESSED_DATA_PATH = Path("data/processed/car_prices_clean.csv")
MODEL_PATH = Path("models/autogluon_car_price")
METRICS_PATH = Path("models/training_metrics.json")
LEADERBOARD_PATH = Path("models/training_leaderboard.csv")

TARGET_COLUMN = "Price"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8
TIME_LIMIT_SECONDS = 600
QUALITY_PRESET = "medium_quality"

# Wczytuje przetworzone dane z pliku CSV
# Zwraca błąd, jeśli plik nie istnieje
def load_processed_data():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data file does not exist: {PROCESSED_DATA_PATH}"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)

# Dzieli dane na zestawy treningowy i testowy
def split_train_test(data):
    """Split data into training and test sets."""
    train_data = data.sample(frac=TRAIN_FRACTION, random_state=RANDOM_STATE)
    test_data = data.drop(train_data.index)

    return train_data, test_data

# Trenuje model i zapisuje go na dysku
def train_model(train_data):
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        path=str(MODEL_PATH),
        problem_type="regression",
        eval_metric="root_mean_squared_error",
    )

    return predictor.fit(
        train_data=train_data,
        presets=QUALITY_PRESET,
        time_limit=TIME_LIMIT_SECONDS,
    )


# Przygotowuje metryki do zapisu w pliku JSON
def prepare_metrics_for_json(metrics):
    prepared_metrics = {}

    for name, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        prepared_metrics[name] = value

    return prepared_metrics

# Zapisuje metryki i leaderboard do plików
def save_training_report(predictor, test_data):
    metrics = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=True)
    prepared_metrics = prepare_metrics_for_json(metrics)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(prepared_metrics, metrics_file, indent=4)

    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved leaderboard: {LEADERBOARD_PATH}")


def main():
    data = load_processed_data()

    train_data, test_data = split_train_test(data)

    print(f"Training rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Model output path: {MODEL_PATH}")

    predictor = train_model(train_data)
    save_training_report(predictor, test_data)

    print("AutoGluon model training finished.")


if __name__ == "__main__":
    main()