from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from car_price_prediction import config
from car_price_prediction.data.download import download_dataset
from car_price_prediction.data.preprocessing import (
    load_raw_data,
    preprocess_data,
    save_processed_data,
)
from car_price_prediction.model.train import load_processed_data, train_and_save_model


@dataclass(frozen=True)
class PipelineResult:
    raw_data_path: Path
    processed_data_path: Path
    model_path: Path
    selected_model_name: str
    rows_count: int


def run_pipeline(force_download: bool = False) -> PipelineResult:
    config.ensure_project_directories()

    print("Step 1/3: downloading raw dataset")
    download_dataset(force=force_download)

    print("Step 2/3: preprocessing raw dataset")
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data)
    print(f"Saved processed data: {config.PROCESSED_DATA_PATH}")

    print("Step 3/3: training model from processed dataset")
    training_data = load_processed_data()
    selected_model_name = train_and_save_model(training_data)
    print(f"Selected model: {selected_model_name}")
    print(f"Saved model: {config.MODEL_PATH}")
    print(f"Saved metadata: {config.MODEL_METADATA_PATH}")
    print(f"Saved metrics: {config.TRAINING_METRICS_PATH}")
    print(f"Saved feature options: {config.FEATURE_OPTIONS_PATH}")

    return PipelineResult(
        raw_data_path=config.RAW_DATA_PATH,
        processed_data_path=config.PROCESSED_DATA_PATH,
        model_path=config.MODEL_PATH,
        selected_model_name=selected_model_name,
        rows_count=len(training_data),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, preprocess and train the car price model."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the Kaggle dataset again even if the raw CSV already exists.",
    )
    args = parser.parse_args()

    run_pipeline(force_download=args.force_download)


if __name__ == "__main__":
    main()
