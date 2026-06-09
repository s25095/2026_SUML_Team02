"""End-to-end data preparation and model training pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from car_price_prediction import config
from car_price_prediction.data.download import download_dataset
from car_price_prediction.data.preprocessing import (
    load_raw_data,
    preprocess_data,
    save_processed_data,
)
from car_price_prediction.logging_config import setup_logging
from car_price_prediction.model.train import load_processed_data, train_and_save_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Summary of artifacts produced by a full pipeline run."""

    raw_data_path: Path
    processed_data_path: Path
    model_path: Path
    selected_model_name: str
    rows_count: int


def run_pipeline(force_download: bool = False) -> PipelineResult:
    """Download raw data, preprocess it, train the model and save artifacts."""

    config.ensure_project_directories()

    logger.info("Step 1/3: downloading raw dataset")
    download_dataset(force=force_download)

    logger.info("Step 2/3: preprocessing raw dataset")
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data)
    logger.info("Saved processed data: %s", config.PROCESSED_DATA_PATH)

    logger.info("Step 3/3: training model from processed dataset")
    training_data = load_processed_data()
    selected_model_name = train_and_save_model(training_data)
    logger.info("Selected model: %s", selected_model_name)
    logger.info("Saved model: %s", config.MODEL_PATH)
    logger.info("Saved metadata: %s", config.MODEL_METADATA_PATH)
    logger.info("Saved metrics: %s", config.TRAINING_METRICS_PATH)
    logger.info("Saved feature options: %s", config.FEATURE_OPTIONS_PATH)

    return PipelineResult(
        raw_data_path=config.RAW_DATA_PATH,
        processed_data_path=config.PROCESSED_DATA_PATH,
        model_path=config.MODEL_PATH,
        selected_model_name=selected_model_name,
        rows_count=len(training_data),
    )


def main() -> None:
    """CLI entrypoint for `uv run build-model`."""

    setup_logging()
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
