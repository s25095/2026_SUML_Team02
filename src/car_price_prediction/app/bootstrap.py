from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from car_price_prediction import config
from car_price_prediction.logging_config import setup_logging
from car_price_prediction.model.train import load_processed_data, train_and_save_model
from car_price_prediction.pipeline import run_pipeline


logger = logging.getLogger(__name__)
BootstrapAction = Literal["ready", "pipeline", "trained"]


@dataclass(frozen=True)
class BootstrapResult:
    action: BootstrapAction
    missing_artifacts: tuple[Path, ...] = ()
    selected_model_name: str | None = None


def model_artifact_paths() -> tuple[Path, ...]:
    return (
        config.MODEL_PATH,
        config.MODEL_METADATA_PATH,
        config.TRAINING_METRICS_PATH,
        config.FEATURE_OPTIONS_PATH,
    )


def missing_model_artifacts() -> tuple[Path, ...]:
    return tuple(path for path in model_artifact_paths() if not path.exists())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def bootstrap_artifacts(
    force_download: bool = False,
    force_train: bool = False,
) -> BootstrapResult:
    config.ensure_project_directories()

    if force_download or not config.PROCESSED_DATA_PATH.exists():
        logger.info("Bootstrap: running full data/model pipeline")
        pipeline_result = run_pipeline(force_download=force_download)
        return BootstrapResult(
            action="pipeline",
            selected_model_name=pipeline_result.selected_model_name,
        )

    missing_artifacts = missing_model_artifacts()
    if force_train or missing_artifacts:
        if missing_artifacts:
            missing = ", ".join(display_path(path) for path in missing_artifacts)
            logger.info("Bootstrap: missing model artifacts: %s", missing)
        logger.info("Bootstrap: training model from existing processed dataset")
        training_data = load_processed_data()
        selected_model_name = train_and_save_model(training_data)
        logger.info("Selected model: %s", selected_model_name)
        logger.info("Saved model: %s", config.MODEL_PATH)
        logger.info("Saved metadata: %s", config.MODEL_METADATA_PATH)
        logger.info("Saved metrics: %s", config.TRAINING_METRICS_PATH)
        logger.info("Saved feature options: %s", config.FEATURE_OPTIONS_PATH)
        return BootstrapResult(
            action="trained",
            missing_artifacts=missing_artifacts,
            selected_model_name=selected_model_name,
        )

    logger.info("Bootstrap: processed data and model artifacts are ready")
    return BootstrapResult(action="ready")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare missing artifacts and start the FastAPI app."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Run the full pipeline and download the Kaggle dataset again.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train the model again even if model artifacts already exist.",
    )
    args = parser.parse_args()

    setup_logging()
    bootstrap_artifacts(
        force_download=args.force_download,
        force_train=args.force_train,
    )

    from car_price_prediction.app.main import serve

    serve()


if __name__ == "__main__":
    main()
