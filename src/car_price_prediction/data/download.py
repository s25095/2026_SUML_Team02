from __future__ import annotations

import argparse
import os
from typing import Protocol

from car_price_prediction import config


class KaggleApiClient(Protocol):
    def authenticate(self) -> None:
        ...

    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
    ) -> None:
        ...


def apply_kaggle_environment() -> None:
    os.environ.update(config.settings.kaggle_env())


def build_kaggle_api() -> KaggleApiClient:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(force: bool = False) -> None:
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_DATA_PATH.exists() and not force:
        print(f"Raw dataset already exists: {config.RAW_DATA_PATH}")
        return

    if not config.settings.has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials are missing. Copy .env.example to .env and set "
            "KAGGLE_API_TOKEN."
        )

    apply_kaggle_environment()
    api = build_kaggle_api()

    try:
        api.dataset_download_files(
            config.KAGGLE_DATASET_ID,
            path=str(config.RAW_DATA_DIR),
            force=force,
            quiet=False,
            unzip=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Kaggle dataset download failed. Check KAGGLE_API_TOKEN in .env."
        ) from error

    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Kaggle download finished, but expected CSV was not found: "
            f"{config.RAW_DATA_PATH}"
        )

    print(f"Downloaded dataset to: {config.RAW_DATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle car dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the raw CSV already exists.",
    )
    args = parser.parse_args()

    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
