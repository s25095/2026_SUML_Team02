"""Tests for the end-to-end pipeline coordinator."""

from __future__ import annotations

# pylint: disable=missing-function-docstring

import pandas as pd

from car_price_prediction import pipeline


def test_run_pipeline_downloads_preprocesses_and_trains(monkeypatch, tmp_path):
    calls = []
    raw_data = pd.DataFrame({"raw": [1, 2]})
    processed_data = pd.DataFrame({"processed": [1]})
    training_data = pd.DataFrame({"processed": [1], "from_disk": [True]})

    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"

    monkeypatch.setattr(pipeline.config, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(pipeline.config, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(pipeline.config, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        pipeline.config, "MODEL_METADATA_PATH", tmp_path / "metadata.json"
    )
    monkeypatch.setattr(
        pipeline.config, "TRAINING_METRICS_PATH", tmp_path / "metrics.json"
    )
    monkeypatch.setattr(
        pipeline.config,
        "ensure_project_directories",
        lambda: calls.append(("ensure_dirs", None)),
    )
    monkeypatch.setattr(
        pipeline,
        "download_dataset",
        lambda force=False: calls.append(("download", force)),
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_data",
        lambda: calls.append(("load_raw", None)) or raw_data,
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_data",
        lambda data: calls.append(("preprocess", data)) or processed_data,
    )
    monkeypatch.setattr(
        pipeline,
        "save_processed_data",
        lambda data: calls.append(("save_processed", data)),
    )
    monkeypatch.setattr(
        pipeline,
        "load_processed_data",
        lambda: calls.append(("load_processed", None)) or training_data,
    )
    monkeypatch.setattr(
        pipeline,
        "train_and_save_model",
        lambda data: calls.append(("train", data)) or "ridge",
    )

    result = pipeline.run_pipeline(force_download=True)

    assert calls == [
        ("ensure_dirs", None),
        ("download", True),
        ("load_raw", None),
        ("preprocess", raw_data),
        ("save_processed", processed_data),
        ("load_processed", None),
        ("train", training_data),
    ]
    assert result.raw_data_path == raw_path
    assert result.processed_data_path == processed_path
    assert result.model_path == model_path
    assert result.selected_model_name == "ridge"
    assert result.rows_count == 1
