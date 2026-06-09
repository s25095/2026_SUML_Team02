from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from car_price_prediction.app import bootstrap


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def configure_paths(monkeypatch, tmp_path):
    processed_path = tmp_path / "data" / "processed" / "cars.csv"
    artifact_paths = {
        "MODEL_PATH": tmp_path / "models" / "model.joblib",
        "MODEL_METADATA_PATH": tmp_path / "models" / "metadata.json",
        "TRAINING_METRICS_PATH": tmp_path / "models" / "metrics.json",
        "FEATURE_OPTIONS_PATH": tmp_path / "models" / "feature_options.json",
    }

    monkeypatch.setattr(bootstrap.config, "PROCESSED_DATA_PATH", processed_path)
    for name, path in artifact_paths.items():
        monkeypatch.setattr(bootstrap.config, name, path)
    monkeypatch.setattr(bootstrap.config, "ensure_project_directories", lambda: None)

    return processed_path, tuple(artifact_paths.values())


def test_bootstrap_artifacts_ready_when_processed_data_and_model_exist(
    monkeypatch,
    tmp_path,
):
    processed_path, artifact_paths = configure_paths(monkeypatch, tmp_path)
    touch(processed_path)
    for path in artifact_paths:
        touch(path)

    result = bootstrap.bootstrap_artifacts()

    assert result.action == "ready"
    assert result.missing_artifacts == ()


def test_bootstrap_artifacts_runs_full_pipeline_when_processed_data_is_missing(
    monkeypatch,
    tmp_path,
):
    configure_paths(monkeypatch, tmp_path)
    calls = []

    def fake_run_pipeline(force_download: bool):
        calls.append(force_download)
        return SimpleNamespace(
            raw_data_path=tmp_path / "raw.csv",
            processed_data_path=tmp_path / "processed.csv",
            model_path=tmp_path / "model.joblib",
            selected_model_name="lightgbm",
            rows_count=10,
        )

    monkeypatch.setattr(bootstrap, "run_pipeline", fake_run_pipeline)

    result = bootstrap.bootstrap_artifacts()

    assert result.action == "pipeline"
    assert result.selected_model_name == "lightgbm"
    assert calls == [False]


def test_bootstrap_artifacts_trains_when_model_artifact_is_missing(
    monkeypatch,
    tmp_path,
):
    processed_path, artifact_paths = configure_paths(monkeypatch, tmp_path)
    touch(processed_path)
    for path in artifact_paths[1:]:
        touch(path)
    calls = []

    monkeypatch.setattr(bootstrap, "load_processed_data", lambda: "training-data")

    def fake_train_and_save_model(data):
        calls.append(data)
        return "lightgbm"

    monkeypatch.setattr(bootstrap, "train_and_save_model", fake_train_and_save_model)

    result = bootstrap.bootstrap_artifacts()

    assert result.action == "trained"
    assert result.selected_model_name == "lightgbm"
    assert result.missing_artifacts == (artifact_paths[0],)
    assert calls == ["training-data"]
