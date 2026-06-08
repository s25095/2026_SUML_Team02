from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_price_prediction import config


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    estimator: BaseEstimator


@dataclass(frozen=True)
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float | None


def load_processed_data(
    input_path: Path = config.PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed data file does not exist: {input_path}")

    return pd.read_csv(input_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURE_COLUMNS),
        ]
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def default_model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate("dummy_median", DummyRegressor(strategy="median")),
        ModelCandidate("ridge", Ridge(alpha=1.0, random_state=config.RANDOM_STATE)),
        ModelCandidate(
            "random_forest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelCandidate(
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelCandidate(
            "lightgbm",
            LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=31,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = sorted(set(config.MODEL_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing model columns: {', '.join(missing_columns)}")

    return data[config.FEATURE_COLUMNS].copy(), data[config.TARGET_COLUMN].copy()


def split_train_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = split_features_target(data)
    return train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> ModelResult:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    r2 = float(r2_score(y_true, predictions))
    if not np.isfinite(r2):
        r2 = None

    return ModelResult(name="", rmse=rmse, mae=mae, r2=r2)


def evaluate_candidates(
    data: pd.DataFrame,
    candidates: list[ModelCandidate] | None = None,
) -> list[ModelResult]:
    candidates = candidates or default_model_candidates()
    x_train, x_test, y_train, y_test = split_train_test(data)
    results: list[ModelResult] = []

    for candidate in candidates:
        pipeline = build_pipeline(clone(candidate.estimator))
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = calculate_metrics(y_test, predictions)
        results.append(
            ModelResult(
                name=candidate.name,
                rmse=metrics.rmse,
                mae=metrics.mae,
                r2=metrics.r2,
            )
        )

    return sorted(results, key=lambda result: result.rmse)


def select_model(
    results: list[ModelResult],
    preferred_model_name: str = "lightgbm",
    baseline_model_name: str = "dummy_median",
    preferred_rmse_tolerance: float = 1.05,
) -> str:
    if not results:
        raise ValueError("At least one model result is required.")

    best = results[0]
    by_name = {result.name: result for result in results}
    preferred = by_name.get(preferred_model_name)
    baseline = by_name.get(baseline_model_name)

    if preferred and baseline:
        preferred_beats_baseline = preferred.rmse < baseline.rmse
        preferred_close_to_best = preferred.rmse <= best.rmse * preferred_rmse_tolerance
        if preferred_beats_baseline and preferred_close_to_best:
            return preferred.name

    return best.name


def train_final_pipeline(
    data: pd.DataFrame,
    selected_model_name: str,
    candidates: list[ModelCandidate] | None = None,
) -> Pipeline:
    candidates = candidates or default_model_candidates()
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    if selected_model_name not in candidate_by_name:
        raise ValueError(f"Unknown selected model: {selected_model_name}")

    features, target = split_features_target(data)
    pipeline = build_pipeline(clone(candidate_by_name[selected_model_name].estimator))
    pipeline.fit(features, target)
    return pipeline


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def results_to_dicts(results: list[ModelResult]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": result.name,
            "rmse": to_jsonable(result.rmse),
            "mae": to_jsonable(result.mae),
            "r2": to_jsonable(result.r2),
        }
        for result in results
    ]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def save_model_artifacts(
    pipeline: Pipeline,
    results: list[ModelResult],
    selected_model_name: str,
    rows_count: int,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.MODEL_METADATA_PATH,
    metrics_path: Path = config.TRAINING_METRICS_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    trained_at = datetime.now(UTC).isoformat()
    metrics = results_to_dicts(results)
    metadata = {
        "trained_at_utc": trained_at,
        "selected_model": selected_model_name,
        "selection_rule": (
            "Prefer lightgbm when it beats dummy_median and is within 5% "
            "of the best holdout RMSE; otherwise use the best RMSE model."
        ),
        "target_column": config.TARGET_COLUMN,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_feature_columns": config.NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": config.CATEGORICAL_FEATURE_COLUMNS,
        "training_rows": rows_count,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "model_path": display_path(model_path),
        "metrics_path": display_path(metrics_path),
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump({"models": metrics}, metrics_file, ensure_ascii=False, indent=2)


def train_and_save_model(
    data: pd.DataFrame,
    candidates: list[ModelCandidate] | None = None,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.MODEL_METADATA_PATH,
    metrics_path: Path = config.TRAINING_METRICS_PATH,
) -> str:
    candidates = candidates or default_model_candidates()
    results = evaluate_candidates(data, candidates=candidates)
    selected_model_name = select_model(results)
    pipeline = train_final_pipeline(
        data,
        selected_model_name=selected_model_name,
        candidates=candidates,
    )
    save_model_artifacts(
        pipeline,
        results,
        selected_model_name,
        rows_count=len(data),
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
    )
    return selected_model_name


def main() -> None:
    data = load_processed_data()
    selected_model_name = train_and_save_model(data)

    print(f"Selected model: {selected_model_name}")
    print(f"Saved model: {config.MODEL_PATH}")
    print(f"Saved metadata: {config.MODEL_METADATA_PATH}")
    print(f"Saved metrics: {config.TRAINING_METRICS_PATH}")


if __name__ == "__main__":
    main()
