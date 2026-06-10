"""Model selection, training and artifact persistence for car price regression."""

from __future__ import annotations

import json
import logging
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
from car_price_prediction.feature_options import save_feature_options
from car_price_prediction.logging_config import setup_logging
from car_price_prediction.model.feature_names import (
    source_feature_name,
    transformed_feature_names,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCandidate:
    """Named estimator candidate evaluated during model selection."""

    name: str
    estimator: BaseEstimator


@dataclass(frozen=True)
class ModelResult:
    """Holdout metrics for one evaluated model candidate."""

    name: str
    rmse: float
    mae: float
    r2: float | None


@dataclass(frozen=True)
class TrainingArtifactPaths:
    """Output paths written by model training."""

    model: Path = config.MODEL_PATH
    metadata: Path = config.MODEL_METADATA_PATH
    metrics: Path = config.TRAINING_METRICS_PATH
    feature_options: Path = config.FEATURE_OPTIONS_PATH


@dataclass(frozen=True)
class TrainingArtifactSummary:
    """Model-selection summary persisted next to the trained pipeline."""

    results: list[ModelResult]
    selected_model_name: str
    rows_count: int
    vehicle_age_reference_year: int


def load_processed_data(
    input_path: Path = config.PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load the processed training dataset from disk."""

    if not input_path.exists():
        raise FileNotFoundError(f"Processed data file does not exist: {input_path}")

    return pd.read_csv(input_path)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numeric scaling and categorical one-hot encoding."""

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
    """Wrap a regression estimator in the standard preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def default_model_candidates() -> list[ModelCandidate]:
    """Return baseline and production-grade regressors compared by training."""

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
        ModelCandidate(
            "lightgbm_xt_autogluon_inspired",
            LGBMRegressor(
                n_estimators=2180,
                learning_rate=0.05,
                num_leaves=31,
                extra_trees=True,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        ModelCandidate(
            "lightgbm_large_autogluon_inspired",
            LGBMRegressor(
                n_estimators=180,
                learning_rate=0.03,
                num_leaves=128,
                colsample_bytree=0.9,
                min_child_samples=3,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def normalize_feature_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas string missing values into sklearn-friendly nulls."""

    normalized = features.copy()

    for column in config.CATEGORICAL_FEATURE_COLUMNS:
        values = normalized[column].astype("string")
        normalized[column] = pd.Series(
            values.to_numpy(dtype=object, na_value=np.nan),
            index=normalized.index,
        )

    return normalized


def ensure_vehicle_age_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Add vehicle age if a caller still provides production year only."""

    if config.VEHICLE_AGE_COLUMN in data.columns:
        return data
    if config.PRODUCTION_YEAR_COLUMN not in data.columns:
        return data

    data_with_age = data.copy()
    reference_year = config.infer_vehicle_age_reference_year(
        data_with_age[config.PRODUCTION_YEAR_COLUMN],
        default=config.MAX_PRODUCTION_YEAR,
    )
    data_with_age[config.VEHICLE_AGE_COLUMN] = (
        reference_year - data_with_age[config.PRODUCTION_YEAR_COLUMN]
    )
    return data_with_age


def vehicle_age_reference_year_for_training(data: pd.DataFrame) -> int:
    """Return the reference year persisted for production-year inference."""

    if config.PRODUCTION_YEAR_COLUMN not in data.columns:
        return config.MAX_PRODUCTION_YEAR

    return config.infer_vehicle_age_reference_year(
        data[config.PRODUCTION_YEAR_COLUMN],
        default=config.MAX_PRODUCTION_YEAR,
    )


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate model columns and split cleaned data into features and target."""

    data = ensure_vehicle_age_feature(data)
    missing_columns = sorted(set(config.MODEL_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing model columns: {', '.join(missing_columns)}")

    features = normalize_feature_missing_values(data[config.FEATURE_COLUMNS])
    target = data[config.TARGET_COLUMN].copy()
    return features, target


def split_train_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the deterministic holdout split used to compare model candidates."""

    features, target = split_features_target(data)
    return train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> ModelResult:
    """Calculate regression metrics used for candidate selection."""

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
    """Train every candidate on the holdout split and rank by RMSE."""

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
    preferred_model_name: str | tuple[str, ...] = (
        "lightgbm_xt_autogluon_inspired",
        "lightgbm_large_autogluon_inspired",
        "lightgbm",
    ),
    baseline_model_name: str = "dummy_median",
    preferred_rmse_tolerance: float = 1.05,
) -> str:
    """Choose the best explainable LightGBM-family model when close to best."""

    if not results:
        raise ValueError("At least one model result is required.")

    best = results[0]
    by_name = {result.name: result for result in results}
    preferred_model_names = (
        (preferred_model_name,)
        if isinstance(preferred_model_name, str)
        else preferred_model_name
    )
    preferred_results = [
        by_name[name] for name in preferred_model_names if name in by_name
    ]
    baseline = by_name.get(baseline_model_name)

    if preferred_results and baseline:
        eligible_preferred = [
            result
            for result in preferred_results
            if result.rmse < baseline.rmse
            and result.rmse <= best.rmse * preferred_rmse_tolerance
        ]
        if eligible_preferred:
            return min(eligible_preferred, key=lambda result: result.rmse).name

    return best.name


def train_final_pipeline(
    data: pd.DataFrame,
    selected_model_name: str,
    candidates: list[ModelCandidate] | None = None,
) -> Pipeline:
    """Fit the selected model on the full processed training dataset."""

    candidates = candidates or default_model_candidates()
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    if selected_model_name not in candidate_by_name:
        raise ValueError(f"Unknown selected model: {selected_model_name}")

    features, target = split_features_target(data)
    pipeline = build_pipeline(clone(candidate_by_name[selected_model_name].estimator))
    pipeline.fit(features, target)
    return pipeline


def empty_feature_importance_frame() -> pd.DataFrame:
    """Return the empty schema used when an estimator has no importance data."""

    return pd.DataFrame(
        columns=[
            "feature",
            "source_feature",
            "importance",
            "importance_share",
            "importance_type",
        ]
    )


def model_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return transformed-feature importance for tree or linear estimators."""

    model = pipeline.named_steps["model"]
    importance_type = "feature_importance"
    values = getattr(model, "feature_importances_", None)

    if values is None:
        values = getattr(model, "coef_", None)
        importance_type = "absolute_coefficient"

    if values is None:
        return empty_feature_importance_frame()

    feature_names = transformed_feature_names(pipeline)
    importance = np.abs(np.asarray(values, dtype=float).ravel())
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    if len(importance) != len(feature_names):
        raise ValueError(
            "Model importance length does not match transformed feature count: "
            f"{len(importance)} != {len(feature_names)}"
        )

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "source_feature": [
                source_feature_name(feature_name) for feature_name in feature_names
            ],
            "importance": importance,
        }
    )
    total_importance = float(frame["importance"].sum())
    frame["importance_share"] = (
        frame["importance"] / total_importance if total_importance > 0 else 0.0
    )
    frame["importance_type"] = importance_type

    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def aggregate_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Aggregate transformed-feature importance back to source columns."""

    importance = model_feature_importance(pipeline)
    if importance.empty:
        return pd.DataFrame(
            columns=[
                "source_feature",
                "importance",
                "importance_share",
                "importance_type",
            ]
        )

    grouped = (
        importance.groupby("source_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    total_importance = float(grouped["importance"].sum())
    grouped["importance_share"] = (
        grouped["importance"] / total_importance if total_importance > 0 else 0.0
    )
    grouped["importance_type"] = importance["importance_type"].iloc[0]
    return grouped


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats before JSON serialization."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def results_to_dicts(results: list[ModelResult]) -> list[dict[str, Any]]:
    """Serialize model-selection results for the metrics artifact."""

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
    """Format artifact paths as repository-relative strings when possible."""

    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def save_model_artifacts(
    pipeline: Pipeline,
    summary: TrainingArtifactSummary,
    paths: TrainingArtifactPaths = TrainingArtifactPaths(),
) -> None:
    """Persist the trained pipeline using summary and path wrapper dataclasses."""

    paths.model.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, paths.model)

    trained_at = datetime.now(UTC).isoformat()
    metrics = results_to_dicts(summary.results)
    metadata = {
        "trained_at_utc": trained_at,
        "selected_model": summary.selected_model_name,
        "vehicle_age_reference_year": summary.vehicle_age_reference_year,
        "selection_rule": (
            "Prefer the best explainable LightGBM-family candidate when it "
            "beats dummy_median and is within 5% of the best holdout RMSE; "
            "otherwise use the best RMSE model."
        ),
        "target_column": config.TARGET_COLUMN,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_feature_columns": config.NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": config.CATEGORICAL_FEATURE_COLUMNS,
        "training_rows": summary.rows_count,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "model_path": display_path(paths.model),
        "metrics_path": display_path(paths.metrics),
        "feature_options_path": display_path(paths.feature_options),
    }

    with paths.metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    with paths.metrics.open("w", encoding="utf-8") as metrics_file:
        json.dump({"models": metrics}, metrics_file, ensure_ascii=False, indent=2)


def train_and_save_model(
    data: pd.DataFrame,
    candidates: list[ModelCandidate] | None = None,
    artifact_paths: TrainingArtifactPaths = TrainingArtifactPaths(),
) -> str:
    """Evaluate candidates, train the selected model and save wrapped artifacts."""

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
        TrainingArtifactSummary(
            results=results,
            selected_model_name=selected_model_name,
            rows_count=len(data),
            vehicle_age_reference_year=vehicle_age_reference_year_for_training(data),
        ),
        paths=artifact_paths,
    )
    save_feature_options(data, artifact_paths.feature_options)
    return selected_model_name


def log_training_artifacts(selected_model_name: str) -> None:
    """Log the standard model-training artifact summary."""

    logger.info("Selected model: %s", selected_model_name)
    logger.info("Saved model: %s", config.MODEL_PATH)
    logger.info("Saved metadata: %s", config.MODEL_METADATA_PATH)
    logger.info("Saved metrics: %s", config.TRAINING_METRICS_PATH)
    logger.info("Saved feature options: %s", config.FEATURE_OPTIONS_PATH)


def main() -> None:
    """CLI entrypoint for `uv run train-model`."""

    setup_logging()
    data = load_processed_data()
    selected_model_name = train_and_save_model(data)
    log_training_artifacts(selected_model_name)


if __name__ == "__main__":
    main()
