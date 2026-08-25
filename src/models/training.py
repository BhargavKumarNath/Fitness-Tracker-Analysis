from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_TREES = 50

# Below this row count, an 80/20 stratified split cannot be trusted (or, for
# tiny test fixtures, cannot even be constructed). Held-out metrics are
MIN_ROWS_FOR_HELD_OUT_EVAL = 20


def _held_out_classification_accuracy(
    pipeline_template: Pipeline, X: pd.DataFrame, y: pd.Series
) -> float | None:
    """Fit a clone of the pipeline on an 80/20 split and score it on the held-out 20%.

    The shipped model is still fit on 100% of the data by the caller; this
    only exists to produce an honest accuracy number, since the training
    pipeline otherwise never evaluates itself.
    """
    if len(y) < MIN_ROWS_FOR_HELD_OUT_EVAL or y.value_counts().min() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_pipeline = clone(pipeline_template)
    eval_pipeline.fit(X_train, y_train)
    return float(accuracy_score(y_test, eval_pipeline.predict(X_test)))


def _held_out_regression_metrics(
    pipeline_template: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[float, float] | None:
    """Fit a clone of the pipeline on an 80/20 split; return (r2, rmse) on the held-out 20%."""
    if len(y) < MIN_ROWS_FOR_HELD_OUT_EVAL:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_pipeline = clone(pipeline_template)
    eval_pipeline.fit(X_train, y_train)
    predicted = eval_pipeline.predict(X_test)
    r2 = float(r2_score(y_test, predicted))
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    return r2, rmse


def train_dashboard_models(df: pd.DataFrame, models_dir: Path | str) -> list[Path]:
    """Train and persist the classifier, regressor, segmentation model, and metrics."""
    destination = Path(models_dir)
    destination.mkdir(parents=True, exist_ok=True)

    class_features = ["steps", "calories_burned", "heart_rate_avg"]
    class_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=MODEL_TREES,
                max_depth=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    held_out_accuracy = _held_out_classification_accuracy(
        class_pipeline, df[class_features], df["activity_type"]
    )
    class_pipeline.fit(df[class_features], df["activity_type"])

    reg_features = ["steps", "heart_rate_avg", "sleep_hours", "activity_type"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["steps", "heart_rate_avg", "sleep_hours"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["activity_type"]),
        ]
    )
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=MODEL_TREES,
                max_depth=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    held_out_reg_metrics = _held_out_regression_metrics(
        reg_pipeline, df[reg_features], df["calories_burned"]
    )
    reg_pipeline.fit(df[reg_features], df["calories_burned"])

    user_df = df.groupby("user_id").agg({
        "steps": "mean",
        "calories_burned": "mean",
        "heart_rate_avg": "mean",
    }).rename(columns={
        "steps": "avg_steps",
        "calories_burned": "avg_calories",
        "heart_rate_avg": "avg_hr",
    })
    if user_df.empty:
        raise ValueError("No user profiles available for clustering.")

    cluster_features = ["avg_steps", "avg_calories", "avg_hr"]
    cluster_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=min(5, len(user_df)), random_state=42, n_init=20)),
    ])
    cluster_pipeline.fit(user_df[cluster_features])

    metrics = {
        "activity_classifier": {"held_out_accuracy": held_out_accuracy},
        "calories_regressor": {
            "held_out_r2": held_out_reg_metrics[0] if held_out_reg_metrics else None,
            "held_out_rmse": held_out_reg_metrics[1] if held_out_reg_metrics else None,
        },
        "methodology": (
            "Held-out metrics come from an 80/20 split (stratified for the classifier), "
            "evaluated with the same pipeline and hyperparameters as the shipped model. "
            "The shipped model itself is fit on 100% of the processed data, this function "
            "never scored it before, so these numbers previously did not exist anywhere "
            "in this repository."
        ),
    }

    artifacts = {
        "activity_classifier.pkl": class_pipeline,
        "calories_regressor.pkl": reg_pipeline,
        "user_segmentation.pkl": cluster_pipeline,
        "cluster_features.pkl": cluster_features,
    }
    artifact_paths = []
    for filename, artifact in artifacts.items():
        path = destination / filename
        joblib.dump(artifact, path)
        artifact_paths.append(path)

    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    artifact_paths.append(metrics_path)

    return artifact_paths
