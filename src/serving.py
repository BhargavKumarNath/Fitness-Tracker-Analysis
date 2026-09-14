"""Read processed data and trained model artifacts for downstream consumers."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd

from src.config import get_runtime_paths

__all__ = [
    "get_model_path",
    "load_dataset",
    "get_activity_categories",
    "load_model_metrics",
    "load_user_segmentation_model",
    "get_user_segments",
]


def _cached(fn):
    """Minimal memoizing cache exposing .clear(), so tests/tools can reset state."""
    cache: dict = {}

    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    wrapper.clear = cache.clear
    return wrapper


def get_model_path(model_name: str) -> Path:
    """Return a model artifact path for the current runtime root."""
    if Path(model_name).name != model_name or not model_name.isidentifier():
        raise ValueError(f"Invalid model name: {model_name}")
    return get_runtime_paths()["models_dir"] / f"{model_name}.pkl"


@_cached
def load_dataset() -> pd.DataFrame:
    """Load the processed dataset from Parquet using repo-root-aware paths."""
    data_path = get_runtime_paths()["processed_data_dir"]
    if not data_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(data_path)


# Fallback categories match the real training data (src/models/training.py
# fits its OneHotEncoder on activity_type as-is: lowercase, snake_case).
# Used only when the processed dataset itself is unavailable to read from.
_FALLBACK_ACTIVITY_CATEGORIES = ["cycling", "gym_workout", "hiking", "running", "swimming", "walking", "yoga"]


@_cached
def get_activity_categories() -> list[str]:
    """Single source of truth for activity labels, shared by every consumer.

    Every caller that needs the set of activity_type values must build its
    options from this function rather than a separately hand-typed list, so
    it cannot drift from the categories the model was actually fit on.
    """
    df = load_dataset()
    if not df.empty and "activity_type" in df.columns:
        return sorted(df["activity_type"].dropna().unique().tolist())
    return _FALLBACK_ACTIVITY_CATEGORIES


@_cached
def load_model_metrics() -> dict | None:
    """Load the held-out evaluation metrics the training script writes for itself.

    Returns None if the pipeline has not been run yet, callers should treat
    that as "not yet computed" rather than falling back to a guessed number.
    """
    metrics_path = get_runtime_paths()["models_dir"] / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@_cached
def load_user_segmentation_model():
    """Load the user segmentation model and its feature list if present locally."""
    model_path = get_model_path("user_segmentation")
    features_path = get_model_path("cluster_features")
    if not model_path.exists() or not features_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(features_path)


def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign user segments, falling back to deterministic activity bands without a trained model."""
    pipeline, features = load_user_segmentation_model()

    user_summary_df = (
        df.groupby("user_id")
        .agg({"steps": "mean", "calories_burned": "mean", "heart_rate_avg": "mean"})
        .rename(columns={"steps": "avg_steps", "calories_burned": "avg_calories", "heart_rate_avg": "avg_hr"})
        .reset_index()
    )

    if pipeline is None:
        return _build_baseline_segments(user_summary_df)

    user_summary_df["prediction"] = pipeline.predict(user_summary_df[features])
    return user_summary_df


def _build_baseline_segments(user_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic activity bands when the optional model is absent."""
    if user_summary_df.empty:
        return user_summary_df.assign(prediction=pd.Series(dtype="int64"))

    ranked_steps = user_summary_df["avg_steps"].rank(method="first")
    cluster_count = min(5, len(user_summary_df))
    user_summary_df = user_summary_df.copy()
    user_summary_df["prediction"] = (
        ((ranked_steps - 1) * cluster_count / len(user_summary_df)).astype(int).clip(upper=cluster_count - 1)
    )
    return user_summary_df
