from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import pyarrow.dataset as ds
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_runtime_paths
from src.etl_pipeline import transform_data


def load_raw_data(project_root: Path | None = None) -> pd.DataFrame:
    paths = get_runtime_paths() if project_root is None else {
        "raw_data_dir": project_root / "data_lake" / "raw" / "synthetic_user_data",
        "processed_data_dir": project_root / "data_lake" / "processed" / "fitness_data",
        "models_dir": project_root / "dashboard" / "models",
    }

    raw_dir = paths["raw_data_dir"]
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    return pd.read_parquet([str(path) for path in parquet_files])


def ensure_pipeline_output_dirs(project_root: Path | None = None) -> dict[str, Path]:
    paths = get_runtime_paths() if project_root is None else {
        "raw_data_dir": project_root / "data_lake" / "raw" / "synthetic_user_data",
        "processed_data_dir": project_root / "data_lake" / "processed" / "fitness_data",
        "models_dir": project_root / "dashboard" / "models",
    }
    for key in ("processed_data_dir", "models_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _write_parquet_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = __import__("pyarrow").Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month"],
        existing_data_behavior="overwrite_or_ignore",
    )


def train_models_for_pipeline(df: pd.DataFrame, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    class_features = ["steps", "calories_burned", "heart_rate_avg"]
    target_class = "activity_type"

    class_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    class_pipeline.fit(df[class_features], df[target_class])
    joblib.dump(class_pipeline, models_dir / "activity_classifier.pkl")

    reg_features = ["steps", "heart_rate_avg", "sleep_hours", "activity_type"]
    target_reg = "calories_burned"

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
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    reg_pipeline.fit(df[reg_features], df[target_reg])
    joblib.dump(reg_pipeline, models_dir / "calories_regressor.pkl")

    user_df = df.groupby("user_id").agg({
        "steps": "mean",
        "calories_burned": "mean",
        "heart_rate_avg": "mean",
    }).rename(columns={
        "steps": "avg_steps",
        "calories_burned": "avg_calories",
        "heart_rate_avg": "avg_hr",
    })

    cluster_features = ["avg_steps", "avg_calories", "avg_hr"]
    cluster_count = min(5, len(user_df))

    if len(user_df) == 0:
        raise ValueError("No user segments available for clustering.")
    if cluster_count < 1:
        raise ValueError("At least one user record is required for clustering.")

    cluster_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=cluster_count, random_state=42, n_init=20)),
    ])
    cluster_pipeline.fit(user_df[cluster_features])
    joblib.dump(cluster_pipeline, models_dir / "user_segmentation.pkl")
    joblib.dump(cluster_features, models_dir / "cluster_features.pkl")


def run_pipeline(project_root: Path | str | None = None) -> dict[str, Path]:
    root = Path(project_root).resolve() if project_root is not None else Path(os.environ.get("FITNESS_TRACKER_ROOT", Path(__file__).resolve().parent.parent)).resolve()
    paths = {
        "raw_data_dir": root / "data_lake" / "raw" / "synthetic_user_data",
        "processed_data_dir": root / "data_lake" / "processed" / "fitness_data",
        "models_dir": root / "dashboard" / "models",
    }

    raw_df = load_raw_data(project_root=root)
    raw_df = raw_df.copy()
    if "date" in raw_df.columns:
        raw_df["date"] = pd.to_datetime(raw_df["date"])

    spark = __import__("pyspark.sql").sql.SparkSession.builder.master("local[1]").appName("pipeline").getOrCreate()
    transformed = transform_data(spark.createDataFrame(raw_df))
    transformed_df = transformed.toPandas()

    transformed_df = transformed_df.drop(columns=[col for col in ["year", "month"] if col in transformed_df.columns], errors="ignore")
    transformed_df["year"] = transformed_df["date"].dt.year
    transformed_df["month"] = transformed_df["date"].dt.month

    processed_dir = paths["processed_data_dir"]
    _write_parquet_dataset(transformed_df, processed_dir)

    train_models_for_pipeline(transformed_df, paths["models_dir"])
    spark.stop()
    return paths


if __name__ == "__main__":
    run_pipeline()
