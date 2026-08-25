from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
from pyspark.sql import SparkSession

from src.config import get_runtime_paths
from src.etl.extract import load_raw_data
from src.etl.load import write_processed_data
from src.etl.transform import transform_data
from src.models.training import train_dashboard_models


def run_pipeline(project_root: Path | str | None = None) -> dict[str, Path]:
    root = Path(project_root).resolve() if project_root is not None else Path(os.environ.get("FITNESS_TRACKER_ROOT", Path(__file__).resolve().parent.parent)).resolve()
    paths = {
        "raw_data_dir": root / "data_lake" / "raw" / "synthetic_user_data",
        "processed_data_dir": root / "data_lake" / "processed" / "fitness_data",
        "models_dir": root / "dashboard" / "models",
    }

    raw_df = load_raw_data(paths["raw_data_dir"])
    raw_df = raw_df.copy()
    if "date" in raw_df.columns:
        raw_df["date"] = pd.to_datetime(raw_df["date"])

    spark = SparkSession.builder.master("local[1]").appName("pipeline").getOrCreate()
    try:
        transformed = transform_data(spark.createDataFrame(raw_df))
        transformed_df = transformed.toPandas()

        transformed_df = transformed_df.drop(
            columns=[column for column in ["year", "month"] if column in transformed_df.columns],
            errors="ignore",
        )
        transformed_df["year"] = transformed_df["date"].dt.year
        transformed_df["month"] = transformed_df["date"].dt.month

        processed_dir = paths["processed_data_dir"]
        write_processed_data(transformed_df, processed_dir)
        train_dashboard_models(transformed_df, paths["models_dir"])
    finally:
        spark.stop()

    return paths


if __name__ == "__main__":
    run_pipeline()
