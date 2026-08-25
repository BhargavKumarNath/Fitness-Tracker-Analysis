# src/etl_pipeline.py

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, date_format, month, to_date, when, year

from src.config import get_runtime_paths


def transform_data(df: DataFrame) -> DataFrame:
    """Apply the ETL feature engineering on the raw fitness dataset."""
    df_transformed = df.withColumn("day_of_week", date_format(col("date"), "E"))
    df_transformed = df_transformed.withColumn(
        "calories_to_steps_ratio",
        when(col("steps") > 0, col("calories_burned") / col("steps")).otherwise(0),
    )
    return df_transformed


def main() -> None:
    paths = get_runtime_paths()
    raw_data_path = paths["raw_data_dir"]
    processed_data_path = paths["processed_data_dir"]

    raw_data_path.mkdir(parents=True, exist_ok=True)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName("FitnessTrackerETL").getOrCreate()
    print("SparkSession created. Starting ETL process...")

    parquet_files = [str(path) for path in raw_data_path.rglob("*.parquet")]
    if not parquet_files:
        print(f"No Parquet files found under {raw_data_path}. Exiting.")
        spark.stop()
        return

    df = spark.read.parquet(*parquet_files)
    df = df.withColumn("date", to_date(col("date")))
    print(f"Successfully extracted {df.count()} records from Parquet files.")

    df_transformed = transform_data(df)
    print("Transformation complete. New features added.")

    df_to_load = df_transformed.withColumn("year", year(col("date")))
    df_to_load = df_to_load.withColumn("month", month(col("date")))

    df_to_load.write.mode("overwrite").partitionBy("year", "month").parquet(str(processed_data_path))

    print(f"Data successfully loaded to {processed_data_path}")
    spark.stop()


if __name__ == "__main__":
    main()
