from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, to_date, year

from src.etl.transform import transform_data


def run_etl(raw_data_dir: Path | str, processed_data_dir: Path | str) -> Path:
    """Extract, transform, and load raw parquet data with Spark."""
    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_data_dir)
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    spark = SparkSession.builder.appName("FitnessTrackerETL").getOrCreate()
    try:
        frame = spark.read.parquet(*(str(path) for path in parquet_files))
        frame = frame.withColumn("date", to_date(col("date")))
        transformed = transform_data(frame)
        output = transformed.withColumn("year", year(col("date"))).withColumn(
            "month", month(col("date"))
        )
        output.write.mode("overwrite").partitionBy("year", "month").parquet(str(processed_dir))
    finally:
        spark.stop()

    return processed_dir