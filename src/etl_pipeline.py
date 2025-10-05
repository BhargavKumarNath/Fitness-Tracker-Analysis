# src/etl_pipeline.py

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format, when, year, month

def main():
    raw_data_path = "/app/data_lake/raw/synthetic_user_data"
    processed_data_path = "/app/data_lake/processed/fitness_data"

    spark = SparkSession.builder.appName("FitnessTrackerETL").getOrCreate()
    print("SparkSession created. Starting ETL process...")

    # EXTRACT
    parquet_files = [str(p) for p in Path(raw_data_path).rglob("*.parquet")]
    if not parquet_files:
        print("No Parquet files found. Exiting.")
        spark.stop()
        return

    df = spark.read.parquet(*parquet_files)
    df = df.withColumn("date", to_date(col("date")))
    print(f"Successfully extracted {df.count()} records from Parquet files.")

    # TRANSFORM
    df_transformed = df.withColumn("day_of_week", date_format(col("date"), "E"))
    df_transformed = df_transformed.withColumn(
        "calories_to_steps_ratio",
        when(col("steps") > 0, col("calories_burned") / col("steps")).otherwise(0)
    )
    print("Transformation complete. New features added.")

    # LOAD
    df_to_load = df_transformed.withColumn("year", year(col("date")))
    df_to_load = df_to_load.withColumn("month", month(col("date")))
    
    df_to_load.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(processed_data_path)

    print(f"Data successfully loaded to {processed_data_path}")
    spark.stop()

if __name__ == "__main__":
    main()
