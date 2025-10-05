# tests/test_etl.py

import pytest
from pyspark.sql import SparkSession
from src.etl_pipeline import transform_data
from pyspark.sql.functions import col

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for the tests."""
    return SparkSession.builder \
        .appName("ETLTests") \
        .master("local[1]") \
        .getOrCreate()

def test_transform_data(spark):
    """
    Unit test for the transform_data function.
    """
    # 1. Create sample input data
    input_data = [
        ("2023-04-01", 100, 10.0),
        ("2023-04-02", 0, 5.0),
    ]
    input_schema = ["date", "steps", "calories_burned"]
    input_df = spark.createDataFrame(input_data, input_schema) \
        .withColumn("date", col("date").cast("date"))

    # 2. Apply the transformation
    output_df = transform_data(input_df)

    # 3. Assert the expected outcome
    assert "day_of_week" in output_df.columns
    assert "calories_to_steps_ratio" in output_df.columns

    # Check the calculated values
    results = output_df.collect()
    assert results[0]["day_of_week"] == "Sat"
    assert results[0]["calories_to_steps_ratio"] == 0.1  # 10.0 / 100
    
    # Check the division-by-zero case
    assert results[1]["calories_to_steps_ratio"] == 0.0