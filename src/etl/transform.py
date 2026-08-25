from pyspark.sql import DataFrame
from pyspark.sql.functions import col, date_format, when


def transform_data(df: DataFrame) -> DataFrame:
    """Apply feature engineering to the raw fitness dataset."""
    transformed = df.withColumn("day_of_week", date_format(col("date"), "E"))
    return transformed.withColumn(
        "calories_to_steps_ratio",
        when(col("steps") > 0, col("calories_burned") / col("steps")).otherwise(0),
    )