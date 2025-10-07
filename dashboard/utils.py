import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import sys
sys.path.append("/app")  

def get_user_segments(spark: SparkSession, df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs user segmentation using K-Means on the input DataFrame.
    Args:
        spark: The active SparkSession.
        df: The input data as a Pandas DataFrame.
    Returns:
        A Pandas DataFrame with user_id and cluster prediction.
    """
    spark_df = spark.createDataFrame(df)
    
    # User-level summary
    user_summary_df = spark_df.groupBy("user_id").agg(
        avg("steps").alias("avg_steps"),
        avg("calories_burned").alias("avg_calories"),
        avg("heart_rate_avg").alias("avg_hr")
    )

    # ML Pipeline for clustering (as in Phase 3)
    features_for_clustering = ['avg_steps', 'avg_calories', 'avg_hr']
    assembler = VectorAssembler(inputCols=features_for_clustering, outputCol="features_unscaled")
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    kmeans = KMeans(featuresCol="features", k=3, seed=1)
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    model = pipeline.fit(user_summary_df)
    predictions = model.transform(user_summary_df)
    
    # Return results as a Pandas DataFrame
    return predictions.toPandas()
