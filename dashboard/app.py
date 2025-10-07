# dashboard/app.py

import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import avg, count
from pyspark.ml import Pipeline

# Page Configuration and Spark Session (Keep as is)
st.set_page_config(page_title="Fitness Tracker Dashboard", page_icon="🏃", layout="wide")
@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("FitnessDashboard").master("local[*]").getOrCreate()
spark = get_spark_session()

# Data Loading (Keep as is)
@st.cache_data
def load_data() -> pd.DataFrame:
    df_spark = spark.read.parquet("data_lake/processed/fitness_data")
    return df_spark.toPandas()

# Function to run K-Means Clustering
@st.cache_data
def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Performs user segmentation using K-Means."""
    # Convert pandas df to spark df for clustering
    spark_df = spark.createDataFrame(df)
    
    # User-level summary
    user_summary_df = spark_df.groupBy("user_id").agg(
        avg("steps").alias("avg_steps"),
        avg("calories_burned").alias("avg_calories"),
        avg("heart_rate_avg").alias("avg_hr")
    )

    features_for_clustering = ['avg_steps', 'avg_calories', 'avg_hr']
    assembler = VectorAssembler(inputCols=features_for_clustering, outputCol="features_unscaled")
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    kmeans = KMeans(featuresCol="features", k=3, seed=1)
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    model = pipeline.fit(user_summary_df)
    predictions = model.transform(user_summary_df)
    
    return predictions.toPandas()


def main():
    st.title("Fitness Tracker Data Dashboard")
    df = load_data()
    st.success(f"Data loaded successfully! Total records: {len(df)}")
    
    # Sidebar for Filters
    st.sidebar.header("Filters")
    selected_activity = st.sidebar.selectbox("Select an Activity", ["All"] + sorted(df['activity_type'].unique()))

    # Filter data based on selection
    if selected_activity != "All":
        df_filtered = df[df['activity_type'] == selected_activity]
    else:
        df_filtered = df

    # Section 1: Metrics
    st.header(f"Displaying Metrics for: {selected_activity}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Steps", f"{df_filtered['steps'].mean():,.0f}")
    col2.metric("Avg. Calories Burned", f"{df_filtered['calories_burned'].mean():,.0f}")
    col3.metric("Avg. Heart Rate", f"{df_filtered['heart_rate_avg'].mean():,.0f}")

    # Section 2: Visualizations
    st.header("Visual Insights")
    st.subheader("Distribution of Steps")
    st.bar_chart(df_filtered.groupby('day_of_week')['steps'].mean())

    # Section 3: User Segmentation
    st.header("User Segments")
    with st.spinner("Calculating user segments..."):
        user_segments_df = get_user_segments(df)

    st.subheader("Cluster Profiles")
    # Calculate profiles from the predictions
    cluster_profiles = user_segments_df.groupby('prediction').agg(
        avg_steps=('avg_steps', 'mean'),
        avg_calories=('avg_calories', 'mean'),
        avg_hr=('avg_hr', 'mean'),
        num_users=('user_id', 'count')
    ).reset_index()
    
    st.dataframe(cluster_profiles)
    
    st.subheader("Distribution of Users Across Segments")
    st.bar_chart(cluster_profiles.set_index('prediction')['num_users'])


if __name__ == "__main__":
    main()