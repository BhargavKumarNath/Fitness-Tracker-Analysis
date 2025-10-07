import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel
from pyspark.ml import PipelineModel
from dashboard.utils import get_user_segments
import sys
import os
sys.path.append("/app")  

# @st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("FitnessDashboard").master("local[*]").getOrCreate()

# @st.cache_data
def load_data():
    spark = get_spark_session()
    df_spark = spark.read.parquet("/app/data_lake/processed/fitness_data")
    return df_spark.toPandas()

st.set_page_config(page_title="EDA & Clustering", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis & User Clustering")

spark = get_spark_session() 
df = load_data()

# Visualizations 
st.header("Activity Analysis")
activity_summary = df.groupby("activity_type").agg(
    avg_steps=("steps", "mean"),
    avg_calories=("calories_burned", "mean")
).reset_index()
st.bar_chart(activity_summary.set_index("activity_type")['avg_steps'], height=400)

# User Segmentation
st.header("User Segments (K-Means Clustering)")
user_segments_df = get_user_segments(spark, df) 

cluster_profiles = user_segments_df.groupby('prediction').agg(
    avg_steps=('avg_steps', 'mean'),
    avg_calories=('avg_calories', 'mean'),
    avg_hr=('avg_hr', 'mean'),
    num_users=('user_id', 'count')
).reset_index()

st.dataframe(cluster_profiles)
st.bar_chart(cluster_profiles.set_index('prediction')['num_users'])
