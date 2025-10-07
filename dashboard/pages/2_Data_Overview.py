import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import sys
sys.path.append("/app")  # Docker mounts the project at /app

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("FitnessDashboard").master("local[*]").getOrCreate()

@st.cache_data
def load_data():
    spark = get_spark_session()
    df_spark = spark.read.parquet("/app/data_lake/processed/fitness_data")
    return df_spark.toPandas()

st.title("📊 Data Overview")
st.markdown("Here's a look at the clean, processed data that powers our analysis and models")

df = load_data()

st.dataframe(df.head(100))

st.markdown("### Data Schema")
st.text(
    "user_id: Unique identifier for the user\n"
    "date: Date of the activity\n"
    "steps: Number of steps recorded\n"
    "calories_burned: Calories burned\n"
    "heart_rate_avg: Average heart rate during the activity\n"
    "sleep_hours: Hours slept on the preceding night\n"
    "activity_type: Type of activity performed\n"
    "day_of_week: Day of the week (engineered feature)\n"
    "calories_to_steps_ratio: (engineered feature)\n"
    "year, month: Partition columns"
)