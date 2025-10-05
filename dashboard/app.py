# dashboard/app.py

import streamlit as st
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import avg, count
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Fitness Tracker Dashboard",
    page_icon="🏃",
    layout="wide"
)

# --- SparkSession Initialization ---
@st.cache_resource
def get_spark_session():
    """Get or create a Spark Session."""
    return SparkSession.builder \
        .appName("FitnessDashboard") \
        .master("local[*]") \
        .getOrCreate()

spark = get_spark_session()

# --- Data Loading ---
@st.cache_data
def load_data() -> DataFrame:
    """Load the processed fitness data."""
    df = spark.read.parquet("../data_lake/processed/fitness_data")
    return df

# The main function to run the app
def main():
    st.title("Fitness Tracker Data Dashboard")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    st.success(f"Data loaded successfully! Total records: {df.count()}")

    # --- Section 1: High-Level Metrics ---
    st.header("Overall Activity Summary")

    activity_summary_spark = df.groupBy("activity_type") \
        .agg(
            count("*").alias("num_records"),
            avg("steps").alias("avg_steps"),
            avg("calories_burned").alias("avg_calories")
        ).orderBy("activity_type")

    # Convert to Pandas for display
    activity_summary_pd = activity_summary_spark.toPandas()

    st.dataframe(activity_summary_pd)

    # --- Section 2: Visualizations ---
    st.header("Visual Insights")
    
    # Bar Chart of Average Steps
    st.subheader("Average Steps by Activity")
    st.bar_chart(activity_summary_pd.set_index("activity_type")['avg_steps'])

    # --- Section 3: User Segmentation ---
    st.header("User Segments")
    st.write("This shows the user segmentation we performed during our EDA phase.")
    
    st.info("User segmentation results would be displayed here.")


if __name__ == "__main__":
    main()