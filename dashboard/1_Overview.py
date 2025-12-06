import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.utils import load_dataset

st.set_page_config(
    page_title="Fitness Analysis - Overview",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("dashboard/style.css")
except FileNotFoundError:
    pass # Handle case where css might be missing temporarily

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🏃 Fitness Tracker Analysis</h1>
    <p style="font-size: 1.2rem; color: #9CA3AF;">End-to-End Data Engineering & Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

# Data Loading
df = load_dataset()

# Key Metrics Row
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['user_id'].nunique()}")
    with col3:
        st.metric("Avg Steps", f"{df['steps'].mean():,.0f}")
    with col4:
        st.metric("Avg Calories", f"{df['calories_burned'].mean():,.0f}")
    
    st.divider()

# Main Layout: Project Context & Data Preview
col_context, col_preview = st.columns([1, 1.5], gap="large")

with col_context:
    st.subheader("🚀 Project Scope")
    st.markdown("""
    This platform demonstrates a scalable data pipeline and advanced analytics for fitness tracker data.
    
    **Architecture & Stack:**
    - **ETL Layer:** Simulating Spark-like processing with Pandas on partitioned Parquet files.
    - **ML Layer:** Sklearn pipelines for User Segmentation (KMeans) and Calorie Prediction (RandomForest/XGBoost).
    - **Serving:** Streamlit with optimized caching and modular components.
    
    **Objectives:**
    1.  **Ingest & Process**: Handle raw streaming data simulation.
    2.  **Analyze**: Uncover patterns in user activity and vitals.
    3.  **Predict**: Provide real-time inference for health metrics.
    """)

    st.info("Navigate to 'Exploratory Analysis' for deep dives or 'Live Inference' to test models.")

with col_preview:
    st.subheader("📋 Dataset Snapshot")
    if not df.empty:
        st.dataframe(df.head(100), height=300, width="stretch")
    else:
        st.warning("Data not available.")

    st.caption("Showing top 100 records from processed data lake.")

st.divider()

# Schema Reference
with st.expander("View Data Schema Reference"):
    st.markdown("""
    | Column | Description | Type |
    |---|---|---|
    | `user_id` | Unique User Identifier | String |
    | `date` | Activity Date | Date |
    | `steps` | Daily Steps Count | Int |
    | `calories_burned` | Total Calories Burned | Float |
    | `heart_rate_avg` | Average Daily Heart Rate | Float |
    | `sleep_hours` | Sleep Duration | Float |
    | `activity_type` | Categorical Activity Label | String |
    """)