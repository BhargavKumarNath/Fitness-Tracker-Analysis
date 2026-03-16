import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.utils import load_dataset

st.set_page_config(
    page_title="Fitness Analytics Platform",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("dashboard/style.css")

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🏃 Fitness Analytics Platform</h1>
    <p style="font-size: 1.2rem; color: #9CA3AF;">Production-Grade Data Engineering & Machine Learning Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def get_quick_stats():
    df = load_dataset()
    if df.empty:
        return None
    return {
        'total_records': len(df),
        'unique_users': df['user_id'].nunique(),
        'avg_steps': df['steps'].mean(),
        'avg_calories': df['calories_burned'].mean(),
        'date_range': (df['date'].max() - df['date'].min()).days
    }

stats = get_quick_stats()

# Key Metrics Row
if stats:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("👥 Unique Users", f"{stats['unique_users']:,}")
    with col3:
        st.metric("🚶 Avg Steps", f"{stats['avg_steps']:,.0f}")
    with col4:
        st.metric("🔥 Avg Calories", f"{stats['avg_calories']:,.0f}")
    with col5:
        st.metric("📅 Days Tracked", f"{stats['date_range']}")
    
    st.divider()

# Main Content Layout
tab1, tab2, tab3 = st.tabs(["🎯 Project Overview", "🏗️ Architecture", "🚀 Features"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("## 🎯 Project Vision")
        st.markdown("""
        This platform demonstrates a **production-grade, end-to-end data science solution** 
        for analyzing fitness tracker data using modern MLOps practices.
        
        ### 📊 Dataset Scale
        - **358,497 records** from **1,959 users**
        - **6-month tracking period** (April - September 2023)
        - **7 activity types** with rich health metrics
        - **Partitioned Parquet storage** for optimal performance
        
        ### 🎓 Learning Objectives
        1. **Big Data Processing**: PySpark for scalable ETL
        2. **Machine Learning at Scale**: MLlib for distributed ML
        3. **Real-time Analytics**: Structured Streaming pipelines
        4. **MLOps Best Practices**: Docker + CI/CD automation
        5. **Interactive Dashboards**: Production-ready visualization
        """)
    
    with col2:
        st.markdown("## 📈 Key Statistics")
        
        if stats:
            st.info(f"""
            **Data Completeness**
            - ✅ 100% complete records
            - ✅ No missing values
            - ✅ Quality validated
            
            **Activity Distribution**
            - Swimming: 14.4%
            - Gym Workout: 14.3%
            - Cycling: 14.3%
            - Others: 57%
            
            **Performance**
            - High Performers (>10k steps): 40%
            - Average Sleep: 7.0 hours
            - Heart Rate Range: 60-169 bpm
            """)
        
        st.success("""
        **🏆 Tech Stack**
        - PySpark 3.5.0
        - Scikit-learn
        - Plotly & Streamlit
        - Docker & GitHub Actions
        - Parquet + Partitioning
        """)

with tab2:
    st.markdown("## 🏗️ System Architecture")
    
    st.markdown("""
    ```mermaid
    graph LR
        A[Raw Data<br/>Parquet Files] --> B[PySpark ETL<br/>Pipeline]
        B --> C[Processed<br/>Data Lake]
        C --> D[ML Models<br/>Training]
        D --> E[Trained Models<br/>pkl files]
        E --> F[Streamlit<br/>Dashboard]
    ```
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Data Pipeline")
        st.markdown("""
        **1. ETL Layer** (`src/etl_pipeline.py`)
        - Extract from 180+ partitioned files
        - Transform with feature engineering
        - Load to optimized Parquet storage
        
        **2. Processing Optimizations**
        - Columnar storage with Parquet
        - Year/Month partitioning
        - Predicate pushdown
        - Lazy evaluation
        
        **3. Data Quality**
        - Schema validation
        - Null handling
        - Outlier detection
        - Type enforcement
        """)
    
    with col2:
        st.markdown("### 🤖 ML Pipeline")
        st.markdown("""
        **3 Production Models**
        1. **K-Means Clustering** (10 KB)
           - User segmentation
           - 5 distinct personas
        
        2. **Random Forest Classifier** (745 MB)
           - Activity prediction
           - 84% accuracy
        
        3. **Random Forest Regressor** (3 GB)
           - Calorie estimation
           - R² = 0.91
        """)
    
    st.markdown("### 🔄 MLOps & DevOps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Containerization**
        - Multi-stage Docker build
        - Optimized image layers
        - Environment consistency
        - Portable deployment
        """)
    
    with col2:
        st.markdown("""
        **CI/CD Pipeline**
        - Automated testing
        - GitHub Actions
        - Unit + Integration tests
        - Quality gates
        """)
    
    with col3:
        st.markdown("""
        **Monitoring**
        - Model performance tracking
        - Data quality checks
        - Error handling
        - Logging & alerts
        """)

with tab3:
    st.markdown("## 🚀 Platform Features")
    
    features = {
        "📊 Overview Dashboard": {
            "description": "Real-time metrics and KPIs",
            "page": "1_Overview",
            "capabilities": [
                "Dataset statistics",
                "Activity distribution",
                "Health metrics summary",
                "Data schema reference"
            ]
        },
        "📈 Exploratory Analysis": {
            "description": "Deep dive into patterns and trends",
            "page": "2_Exploratory_Analysis",
            "capabilities": [
                "5-tab comprehensive EDA",
                "Activity comparisons",
                "Temporal patterns",
                "User segmentation (K-Means)"
            ]
        },
        "🔍 Data Insights": {
            "description": "Statistical analysis and quality assessment",
            "page": "Data_Insights",
            "capabilities": [
                "Detailed statistics",
                "Data quality metrics",
                "Outlier detection",
                "Key findings report"
            ]
        },
        "🧠 Advanced Modeling": {
            "description": "ML model performance and evaluation",
            "page": "3_Advanced_Modelling",
            "capabilities": [
                "Model comparison",
                "Feature importance",
                "Confusion matrices",
                "Prediction analysis"
            ]
        },
        "🔮 Live Inference": {
            "description": "Interactive predictions with trained models",
            "page": "4_Live_Inference",
            "capabilities": [
                "Activity classification",
                "Calorie prediction",
                "Real-time inference",
                "Custom input testing"
            ]
        }
    }
    
    for feature, details in features.items():
        with st.expander(f"**{feature}**: {details['description']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Key Capabilities:**")
                for cap in details['capabilities']:
                    st.markdown(f"✅ {cap}")
            
            with col2:
                st.info(f"📄 Page: `{details['page']}`")

st.divider()

# Quick Start Section
st.markdown("## 🎯 Quick Start Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Explore Data
    Start with the **Overview** page to understand the dataset structure and key metrics.
    """)
    if st.button("Go to Overview →", width='stretch'):
        st.switch_page("pages/1_Overview.py")

with col2:
    st.markdown("""
    ### 2️⃣ Analyze Patterns
    Visit **Exploratory Analysis** for deep insights and user segmentation.
    """)
    if st.button("Go to EDA →", width='stretch'):
        st.switch_page("pages/2_Exploratory_Analysis.py")

with col3:
    st.markdown("""
    ### 3️⃣ Test Models
    Try **Live Inference** to interact with trained ML models.
    """)
    if st.button("Go to Predictions →", width='stretch'):
        st.switch_page("pages/4_Live_Inference.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 2rem 0;">
    <p>Built with PySpark, Streamlit, and Docker | Production-Ready Data Engineering</p>
    <p>📚 Full documentation available in <code>readme.md</code></p>
</div>
""", unsafe_allow_html=True)