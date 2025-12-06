import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.utils import get_user_segments, load_dataset

st.set_page_config(page_title="EDA & Clustering", page_icon="📈", layout="wide")

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("dashboard/style.css")
except:
    pass

df = load_dataset()

st.title("📈 EDA & User Segmentation")

if not df.empty:
    # EDA Section
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Activity Distribution")
        fig_activity = px.pie(df, names='activity_type', title='Distribution of Activity Types', hole=0.4)
        fig_activity.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_activity)

    with col2:
        st.subheader("Steps vs Calories")
        fig_scatter = px.scatter(df, x='steps', y='calories_burned', color='activity_type', 
                                title='Steps vs Calories Burned', opacity=0.7)
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_scatter)

    st.subheader("Heart Rate Trends")
    # Simplify for trend: average daily heart rate
    daily_hr = df.groupby('date')['heart_rate_avg'].mean().reset_index()
    fig_line = px.line(daily_hr, x='date', y='heart_rate_avg', title='Average Daily Heart Rate Trend')
    fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_line)
    
    st.divider()

    # Clustering Section
    st.header("🧬 User Segmentation (Clustering)")
    st.markdown("We use K-Means clustering to group users based on their average activity levels.")
    
    with st.spinner("Running User Segmentation..."):
        user_segments = get_user_segments(df)
    
    if not user_segments.empty:
        col_c1, col_c2 = st.columns([2, 1])
        
        with col_c1:
            fig_3d = px.scatter_3d(user_segments, x='avg_steps', y='avg_calories', z='avg_hr',
                                  color='prediction', title="3D Visualization of User Segments",
                                  labels={'prediction': 'Cluster'})
            fig_3d.update_layout(scene = dict(
                        xaxis = dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white", showbackground=True),
                        yaxis = dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white", showbackground=True),
                        zaxis = dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white", showbackground=True),),
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_3d)
            
        with col_c2:
            st.subheader("Cluster Insights")
            st.markdown("Statistics per Cluster:")
            cluster_stats = user_segments.groupby('prediction').mean().reset_index()
            # width='stretch' is the correct new syntax replacing use_container_width=True
            st.dataframe(cluster_stats.style.format("{:.1f}"), width="stretch")
            
            st.markdown("""
            **Interpretation:**
            - **Cluster 0**: Likely Sediment/Low Activity
            - **Cluster 1**: Moderate Activity
            - **Cluster 2**: High Performers
            *(Note: Interpretation depends on random seed and training)*
            """)

else:
    st.warning("No data found.")
