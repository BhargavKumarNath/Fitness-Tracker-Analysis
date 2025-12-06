import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.utils import get_user_segments, load_dataset

st.set_page_config(page_title="EDA & Insights", page_icon="📈", layout="wide")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css("dashboard/style.css")

df = load_dataset()

st.title("📈 Exploratory Data Analysis & User Segmentation")
st.markdown("Comprehensive insights into fitness tracker data patterns and user behaviors")

if df.empty:
    st.error("No data available")
    st.stop()

# Create tabs for organized content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏃 Activity Analysis", 
    "💓 Health Metrics",
    "📅 Temporal Patterns",
    "🧬 User Segmentation"
])

# TAB 1: Overview
with tab1:
    st.header("Dataset Overview")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Unique Users", f"{df['user_id'].nunique():,}")
    col3.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    col4.metric("Activities", f"{df['activity_type'].nunique()}")
    col5.metric("Avg Daily Steps", f"{df['steps'].mean():,.0f}")
    
    st.divider()
    
    # Distribution overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Activity Type Distribution")
        activity_counts = df['activity_type'].value_counts().reset_index()
        activity_counts.columns = ['Activity', 'Count']
        
        fig = px.pie(
            activity_counts,
            names='Activity',
            values='Count',
            title='Distribution of Activities',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📈 Steps Distribution")
        fig = px.histogram(
            df,
            x='steps',
            nbins=50,
            title='Distribution of Daily Steps',
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Steps",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, width='stretch')
    
    # Summary Statistics
    st.subheader("📋 Summary Statistics")
    summary_stats = df[['steps', 'calories_burned', 'heart_rate_avg', 'sleep_hours']].describe().T
    summary_stats['variance'] = df[['steps', 'calories_burned', 'heart_rate_avg', 'sleep_hours']].var()
    st.dataframe(summary_stats.style.background_gradient(cmap='Blues'), width='stretch')

# TAB 2: Activity Analysis
with tab2:
    st.header("Activity Deep Dive")
    
    # Activity comparison
    activity_stats = df.groupby('activity_type').agg({
        'steps': ['mean', 'median', 'std'],
        'calories_burned': ['mean', 'median', 'std'],
        'heart_rate_avg': ['mean', 'median', 'std'],
        'sleep_hours': 'mean'
    }).round(2)
    
    st.subheader("📊 Activity Comparison")
    
    # Metrics by activity
    activity_summary = df.groupby('activity_type').agg({
        'steps': 'mean',
        'calories_burned': 'mean',
        'heart_rate_avg': 'mean',
        'user_id': 'count'
    }).round(0)
    activity_summary.columns = ['Avg Steps', 'Avg Calories', 'Avg HR', 'Count']
    activity_summary = activity_summary.sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            activity_summary.reset_index(),
            x='activity_type',
            y='Avg Steps',
            title='Average Steps by Activity',
            color='Avg Steps',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            activity_summary.reset_index(),
            x='activity_type',
            y='Avg Calories',
            title='Average Calories by Activity',
            color='Avg Calories',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    # Scatter plots
    st.subheader("🔍 Relationships Between Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='steps',
            y='calories_burned',
            color='activity_type',
            title='Steps vs Calories Burned',
            opacity=0.6,
            trendline="ols"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.scatter(
            df,
            x='heart_rate_avg',
            y='calories_burned',
            color='activity_type',
            title='Heart Rate vs Calories Burned',
            opacity=0.6
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    # Box plots
    st.subheader("📦 Distribution by Activity")
    
    metric_choice = st.selectbox(
        "Select metric to visualize:",
        ['steps', 'calories_burned', 'heart_rate_avg', 'sleep_hours']
    )
    
    fig = px.box(
        df,
        x='activity_type',
        y=metric_choice,
        title=f'{metric_choice.replace("_", " ").title()} Distribution by Activity',
        color='activity_type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False
    )
    st.plotly_chart(fig, width='stretch')

# TAB 3: Health Metrics
with tab3:
    st.header("Health Metrics Analysis")
    
    # Correlation matrix
    st.subheader("🔗 Correlation Matrix")
    
    numeric_cols = ['steps', 'calories_burned', 'heart_rate_avg', 'sleep_hours']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Feature Correlations',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(fig, width='stretch')
    
    # Heart rate analysis
    st.subheader("💓 Heart Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hr_by_activity = df.groupby('activity_type')['heart_rate_avg'].mean().sort_values(ascending=False)
        fig = px.bar(
            hr_by_activity,
            title='Average Heart Rate by Activity',
            color=hr_by_activity.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Activity",
            yaxis_title="Avg Heart Rate (bpm)"
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.histogram(
            df,
            x='heart_rate_avg',
            nbins=30,
            title='Heart Rate Distribution',
            color_discrete_sequence=['#EF553B']
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    # Sleep analysis
    st.subheader("😴 Sleep Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='sleep_hours',
            nbins=20,
            title='Sleep Duration Distribution',
            color_discrete_sequence=['#00CC96']
        )
        fig.add_vline(x=7, line_dash="dash", line_color="yellow", annotation_text="Recommended: 7h")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        sleep_quality = df.groupby('activity_type')['sleep_hours'].mean().sort_values()
        fig = px.bar(
            sleep_quality,
            title='Average Sleep by Activity Type',
            color=sleep_quality.values,
            color_continuous_scale='Teal'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Activity",
            yaxis_title="Avg Sleep Hours"
        )
        st.plotly_chart(fig, width='stretch')

# TAB 4: Temporal Patterns
with tab4:
    st.header("Temporal Patterns")
    
    # Daily trends
    st.subheader("📅 Daily Activity Trends")
    
    daily_stats = df.groupby('date').agg({
        'steps': 'mean',
        'calories_burned': 'mean',
        'heart_rate_avg': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['steps'],
        name='Steps',
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['calories_burned'],
        name='Calories',
        mode='lines',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Daily Averages Over Time',
        yaxis=dict(title='Steps'),
        yaxis2=dict(title='Calories', overlaying='y', side='right'),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')
    
    # Day of week analysis
    st.subheader("📆 Day of Week Patterns")
    
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats = df.groupby('day_of_week').agg({
        'steps': 'mean',
        'calories_burned': 'mean',
        'user_id': 'count'
    }).reindex(day_order)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            dow_stats.reset_index(),
            x='day_of_week',
            y='steps',
            title='Average Steps by Day of Week',
            color='steps',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            dow_stats.reset_index(),
            x='day_of_week',
            y='user_id',
            title='Activity Count by Day of Week',
            color='user_id',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis_title="Count"
        )
        st.plotly_chart(fig, width='stretch')

# TAB 5: User Segmentation
with tab5:
    st.header("🧬 User Segmentation Analysis")
    st.markdown("K-Means clustering to identify user personas based on behavior patterns")
    
    with st.spinner("Running user segmentation..."):
        user_segments = get_user_segments(df)
    
    if not user_segments.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌐 3D Cluster Visualization")
            fig_3d = px.scatter_3d(
                user_segments,
                x='avg_steps',
                y='avg_calories',
                z='avg_hr',
                color='prediction',
                title="User Segments in 3D Space",
                labels={'prediction': 'Cluster'},
                hover_data=['user_id']
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white"),
                    yaxis=dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white"),
                    zaxis=dict(backgroundcolor="rgb(20, 24, 35)", gridcolor="white")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=600
            )
            st.plotly_chart(fig_3d, width='stretch')
        
        with col2:
            st.subheader("📊 Cluster Statistics")
            cluster_stats = user_segments.groupby('prediction').agg({
                'avg_steps': 'mean',
                'avg_calories': 'mean',
                'avg_hr': 'mean',
                'user_id': 'count'
            }).round(1)
            cluster_stats.columns = ['Steps', 'Calories', 'HR', 'Users']
            st.dataframe(
                cluster_stats.style.background_gradient(cmap='Blues'),
                height=400
            )
        
        # Cluster comparison
        st.subheader("📈 Cluster Comparison")
        
        cluster_comparison = user_segments.groupby('prediction').agg({
            'avg_steps': 'mean',
            'avg_calories': 'mean',
            'avg_hr': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        for metric in ['avg_steps', 'avg_calories', 'avg_hr']:
            fig.add_trace(go.Bar(
                name=metric.replace('avg_', '').title(),
                x=cluster_comparison['prediction'],
                y=cluster_comparison[metric],
                text=cluster_comparison[metric].round(0),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Average Metrics by Cluster',
            barmode='group',
            xaxis_title='Cluster',
            yaxis_title='Value',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, width='stretch')
        
        # Cluster insights
        st.subheader("💡 Cluster Insights & Recommendations")
        
        for cluster_id in sorted(user_segments['prediction'].unique()):
            cluster_data = user_segments[user_segments['prediction'] == cluster_id]
            avg_steps = cluster_data['avg_steps'].mean()
            avg_cal = cluster_data['avg_calories'].mean()
            
            with st.expander(f"Cluster {cluster_id}: {len(cluster_data)} users"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Steps", f"{avg_steps:,.0f}")
                col2.metric("Avg Calories", f"{avg_cal:,.0f}")
                col3.metric("Users", len(cluster_data))
                
                # Recommendations
                if avg_steps < 5000:
                    st.info("🔵 **Sedentary Users** - Focus on basic engagement and motivation")
                elif avg_steps < 8000:
                    st.success("🟢 **Moderate Users** - Encourage consistency and gradual increases")
                else:
                    st.warning("🟡 **Active Users** - Provide advanced challenges and goals")
    else:
        st.error("Unable to perform user segmentation")