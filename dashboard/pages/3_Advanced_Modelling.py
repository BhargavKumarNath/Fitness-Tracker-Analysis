import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.utils import load_dataset, load_user_segmentation_model, get_user_segments, get_classifier_model, get_regressor_model
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="Advanced Modeling", page_icon="🧠", layout="wide")

# Load CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css("dashboard/style.css")

st.title("🧠 Advanced Machine Learning Models")
st.markdown("Deep dive into model performance, feature importance, and interactive evaluation.")

if 'STREAMLIT_SHARING_MODE' in os.environ or os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud':
    st.warning("⚠️ **Running on Streamlit Cloud**: Large models (>1GB) may fail to load due to memory limits. User segmentation and classification models should work fine.")

# Load Data with error handling
with st.spinner("Loading dataset..."):
    df = load_dataset()

if df.empty:
    st.error("⚠️ Data not available. Please ensure the ETL pipeline has run.")
    st.stop()

# Sample for performance
SAMPLE_SIZE = 2000
df_sample = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=42)

# Create tabs for different models
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 User Segmentation", 
    "🏃 Activity Classification", 
    "🔥 Calorie Regression",
    "📊 Model Comparison"
])

# TAB 1: User Segmentation (Small model - should always work)
with tab1:
    st.header("User Segmentation Analysis (K-Means)")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Algorithm", "K-Means Clustering")
    with col_info2:
        st.metric("Features", "3 (steps, calories, HR)")
    with col_info3:
        st.metric("Clusters", "5")
    
    st.divider()
    
    pipeline, features = load_user_segmentation_model()
    
    if pipeline:
        user_df_sample = get_user_segments(df_sample)
        
        if not user_df_sample.empty:
            
            # Visualization row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Cluster Distribution")
                cluster_counts = user_df_sample['prediction'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig_dist = px.pie(
                    cluster_counts, 
                    names='Cluster', 
                    values='Count',
                    title='User Distribution Across Clusters',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="white")
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col2:
                st.subheader("🎯 Cluster Characteristics")
                cluster_stats = user_df_sample.groupby('prediction').agg({
                    'avg_steps': 'mean',
                    'avg_calories': 'mean',
                    'avg_hr': 'mean'
                }).round(1)
                
                fig_bar = go.Figure()
                for col in ['avg_steps', 'avg_calories', 'avg_hr']:
                    fig_bar.add_trace(go.Bar(
                        name=col.replace('avg_', '').title(),
                        x=cluster_stats.index,
                        y=cluster_stats[col],
                        text=cluster_stats[col].round(0),
                        textposition='auto'
                    ))
                
                fig_bar.update_layout(
                    title='Average Metrics per Cluster',
                    barmode='group',
                    xaxis_title='Cluster',
                    yaxis_title='Value',
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # 3D Visualization
            st.subheader("🌐 3D Cluster Visualization")
            fig_3d = px.scatter_3d(
                user_df_sample, 
                x='avg_steps', 
                y='avg_calories', 
                z='avg_hr',
                color='prediction',
                title="User Segmentation in 3D Feature Space",
                labels={'prediction': 'Cluster'},
                color_continuous_scale='Viridis'
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
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cluster insights
            st.subheader("💡 Cluster Insights")
            for cluster_id in sorted(user_df_sample['prediction'].unique()):
                cluster_data = user_df_sample[user_df_sample['prediction'] == cluster_id]
                avg_steps = cluster_data['avg_steps'].mean()
                avg_cal = cluster_data['avg_calories'].mean()
                avg_hr = cluster_data['avg_hr'].mean()
                
                with st.expander(f"Cluster {cluster_id} Details ({len(cluster_data)} users)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Steps", f"{avg_steps:,.0f}")
                    c2.metric("Avg Calories", f"{avg_cal:,.0f}")
                    c3.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")
                    
                    if avg_steps < 6000:
                        st.info("🔵 **Low Activity Group** - Focus on engagement strategies")
                    elif avg_steps < 8000:
                        st.success("🟢 **Moderate Activity Group** - Encourage consistency")
                    else:
                        st.warning("🟡 **High Performers** - Challenge with advanced goals")
    else:
        st.error("⚠️ Segmentation model not available")

# TAB 2: Classification 
with tab2:
    st.header("Activity Classification (Random Forest)")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Algorithm", "Random Forest")
    with col_info2:
        st.metric("Accuracy", "84%")
    with col_info3:
        st.metric("Features", "3")
    
    st.divider()
    
    try:
        class_model = get_classifier_model()
        
        if class_model:
            st.success("✅ Classifier Loaded Successfully")
            
            # Feature Importance
            st.subheader("📊 Feature Importance")
            try:
                if hasattr(class_model, 'named_steps'):
                    classifier = class_model.named_steps['classifier']
                    importances = classifier.feature_importances_
                    feature_names = ['steps', 'calories_burned', 'heart_rate_avg']
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance for Activity Prediction",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig_imp.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info(f"Feature importance visualization unavailable: {e}")

            # Evaluation
            if st.checkbox("🔍 Run Model Evaluation on Sample Data"):
                with st.spinner("Running predictions..."):
                    try:
                        X_test = df_sample[['steps', 'calories_burned', 'heart_rate_avg']]
                        y_test = df_sample['activity_type']
                        
                        y_pred = class_model.predict(X_test)
                        
                        # Metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📈 Confusion Matrix")
                            labels = sorted(y_test.unique())
                            cm = confusion_matrix(y_test, y_pred, labels=labels)
                            
                            fig_cm = px.imshow(
                                cm,
                                x=labels,
                                y=labels,
                                text_auto=True,
                                title="Confusion Matrix",
                                color_continuous_scale='Blues',
                                labels=dict(x="Predicted", y="Actual")
                            )
                            fig_cm.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white")
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        with col2:
                            st.subheader("📋 Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(
                                report_df.style.background_gradient(cmap='Blues'),
                                height=400
                            )
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")
        else:
            st.error("⚠️ Classification model could not be loaded")
            st.info("💡 This may be due to memory constraints on Streamlit Cloud. The model works fine locally.")
    except Exception as e:
        st.error(f"⚠️ Error loading classifier: {e}")

# TAB 3: Regression
with tab3:
    st.header("Calorie Prediction (Regression)")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Algorithm", "Random Forest Regressor")
    with col_info2:
        st.metric("R² Score", "0.91")
    with col_info3:
        st.metric("RMSE", "131")
    
    st.divider()
    
    try:
        reg_model = get_regressor_model()
        
        if reg_model:
            st.success("✅ Regressor Loaded Successfully")
            
            if st.checkbox("🔍 Run Regression Evaluation"):
                with st.spinner("Making predictions..."):
                    try:
                        X_reg = df_sample[['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']]
                        y_true = df_sample['calories_burned']
                        
                        y_pred = reg_model.predict(X_reg)
                        
                        # Metrics
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("RMSE", f"{rmse:.2f}")
                        m2.metric("R² Score", f"{r2:.3f}")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Actual vs Predicted")
                            fig_scatter = px.scatter(
                                x=y_true,
                                y=y_pred,
                                labels={'x': 'Actual Calories', 'y': 'Predicted Calories'},
                                title="Prediction Accuracy",
                                opacity=0.6,
                                color=y_true-y_pred,
                                color_continuous_scale='RdYlGn_r'
                            )
                            fig_scatter.add_shape(
                                type="line",
                                x0=y_true.min(),
                                y0=y_true.min(),
                                x1=y_true.max(),
                                y1=y_true.max(),
                                line=dict(color="Red", dash="dash")
                            )
                            fig_scatter.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white")
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with col2:
                            st.subheader("📉 Residual Distribution")
                            residuals = y_true - y_pred
                            fig_hist = px.histogram(
                                residuals,
                                nbins=50,
                                title="Error Distribution",
                                labels={'value': 'Residual (Actual - Predicted)'},
                                color_discrete_sequence=['#636EFA']
                            )
                            fig_hist.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white")
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Residual plot
                        st.subheader("🎯 Residual Plot")
                        fig_residual = px.scatter(
                            x=y_pred,
                            y=residuals,
                            labels={'x': 'Predicted Calories', 'y': 'Residuals'},
                            title="Residuals vs Predicted Values",
                            opacity=0.6
                        )
                        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_residual.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white")
                        )
                        st.plotly_chart(fig_residual, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")
        else:
            st.error("⚠️ Regression model could not be loaded")
            st.info("💡 This 3GB model exceeds Streamlit Cloud's memory limits. It works fine when running locally.")
            st.markdown("""
            **Alternatives for Cloud Deployment:**
            1. Train a smaller model (e.g., reduce `n_estimators`)
            2. Use a simpler algorithm (e.g., LinearRegression)
            3. Deploy to a platform with more resources (AWS, GCP, Azure)
            """)
    except Exception as e:
        st.error(f"⚠️ Error loading regressor: {e}")

# TAB 4: Model Comparison
with tab4:
    st.header("📊 Model Performance Comparison")
    
    # Model metrics summary
    metrics_data = {
        'Model': ['User Segmentation', 'Activity Classification', 'Calorie Regression'],
        'Algorithm': ['K-Means', 'Random Forest', 'Random Forest'],
        'Primary Metric': ['Silhouette Score', 'Accuracy', 'R² Score'],
        'Score': [0.45, 0.84, 0.91],
        'Training Time': ['< 1 min', '~5 mins', '~10 mins'],
        'Model Size': ['10 KB', '745 MB', '3 GB'],
        'Cloud Compatible': ['✅ Yes', '⚠️ Maybe', '❌ No']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    st.subheader("Model Overview")
    st.dataframe(metrics_df, use_container_width=True)
    
    st.info("""
    **Note on Cloud Deployment:**
    - ✅ **User Segmentation** (10 KB): Always works
    - ⚠️ **Activity Classification** (745 MB): Works with caching, may timeout on first load
    - ❌ **Calorie Regression** (3 GB): Exceeds Streamlit Cloud's 1GB RAM limit
    """)
    
    # Performance comparison chart
    st.subheader("Performance Scores")
    fig_comparison = px.bar(
        metrics_df,
        x='Model',
        y='Score',
        color='Model',
        title='Model Performance Comparison',
        text='Score',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_comparison.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_comparison.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Use cases
    st.subheader("💡 Recommended Use Cases")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 🧬 User Segmentation")
            st.write("""
            **Best for:**
            - Marketing campaigns
            - Personalized recommendations
            - Targeted engagement
            - User profiling
            """)
    
    with col2:
        with st.container():
            st.markdown("### 🏃 Activity Classification")
            st.write("""
            **Best for:**
            - Automatic activity detection
            - Workout logging
            - Activity recommendations
            - Fitness tracking apps
            """)
    
    with col3:
        with st.container():
            st.markdown("### 🔥 Calorie Prediction")
            st.write("""
            **Best for:**
            - Nutrition planning
            - Calorie goal setting
            - Weight management
            - Diet recommendations
            """)