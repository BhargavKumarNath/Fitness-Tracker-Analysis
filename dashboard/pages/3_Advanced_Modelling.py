import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.utils import load_dataset, load_user_segmentation_model, load_inference_models, get_user_segments
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Advanced Modeling", page_icon="🧠", layout="wide")

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("dashboard/style.css")
except:
    pass

st.title("🧠 Advanced Machine Learning Models")
st.markdown("Deep dive into model performance, feature importance, and interactive evaluation.")

# Load Data
df = load_dataset()

if df.empty:
    st.warning("Data not available. Please ensure the ETL pipeline has run.")
    st.stop()

# Helper: Sample Data for Performance
# We don't want to compute silhouette score on 1M rows on the fly
SAMPLE_SIZE = 2000
df_sample = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=42)

tab1, tab2, tab3 = st.tabs(["🧬 User Segmentation", "🏃 Activity Classification", "🔥 Calorie Regression"])

# --- Tab 1: User Segmentation ---
with tab1:
    st.header("User Segmentation Analysis (K-Means)")
    
    pipeline, features = load_user_segmentation_model()
    
    if pipeline:
        st.success("✅ Model Loaded Successfully")
        
        # Precompute segments for the sample
        # We need the user-level aggregation for the sample
        user_df_sample = get_user_segments(df_sample) # This function aggregates by user
        
        if not user_df_sample.empty:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Distribution")
                fig_dist = px.pie(user_df_sample, names='prediction', title='User Segments Distribution (Sample)')
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col2:
                st.subheader("Cluster Centroids (Scaled)")
                # Extract centroids
                kmeans = pipeline.named_steps['kmeans']
                centroids = kmeans.cluster_centers_
                feature_names = features
                
                # Plotting centroids
                fig_radar = go.Figure()
                for i in range(len(centroids)):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=centroids[i],
                        theta=feature_names,
                        fill='toself',
                        name=f'Cluster {i}'
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Centroid Features Pattern")
                st.plotly_chart(fig_radar, use_container_width=True)

            # Interactive Metric Calculation
            if st.button("Calculate Silhouette Score (On Sample)", type="primary"):
                with st.spinner("Calculating..."):
                    # We need to transform the data first using the pipeline's scaler
                    # Accessing transformation steps
                    preprocessor = pipeline[:-1] # All steps except the last (kmeans)
                    X_transformed = preprocessor.transform(user_df_sample[features])
                    labels = user_df_sample['prediction']
                    score = silhouette_score(X_transformed, labels)
                    st.metric("Silhouette Score (Higher is better)", f"{score:.3f}")
                    if score > 0.5:
                        st.success("Good cluster separation.")
                    elif score > 0.2:
                        st.warning("Moderate cluster overlapping.")
                    else:
                        st.error("Poor cluster separation.")
        
    else:
        st.error("Model not found. Please check connectivity or model files.")

# --- Tab 2: Classification ---
with tab2:
    st.header("Activity Classification (Random Forest)")
    
    class_model, _ = load_inference_models()
    
    if class_model:
        st.success("✅ Classifier Loaded")
        
        # Feature Importance
        st.subheader("Feature Importance")
        try:
            # Assuming Random Forest is the last step named 'classifier' or 'randomforestclassifier'
            # Check pipeline steps
            if hasattr(class_model, 'named_steps'):
                classifier = class_model.named_steps['classifier']
                importances = classifier.feature_importances_
                # We need feature names. If not preserved, we assume the input columns order:
                # ['steps', 'calories_burned', 'heart_rate_avg'] based on training script
                feature_names_class = ['steps', 'calories_burned', 'heart_rate_avg']
                
                fig_imp = px.bar(x=feature_names_class, y=importances, title="Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.info(f"Could not extract feature importance directly: {e}")

        # Interactive Confusion Matrix
        if st.checkbox("Run Evaluation on Sample Data"):
            with st.spinner("Running Inference on Sample..."):
                 # Prepare X and y
                 X_test = df_sample[['steps', 'calories_burned', 'heart_rate_avg']]
                 y_test = df_sample['activity_type']
                 
                 y_pred = class_model.predict(X_test)
                 
                 # Confusion Matrix
                 labels = sorted(y_test.unique())
                 cm = confusion_matrix(y_test, y_pred, labels=labels)
                 
                 fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
                 st.plotly_chart(fig_cm, use_container_width=True)
                 
                 # Report
                 report = classification_report(y_test, y_pred, output_dict=True)
                 st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.error("Classification model not found.")

# --- Tab 3: Regression ---
with tab3:
    st.header("Calorie Prediction (Regression)")
    
    _, reg_model = load_inference_models()
    
    if reg_model:
        st.success("✅ Regressor Loaded")
        
        if st.checkbox("Run Regression Evaluation"):
             with st.spinner("Predicting..."):
                 # Features: ['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']
                 X_reg = df_sample[['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']]
                 y_true = df_sample['calories_burned']
                 
                 y_pred = reg_model.predict(X_reg)
                 
                 # Metrics
                 rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                 r2 = r2_score(y_true, y_pred)
                 
                 m1, m2 = st.columns(2)
                 m1.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f}")
                 m2.metric("R² Score", f"{r2:.3f}")
                 
                 # Scatter Plot
                 fig_reg = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual Calories', 'y': 'Predicted Calories'}, title="Actual vs Predicted")
                 fig_reg.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(color="Red", dash="dash"))
                 st.plotly_chart(fig_reg, use_container_width=True)
                 
                 # Residuals
                 residuals = y_true - y_pred
                 fig_res = px.histogram(residuals, title="Residual Distribution", nbins=30)
                 st.plotly_chart(fig_res, use_container_width=True)
                 
    else:
        st.error("Regression model not found.")
