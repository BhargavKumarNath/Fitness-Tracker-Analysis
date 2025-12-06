import pandas as pd
import joblib
import os
import streamlit as st
import requests
import time
from pathlib import Path
import urllib3
import hashlib

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL_URLS = {
    "user_segmentation": "1HfWsKfx0hATxI4SonagapM-Zeu6FzJoD",
    "cluster_features": "1eyaogp73oMSKzfw08zoSdPCNYRyLTXhQ",
    "activity_classifier": "1pMa9zmnnAn0xN41NnqNPUHGDzV87mN7Y", 
    "calories_regressor": "1qd4m_l55ueKQdHi6feMSEDZS-iUMnarR"
}

# Define paths
MODELS_DIR = "dashboard/models"
DATA_PATH = "data_lake/processed/fitness_data"

@st.cache_data
def load_dataset():
    """Loads the processed dataset from Parquet."""
    try:
        if os.path.exists(DATA_PATH):
           df = pd.read_parquet(DATA_PATH)
           return df
        else:
           st.error(f"Data not found at {DATA_PATH}. Please run the ETL pipeline.")
           return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def download_file_from_google_drive(id: str, dest_path: str, max_retries: int = 3):
    """
    Downloads a file from Google Drive using gdown with retry logic.
    """
    for attempt in range(max_retries):
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={id}'
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Show progress in Streamlit
            progress_text = f"Downloading {os.path.basename(dest_path)} (attempt {attempt + 1}/{max_retries})..."
            progress_bar = st.progress(0, text=progress_text)
            
            # Download with gdown
            output = gdown.download(url, dest_path, quiet=False, verify=False, fuzzy=True)
            
            progress_bar.progress(100, text=f"✓ Downloaded {os.path.basename(dest_path)}")
            time.sleep(0.5)
            progress_bar.empty()
            
            if output and os.path.exists(dest_path):
                return True
            else:
                st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
                
        except Exception as e:
            st.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to download {os.path.basename(dest_path)} after {max_retries} attempts")
                return False
    
    return False


@st.cache_resource(show_spinner=False)
def load_user_segmentation_model():
    """
    Downloads (if needed) and loads the KMeans clustering model.
    """
    model_path = os.path.join(MODELS_DIR, "user_segmentation.pkl")
    features_path = os.path.join(MODELS_DIR, "cluster_features.pkl")
    
    # Check if models exist, download if needed
    if not os.path.exists(model_path):
        with st.spinner("Downloading user segmentation model..."):
            if not download_file_from_google_drive(MODEL_URLS["user_segmentation"], model_path):
                return None, None
    
    if not os.path.exists(features_path):
        with st.spinner("Downloading cluster features..."):
            if not download_file_from_google_drive(MODEL_URLS["cluster_features"], features_path):
                return None, None

    try:
        with st.spinner("Loading segmentation model..."):
            pipeline = joblib.load(model_path)
            features = joblib.load(features_path)
            st.success("✓ Segmentation model loaded successfully")
            return pipeline, features
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def load_inference_models():
    """
    Downloads (if needed) and loads the classification and regression models.
    Uses lazy loading to avoid memory issues.
    """
    class_model_path = os.path.join(MODELS_DIR, "activity_classifier.pkl")
    reg_model_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")

    class_model = None
    reg_model = None
    
    # Load classification model
    if not os.path.exists(class_model_path):
        with st.spinner("Downloading activity classifier (this may take a minute)..."):
            if not download_file_from_google_drive(MODEL_URLS["activity_classifier"], class_model_path):
                st.error("Failed to download activity classifier")
    
    if os.path.exists(class_model_path):
        try:
            with st.spinner("Loading activity classifier..."):
                class_model = joblib.load(class_model_path)
                st.success("✓ Activity classifier loaded")
        except Exception as e:
            st.error(f"Error loading classifier: {e}")
    
    # Load regression model
    if not os.path.exists(reg_model_path):
        with st.spinner("Downloading calorie regressor (this is a large file, ~3GB)..."):
            if not download_file_from_google_drive(MODEL_URLS["calories_regressor"], reg_model_path):
                st.error("Failed to download calorie regressor")
    
    if os.path.exists(reg_model_path):
        try:
            with st.spinner("Loading calorie regressor..."):
                reg_model = joblib.load(reg_model_path)
                st.success("✓ Calorie regressor loaded")
        except Exception as e:
            st.error(f"Error loading regressor: {e}")

    return class_model, reg_model


def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs user segmentation using the pre-trained KMeans model.
    """
    pipeline, features = load_user_segmentation_model()
    if pipeline is None:
        return pd.DataFrame()

    # User-level summary
    user_summary_df = df.groupby("user_id").agg({
        "steps": "mean",
        "calories_burned": "mean",
        "heart_rate_avg": "mean"
    }).rename(columns={
        "steps": "avg_steps",
        "calories_burned": "avg_calories",
        "heart_rate_avg": "avg_hr"
    }).reset_index()

    # Predict
    predictions = pipeline.predict(user_summary_df[features])
    user_summary_df["prediction"] = predictions
    
    return user_summary_df


# Lazy loading wrapper for models
def get_classifier_model():
    """Lazy load classifier only when needed"""
    class_model, _ = load_inference_models()
    return class_model


def get_regressor_model():
    """Lazy load regressor only when needed"""
    _, reg_model = load_inference_models()
    return reg_model