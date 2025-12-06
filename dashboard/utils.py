import pandas as pd
import joblib
import os
import streamlit as st
import requests
import time
from pathlib import Path
import urllib3

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


def download_file_from_google_drive(id: str, dest_path: str):
    """
    Downloads a file from Google Drive using its ID, handling large file warnings.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    try:
        response = session.get(URL, params={'id': id}, stream=True, verify=False)
    except requests.exceptions.SSLError:
        # Fallback for strict environments
        response = session.get(URL, params={'id': id}, stream=True, verify=False)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True, verify=False)

    save_response_content(response, dest_path)
    return True

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, dest_path):
    CHUNK_SIZE = 32768
    
    # Calculate total size if available (often not available for chunked GDrive)
    total_length = response.headers.get('content-length')
    
    with open(dest_path, "wb") as f:
        if total_length is None: # no content length header
            progress_bar = st.progress(0, text=f"Downloading {os.path.basename(dest_path)} (size unknown)...")
            downloaded = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Just animate generically or update MB count
                    progress_bar.progress(min(downloaded % 100, 100), text=f"Downloading {os.path.basename(dest_path)}: {downloaded/1024/1024:.1f} MB")
            progress_bar.empty()
        else:
            total_length = int(total_length)
            downloaded = 0
            progress_bar = st.progress(0, text=f"Downloading {os.path.basename(dest_path)}...")
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = int(downloaded * 100 / total_length)
                    progress_bar.progress(percent, text=f"Downloading {os.path.basename(dest_path)}: {percent}%")
            progress_bar.empty()

def download_file_from_cloud(url_or_id: str, dest_path: str):
    """
    Router for downloading files. Checks if it's a Drive ID or URL.
    """
    if os.path.exists(dest_path):
        return True # File already exists

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if url_or_id.startswith("http"):
             # Standard URL download logic (Simplified from before)
             headers = {'User-Agent': 'Mozilla/5.0'}
             with requests.get(url_or_id, stream=True, headers=headers) as r:
                r.raise_for_status()
                save_response_content(r, dest_path)
             return True
        else:
             # Assume Google Drive ID
             return download_file_from_google_drive(url_or_id, dest_path)
             
    except Exception as e:
        st.error(f"Failed to download {os.path.basename(dest_path)}: {e}")
        return False

@st.cache_resource(show_spinner="Loading User Segmentation Model...")
def load_user_segmentation_model():
    """
    Downloads (if needed) and loads the KMeans clustering model.
    Cached resource to avoid reloading large pickle files in memory.
    """
    model_path = os.path.join(MODELS_DIR, "user_segmentation.pkl")
    features_path = os.path.join(MODELS_DIR, "cluster_features.pkl")
    
    # Attempt download if not present
    if not os.path.exists(model_path):
        if "example.com" in MODEL_URLS["user_segmentation"]:
             # Fallback check for local dev/testing without download
             if not os.path.exists(model_path): # Double check
                return None, None
        else:
             success_m = download_file_from_cloud(MODEL_URLS["user_segmentation"], model_path)
             success_f = download_file_from_cloud(MODEL_URLS["cluster_features"], features_path)
             if not (success_m and success_f):
                 return None, None

    try:
        pipeline = joblib.load(model_path)
        features = joblib.load(features_path)
        return pipeline, features
    except FileNotFoundError:
        # Silent fail or warning handled by caller
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading Inference Models...")
def load_inference_models():
    """
    Downloads (if needed) and loads the classification and regression models.
    """
    class_model_path = os.path.join(MODELS_DIR, "activity_classifier.pkl")
    reg_model_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")

    # Attempt download if not present
    if not (os.path.exists(class_model_path) and os.path.exists(reg_model_path)):
         if "PLACEHOLDER" in MODEL_URLS["activity_classifier"]:
             # Fallback check for local
             if not (os.path.exists(class_model_path) and os.path.exists(reg_model_path)):
                 return None, None
         else:
             download_file_from_cloud(MODEL_URLS["activity_classifier"], class_model_path)
             download_file_from_cloud(MODEL_URLS["calories_regressor"], reg_model_path)

    try:
        class_model = joblib.load(class_model_path)
        reg_model = joblib.load(reg_model_path)
        return class_model, reg_model
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading inference models: {e}")
        return None, None

def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs user segmentation using the pre-trained KMeans model.
    Args:
        df: The input data as a Pandas DataFrame (raw steps data).
    Returns:
        A Pandas DataFrame with user_id, aggregated metrics, and cluster prediction.
    """
    # Load model
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
