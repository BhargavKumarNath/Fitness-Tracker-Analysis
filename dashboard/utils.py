import pandas as pd
import joblib
import os
import streamlit as st
import time
from pathlib import Path
import urllib3
import gc

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL_URLS = {
    "user_segmentation": "1HfWsKfx0hATxI4SonagapM-Zeu6FzJoD",
    "cluster_features": "1eyaogp73oMSKzfw08zoSdPCNYRyLTXhQ",
    "activity_classifier": "1pMa9zmnnAn0xN41NnqNPUHGDzV87mN7Y", 
    "calories_regressor": "1S99cJb-_KkS7WhNZ0Bvmgiia8bZq-muF"
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
    """Downloads a file from Google Drive using gdown with retry logic."""
    for attempt in range(max_retries):
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={id}'
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            progress_text = f"Downloading {os.path.basename(dest_path)} (attempt {attempt + 1}/{max_retries})..."
            progress_bar = st.progress(0, text=progress_text)
            
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
    """Downloads (if needed) and loads the KMeans clustering model."""
    model_path = os.path.join(MODELS_DIR, "user_segmentation.pkl")
    features_path = os.path.join(MODELS_DIR, "cluster_features.pkl")
    
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


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_classifier_model_internal():
    """Internal function to load classifier - cached separately."""
    class_model_path = os.path.join(MODELS_DIR, "activity_classifier.pkl")
    
    if not os.path.exists(class_model_path):
        with st.spinner("Downloading activity classifier (745 MB, may take 2-3 minutes)..."):
            if not download_file_from_google_drive(MODEL_URLS["activity_classifier"], class_model_path):
                st.error("Failed to download activity classifier")
                return None
    
    if os.path.exists(class_model_path):
        try:
            # Check file size before loading
            file_size_mb = os.path.getsize(class_model_path) / (1024 * 1024)
            st.info(f"Loading classifier model ({file_size_mb:.1f} MB)...")
            
            model = joblib.load(class_model_path)
            st.success("✓ Activity classifier loaded")
            return model
        except MemoryError:
            st.error("⚠️ Not enough memory to load classifier model. Streamlit Cloud has limited resources.")
            return None
        except Exception as e:
            st.error(f"Error loading classifier: {e}")
            return None
    return None


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_regressor_model_internal():
    """Internal function to load regressor - cached separately."""
    reg_model_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")
    
    if not os.path.exists(reg_model_path):
        with st.spinner("Downloading calorie regressor (3 GB - this will take several minutes)..."):
            if not download_file_from_google_drive(MODEL_URLS["calories_regressor"], reg_model_path):
                st.error("Failed to download calorie regressor")
                return None
    
    if os.path.exists(reg_model_path):
        try:
            file_size_mb = os.path.getsize(reg_model_path) / (1024 * 1024)
            
            if file_size_mb > 2000:  
                st.error(f"⚠️ Calorie regressor model is too large ({file_size_mb:.1f} MB) for Streamlit Cloud's memory limits.")
                st.info("💡 This model works locally but cannot run on Streamlit Cloud's free tier (1GB RAM limit).")
                return None
            
            st.info(f"Loading regressor model ({file_size_mb:.1f} MB)...")
            model = joblib.load(reg_model_path)
            st.success("✓ Calorie regressor loaded")
            return model
        except MemoryError:
            st.error("⚠️ Not enough memory to load regressor model. This model is too large for Streamlit Cloud.")
            st.info("💡 Consider using a lighter model or deploying to a platform with more resources.")
            return None
        except Exception as e:
            st.error(f"Error loading regressor: {e}")
            return None
    return None


def load_inference_models():
    """
    Loads models with memory-aware strategy.
    Returns (classifier, regressor) tuple.
    """
    class_model = None
    reg_model = None
    
    # Load classifier first (smaller)
    class_model = _load_classifier_model_internal()
    
    # Only attempt regressor if classifier succeeded
    if class_model is not None:
        # Force garbage collection before loading large model
        gc.collect()
        reg_model = _load_regressor_model_internal()
    
    return class_model, reg_model


def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Performs user segmentation using the pre-trained KMeans model."""
    pipeline, features = load_user_segmentation_model()
    if pipeline is None:
        return pd.DataFrame()

    user_summary_df = df.groupby("user_id").agg({
        "steps": "mean",
        "calories_burned": "mean",
        "heart_rate_avg": "mean"
    }).rename(columns={
        "steps": "avg_steps",
        "calories_burned": "avg_calories",
        "heart_rate_avg": "avg_hr"
    }).reset_index()

    try:
        predictions = pipeline.predict(user_summary_df[features])
        user_summary_df["prediction"] = predictions
        return user_summary_df
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        return pd.DataFrame()


def get_classifier_model():
    """Lazy load classifier only when needed."""
    class_model, _ = load_inference_models()
    return class_model


def get_regressor_model():
    """Lazy load regressor only when needed."""
    _, reg_model = load_inference_models()
    return reg_model