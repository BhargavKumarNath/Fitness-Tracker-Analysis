import gc
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import urllib3

from src.config import get_runtime_paths

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL_URLS = {
    "user_segmentation": "1HfWsKfx0hATxI4SonagapM-Zeu6FzJoD",
    "cluster_features": "1eyaogp73oMSKzfw08zoSdPCNYRyLTXhQ",
    "activity_classifier": "1pMa9zmnnAn0xN41NnqNPUHGDzV87mN7Y",
    "calories_regressor": "1S99cJb-_KkS7WhNZ0Bvmgiia8bZq-muF",
}

RUNTIME_PATHS = get_runtime_paths()
MODELS_DIR = str(RUNTIME_PATHS["models_dir"])
DATA_PATH = str(RUNTIME_PATHS["processed_data_dir"])


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the processed dataset from Parquet using repo-root-aware paths."""
    try:
        if os.path.exists(DATA_PATH):
            return pd.read_parquet(DATA_PATH)
        st.error(f"Data not found at {DATA_PATH}. Please run the ETL pipeline.")
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - UI-level failure path
        st.error(f"Error loading data: {exc}")
        return pd.DataFrame()


def download_file_from_google_drive(file_id: str, dest_path: str, max_retries: int = 3) -> bool:
    """Download a file from Google Drive with a simple retry loop."""
    destination_dir = os.path.dirname(dest_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            import gdown

            url = f"https://drive.google.com/uc?id={file_id}"
            progress_text = f"Downloading {os.path.basename(dest_path)} (attempt {attempt}/{max_retries})..."
            progress_bar = st.progress(0, text=progress_text)

            output = gdown.download(url, dest_path, quiet=False, verify=False, fuzzy=True)

            progress_bar.progress(100, text=f"Downloaded {os.path.basename(dest_path)}")
            time.sleep(0.5)
            progress_bar.empty()

            if output and os.path.exists(dest_path):
                return True

            st.warning(f"Attempt {attempt} failed. Retrying...")
            time.sleep(2)
        except Exception as exc:
            st.warning(f"Download attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            st.error(f"Failed to download {os.path.basename(dest_path)} after {max_retries} attempts")
            return False

    return False


@st.cache_resource(show_spinner=False)
def load_user_segmentation_model():
    """Download and load the user segmentation model if it is not already present."""
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
            st.success("Segmentation model loaded successfully")
            return pipeline, features
    except Exception as exc:
        st.error(f"Error loading segmentation model: {exc}")
        return None, None


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_classifier_model_internal():
    """Load the classifier model, downloading it if needed."""
    class_model_path = os.path.join(MODELS_DIR, "activity_classifier.pkl")

    if not os.path.exists(class_model_path):
        with st.spinner("Downloading activity classifier..."):
            if not download_file_from_google_drive(MODEL_URLS["activity_classifier"], class_model_path):
                st.warning("Classifier model not found locally or on Drive.")
                return None

    if os.path.exists(class_model_path):
        try:
            file_size_mb = os.path.getsize(class_model_path) / (1024 * 1024)
            st.info(f"Loading classifier model ({file_size_mb:.1f} MB)...")
            model = joblib.load(class_model_path)
            st.success("Activity classifier loaded")
            return model
        except Exception as exc:
            st.error(f"Error loading classifier: {exc}")
            return None
    return None


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_regressor_model_internal():
    """Load the regressor model, downloading it if needed."""
    reg_model_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")

    if not os.path.exists(reg_model_path):
        with st.spinner("Downloading calorie regressor..."):
            if not download_file_from_google_drive(MODEL_URLS["calories_regressor"], reg_model_path):
                st.warning("Calorie regressor not found locally or on Drive.")
                return None

    if os.path.exists(reg_model_path):
        try:
            file_size_mb = os.path.getsize(reg_model_path) / (1024 * 1024)
            st.info(f"Loading regressor model ({file_size_mb:.1f} MB)...")
            model = joblib.load(reg_model_path)
            st.success("Calorie regressor loaded")
            return model
        except MemoryError:
            st.error("Not enough memory to load regressor model.")
            return None
        except Exception as exc:
            st.error(f"Error loading regressor: {exc}")
            return None
    return None


def load_inference_models():
    """Load the classifier and regressor with a memory-aware workflow."""
    class_model = None
    reg_model = None

    class_model = _load_classifier_model_internal()

    if class_model is not None:
        gc.collect()
        reg_model = _load_regressor_model_internal()

    return class_model, reg_model


def get_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Perform user segmentation via the pretrained KMeans model."""
    pipeline, features = load_user_segmentation_model()
    if pipeline is None:
        return pd.DataFrame()

    user_summary_df = (
        df.groupby("user_id")
        .agg({
            "steps": "mean",
            "calories_burned": "mean",
            "heart_rate_avg": "mean",
        })
        .rename(columns={
            "steps": "avg_steps",
            "calories_burned": "avg_calories",
            "heart_rate_avg": "avg_hr",
        })
        .reset_index()
    )

    try:
        predictions = pipeline.predict(user_summary_df[features])
        user_summary_df["prediction"] = predictions
        return user_summary_df
    except Exception as exc:
        st.error(f"Error during segmentation: {exc}")
        return pd.DataFrame()


def get_classifier_model():
    """Lazy-load the classifier only when needed."""
    class_model, _ = load_inference_models()
    return class_model


def get_regressor_model():
    """Lazy-load the regressor only when needed."""
    _, reg_model = load_inference_models()
    return reg_model
