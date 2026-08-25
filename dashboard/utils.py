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


def get_model_path(model_name: str) -> Path:
    """Return a model artifact path for the current runtime root."""
    if Path(model_name).name != model_name or not model_name.isidentifier():
        raise ValueError(f"Invalid model name: {model_name}")
    return get_runtime_paths()["models_dir"] / f"{model_name}.pkl"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the processed dataset from Parquet using repo-root-aware paths."""
    data_path = get_runtime_paths()["processed_data_dir"]
    try:
        if data_path.exists():
            return pd.read_parquet(data_path)
        st.error(f"Data not found at {data_path}. Please run the ETL pipeline.")
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

            output = gdown.download(url, dest_path, quiet=False, verify=False)

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
def load_user_segmentation_model(download_if_missing: bool = False):
    """Download and load the user segmentation model if it is not already present."""
    model_path = get_model_path("user_segmentation")
    features_path = get_model_path("cluster_features")

    if download_if_missing and not os.path.exists(model_path):
        with st.spinner("Downloading user segmentation model..."):
            if not download_file_from_google_drive(MODEL_URLS["user_segmentation"], model_path):
                return None, None

    if download_if_missing and not os.path.exists(features_path):
        with st.spinner("Downloading cluster features..."):
            if not download_file_from_google_drive(MODEL_URLS["cluster_features"], features_path):
                return None, None

    if not model_path.exists() or not features_path.exists():
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
    class_model_path = get_model_path("activity_classifier")

    if not os.path.exists(class_model_path):
        st.warning("Activity classifier is not available locally. Using the fast baseline.")
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
    reg_model_path = get_model_path("calories_regressor")

    if not os.path.exists(reg_model_path):
        st.warning("Calorie regressor is not available locally. Using the fast baseline.")
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
    """Perform user segmentation without blocking on remote model downloads."""
    pipeline, features = load_user_segmentation_model()

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

    if pipeline is None:
        return _build_baseline_segments(user_summary_df)

    try:
        predictions = pipeline.predict(user_summary_df[features])
        user_summary_df["prediction"] = predictions
        return user_summary_df
    except Exception as exc:
        st.error(f"Error during segmentation: {exc}")
        return pd.DataFrame()


def _build_baseline_segments(user_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic activity bands when the optional model is absent."""
    if user_summary_df.empty:
        return user_summary_df.assign(prediction=pd.Series(dtype="int64"))

    ranked_steps = user_summary_df["avg_steps"].rank(method="first")
    cluster_count = min(5, len(user_summary_df))
    user_summary_df = user_summary_df.copy()
    user_summary_df["prediction"] = (
        ((ranked_steps - 1) * cluster_count / len(user_summary_df))
        .astype(int)
        .clip(upper=cluster_count - 1)
    )
    return user_summary_df


def get_classifier_model():
    """Lazy-load the classifier only when needed."""
    class_model, _ = load_inference_models()
    return class_model


def get_regressor_model():
    """Lazy-load the regressor only when needed."""
    _, reg_model = load_inference_models()
    return reg_model


def predict_activity_baseline(steps: int, heart_rate: int) -> str:
    """Return an immediate activity estimate when the classifier is unavailable."""
    if heart_rate >= 150 or steps >= 15000:
        return "running"
    if heart_rate >= 125 or steps >= 8000:
        return "cycling"
    if steps == 0 and heart_rate < 90:
        return "yoga"
    return "walking"


def predict_calories_baseline(
    steps: int, heart_rate: int, sleep_hours: float, activity_type: str
) -> float:
    """Estimate calories without loading a remote or oversized model artifact."""
    activity_factor = {
        "Walking": 1.0,
        "Running": 1.35,
        "Cycling": 1.2,
        "Yoga": 0.75,
        "HIIT": 1.45,
        "Strength Training": 1.15,
    }.get(activity_type, 1.0)
    sleep_adjustment = max(0.85, min(1.1, 1.0 + (7.5 - sleep_hours) * 0.02))
    return max(50.0, (80.0 + steps * 0.035 + max(0, heart_rate - 60) * 1.5) * activity_factor * sleep_adjustment)
