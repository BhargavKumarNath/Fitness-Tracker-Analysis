import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import get_runtime_paths
from src.models.training import train_dashboard_models

def train_and_save_models():
    """Train dashboard models from the configured processed dataset."""
    paths = get_runtime_paths()
    data_path = paths["processed_data_dir"]
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    df = pd.read_parquet(data_path)
    return train_dashboard_models(df, paths["models_dir"])

if __name__ == "__main__":
    train_and_save_models()
