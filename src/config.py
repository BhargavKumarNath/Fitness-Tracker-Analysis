from __future__ import annotations

import os
from pathlib import Path


def get_runtime_paths() -> dict[str, Path]:
    """Return repo-root-aware paths for the active project layout."""
    project_root = Path(
        os.environ.get("FITNESS_TRACKER_ROOT", Path(__file__).resolve().parent.parent)
    ).resolve()

    data_lake_dir = project_root / "data_lake"
    raw_data_dir = data_lake_dir / "raw" / "synthetic_user_data"
    processed_data_dir = data_lake_dir / "processed" / "fitness_data"
    models_dir = project_root / "dashboard" / "models"
    streaming_input_dir = data_lake_dir / "streaming_input"

    return {
        "project_root": project_root,
        "data_lake_dir": data_lake_dir,
        "raw_data_dir": raw_data_dir,
        "processed_data_dir": processed_data_dir,
        "models_dir": models_dir,
        "streaming_input_dir": streaming_input_dir,
    }


PROJECT_ROOT = get_runtime_paths()["project_root"]
DATA_LAKE_DIR = get_runtime_paths()["data_lake_dir"]
RAW_DATA_DIR = get_runtime_paths()["raw_data_dir"]
PROCESSED_DATA_DIR = get_runtime_paths()["processed_data_dir"]
MODELS_DIR = get_runtime_paths()["models_dir"]
STREAMING_INPUT_DIR = get_runtime_paths()["streaming_input_dir"]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
