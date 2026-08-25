from pathlib import Path

from src.config import get_runtime_paths


def test_runtime_paths_default_to_repo_layout():
    paths = get_runtime_paths()

    assert isinstance(paths["project_root"], Path)
    assert paths["project_root"].name == "Fitness-Tracker-Analysis"
    assert paths["processed_data_dir"].name == "fitness_data"
    assert paths["models_dir"].name == "models"


def test_runtime_paths_support_env_override(monkeypatch, tmp_path):
    custom_root = tmp_path / "custom_repo"
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(custom_root))

    paths = get_runtime_paths()

    assert paths["project_root"] == custom_root
    assert paths["processed_data_dir"] == custom_root / "data_lake" / "processed" / "fitness_data"
    assert paths["models_dir"] == custom_root / "dashboard" / "models"
