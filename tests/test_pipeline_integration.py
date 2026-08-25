from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.run_pipeline import run_pipeline


def test_run_pipeline_generates_processed_data_and_models(tmp_path, monkeypatch):
    root = tmp_path / "fitness_project"
    raw_dir = root / "data_lake" / "raw" / "synthetic_user_data" / "year=2023" / "month=04" / "day=01"
    raw_dir.mkdir(parents=True)

    frame = pd.DataFrame([
        {
            "user_id": 1,
            "date": "2023-04-01",
            "steps": 1000,
            "calories_burned": 250.0,
            "heart_rate_avg": 120,
            "sleep_hours": 7.5,
            "activity_type": "walking",
        },
        {
            "user_id": 2,
            "date": "2023-04-02",
            "steps": 0,
            "calories_burned": 80.0,
            "heart_rate_avg": 68,
            "sleep_hours": 8.0,
            "activity_type": "yoga",
        },
    ])
    frame.to_parquet(raw_dir / "data.parquet", index=False)

    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(root))

    result = run_pipeline(project_root=root)

    assert result["processed_data_dir"].exists()
    assert result["models_dir"].exists()
    assert (result["models_dir"] / "activity_classifier.pkl").exists()
    assert (result["models_dir"] / "calories_regressor.pkl").exists()
    assert (result["models_dir"] / "user_segmentation.pkl").exists()
    assert (result["models_dir"] / "cluster_features.pkl").exists()

    processed_df = pd.read_parquet(result["processed_data_dir"])
    assert "day_of_week" in processed_df.columns
    assert "calories_to_steps_ratio" in processed_df.columns
