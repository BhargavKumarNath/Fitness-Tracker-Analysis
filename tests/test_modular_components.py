from pathlib import Path

import pandas as pd
import pytest

from src.etl.extract import load_raw_data
from src.etl.load import write_processed_data
from src.models.training import MODEL_TREES, train_dashboard_models
from dashboard.utils import (
    get_model_path,
    get_user_segments,
    load_user_segmentation_model,
    predict_activity_baseline,
    predict_calories_baseline,
)


def test_load_raw_data_reads_nested_parquet_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    expected = pd.DataFrame(
        [{"user_id": 1, "date": "2023-04-01", "steps": 100}]
    )
    expected.to_parquet(raw_dir / "data.parquet", index=False)

    result = load_raw_data(raw_dir)

    pd.testing.assert_frame_equal(result, expected)


def test_load_raw_data_ignores_non_parquet_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    expected = pd.DataFrame([{"user_id": 1, "steps": 100}])
    expected.to_parquet(raw_dir / "data.parquet", index=False)
    (raw_dir / "data.csv").write_text("not parquet\n")

    result = load_raw_data(raw_dir)

    pd.testing.assert_frame_equal(result, expected)


def test_load_raw_data_reports_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="No parquet files found"):
        load_raw_data(tmp_path / "missing")


def test_write_processed_data_creates_partitioned_output(tmp_path):
    frame = pd.DataFrame(
        [
            {"user_id": 1, "year": 2023, "month": 4, "steps": 100},
            {"user_id": 2, "year": 2023, "month": 5, "steps": 200},
        ]
    )

    output_dir = write_processed_data(frame, tmp_path / "processed")

    assert output_dir.exists()
    assert list(output_dir.rglob("*.parquet"))
    result = pd.read_parquet(output_dir)
    assert sorted(result["steps"].tolist()) == [100, 200]


def test_write_processed_data_requires_partition_columns(tmp_path):
    with pytest.raises(KeyError):
        write_processed_data(pd.DataFrame([{"steps": 100}]), tmp_path / "processed")


def test_train_dashboard_models_writes_all_artifacts(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "user_id": 1,
                "steps": 1000,
                "calories_burned": 250.0,
                "heart_rate_avg": 120,
                "sleep_hours": 7.5,
                "activity_type": "walking",
            },
            {
                "user_id": 2,
                "steps": 0,
                "calories_burned": 80.0,
                "heart_rate_avg": 68,
                "sleep_hours": 8.0,
                "activity_type": "yoga",
            },
        ]
    )

    artifacts = train_dashboard_models(frame, tmp_path / "models")

    assert {path.name for path in artifacts} == {
        "activity_classifier.pkl",
        "calories_regressor.pkl",
        "user_segmentation.pkl",
        "cluster_features.pkl",
    }
    assert all(path.exists() for path in artifacts)


def test_training_uses_bounded_parallel_forest_configuration():
    assert MODEL_TREES == 50


def test_dashboard_model_path_uses_current_runtime_root(monkeypatch, tmp_path):
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))

    assert get_model_path("activity_classifier") == (
        tmp_path / "dashboard" / "models" / "activity_classifier.pkl"
    )


def test_dashboard_model_path_rejects_traversal_or_nested_names():
    with pytest.raises(ValueError):
        get_model_path("../activity_classifier")


def test_baseline_predictions_are_immediate_and_bounded():
    assert predict_activity_baseline(0, 68) == "yoga"
    assert predict_activity_baseline(16000, 120) == "running"
    assert predict_calories_baseline(0, 68, 8.0, "Yoga") >= 50


def test_user_segmentation_has_local_fallback_without_model_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))
    load_user_segmentation_model.clear()
    frame = pd.DataFrame([
        {"user_id": 1, "steps": 1000, "calories_burned": 100.0, "heart_rate_avg": 70},
        {"user_id": 2, "steps": 10000, "calories_burned": 300.0, "heart_rate_avg": 130},
    ])

    result = get_user_segments(frame)

    assert result["prediction"].notna().all()
    assert len(result) == 2


def test_missing_segmentation_artifacts_return_none_without_deserialization(monkeypatch, tmp_path):
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))
    load_user_segmentation_model.clear()

    pipeline, features = load_user_segmentation_model()

    assert pipeline is None
    assert features is None
