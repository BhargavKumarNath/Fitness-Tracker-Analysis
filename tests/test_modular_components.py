import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.etl.extract import load_raw_data
from src.etl.load import write_processed_data
from src.models.training import MODEL_TREES, train_dashboard_models
from dashboard.utils import (
    get_activity_categories,
    get_model_path,
    get_user_segments,
    load_model_metrics,
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
        "metrics.json",
    }
    assert all(path.exists() for path in artifacts)


def test_metrics_are_null_rather_than_misleading_on_too_little_data(tmp_path):
    # 2 rows cannot support an honest 80/20 held-out split. The training
    # function must say so (null), not report a number computed from noise.
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

    train_dashboard_models(frame, tmp_path / "models")
    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())

    assert metrics["activity_classifier"]["held_out_accuracy"] is None
    assert metrics["calories_regressor"]["held_out_r2"] is None
    assert metrics["calories_regressor"]["held_out_rmse"] is None


def test_metrics_report_real_held_out_performance_on_enough_data(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    activities = rng.choice(["walking", "running", "cycling", "yoga"], size=n)
    steps = rng.integers(100, 20000, size=n)
    frame = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "steps": steps,
            "calories_burned": steps * 0.05 + rng.normal(0, 20, size=n),
            "heart_rate_avg": rng.integers(60, 180, size=n),
            "sleep_hours": rng.uniform(4, 10, size=n),
            "activity_type": activities,
        }
    )

    train_dashboard_models(frame, tmp_path / "models")
    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())

    accuracy = metrics["activity_classifier"]["held_out_accuracy"]
    r2 = metrics["calories_regressor"]["held_out_r2"]
    assert accuracy is not None and 0.0 <= accuracy <= 1.0
    assert r2 is not None and r2 <= 1.0


def test_training_uses_bounded_parallel_forest_configuration():
    assert MODEL_TREES == 50


def test_live_inference_activity_options_match_training_categories(monkeypatch, tmp_path):
    # The Live Inference dropdown and the regressor's OneHotEncoder must be
    # driven from the same source. Regression test for a real bug: the
    # dropdown previously offered capitalized labels ("Walking", "HIIT")
    # that never matched the model's real, lowercase training categories,
    # silently zeroing the categorical feature for every option.
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))
    processed_dir = tmp_path / "data_lake" / "processed" / "fitness_data"
    processed_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {"activity_type": "hiking", "steps": 1},
            {"activity_type": "yoga", "steps": 2},
            {"activity_type": "walking", "steps": 3},
        ]
    )
    frame.to_parquet(processed_dir / "data.parquet", index=False)

    from dashboard.utils import load_dataset

    load_dataset.clear()
    get_activity_categories.clear()
    categories = get_activity_categories()

    assert categories == ["hiking", "walking", "yoga"]
    assert all(c == c.lower() for c in categories)


def test_load_model_metrics_returns_none_before_training_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))
    load_model_metrics.clear()

    assert load_model_metrics() is None


def test_load_model_metrics_reads_what_training_wrote(monkeypatch, tmp_path):
    monkeypatch.setenv("FITNESS_TRACKER_ROOT", str(tmp_path))
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
    train_dashboard_models(frame, tmp_path / "dashboard" / "models")
    load_model_metrics.clear()

    metrics = load_model_metrics()

    assert metrics is not None
    assert "held_out_accuracy" in metrics["activity_classifier"]
    assert "held_out_r2" in metrics["calories_regressor"]


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
    assert predict_calories_baseline(0, 68, 8.0, "yoga") >= 50


def test_calorie_baseline_differentiates_real_activity_categories():
    # Regression test: activity_factor used to be keyed on capitalized
    # labels ("Walking", "HIIT") that never matched the real, lowercase
    # activity_type values, so every activity silently fell through to the
    # same default factor. hiking (real high-calorie activity) must now
    # score meaningfully higher than yoga (real low-calorie activity) for
    # identical steps/heart-rate/sleep inputs.
    yoga = predict_calories_baseline(8000, 130, 7.5, "yoga")
    hiking = predict_calories_baseline(8000, 130, 7.5, "hiking")
    unknown = predict_calories_baseline(8000, 130, 7.5, "Yoga")

    assert hiking > yoga
    assert unknown != yoga


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
