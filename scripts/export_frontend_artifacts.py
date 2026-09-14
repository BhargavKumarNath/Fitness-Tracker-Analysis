"""Export deterministic, static artifacts for the frontend application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictions import predict_activity_baseline, predict_calories_baseline
from src.config import get_runtime_paths

NUMERIC_COLUMNS = ["steps", "calories_burned", "heart_rate_avg", "sleep_hours"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _histogram(series: pd.Series, bins: int) -> list[dict[str, float | int]]:
    counts, edges = np.histogram(series.dropna().to_numpy(), bins=bins)
    return [
        {"start": round(float(edges[index]), 4), "end": round(float(edges[index + 1]), 4), "count": int(count)}
        for index, count in enumerate(counts)
    ]


def _date_string(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _summary_stats(frame: pd.DataFrame) -> list[dict[str, object]]:
    result = []
    for column in NUMERIC_COLUMNS:
        series = frame[column]
        result.append({
            "metric": column,
            "mean": _json_value(series.mean()),
            "median": _json_value(series.median()),
            "std": _json_value(series.std()),
            "min": _json_value(series.min()),
            "max": _json_value(series.max()),
            "q1": _json_value(series.quantile(0.25)),
            "q3": _json_value(series.quantile(0.75)),
            "iqr": _json_value(series.quantile(0.75) - series.quantile(0.25)),
            "skewness": _json_value(series.skew()),
            "kurtosis": _json_value(series.kurtosis()),
        })
    return result


def _activity_analysis(frame: pd.DataFrame) -> dict[str, object]:
    summary = []
    for activity, group in frame.groupby("activity_type", sort=True):
        def metric(column: str) -> dict[str, object]:
            return {"mean": _json_value(group[column].mean()), "median": _json_value(group[column].median()), "std": _json_value(group[column].std())}

        summary.append({
            "activityType": activity,
            "count": int(len(group)),
            "steps": metric("steps"),
            "calories": metric("calories_burned"),
            "heartRate": metric("heart_rate_avg"),
            "sleepHours": metric("sleep_hours"),
        })
    sample = frame.sample(n=min(len(frame), 500), random_state=42)
    return {
        "categories": sorted(frame["activity_type"].dropna().unique().tolist()),
        "summary": summary,
        "stepsCaloriesSample": [{"steps": _json_value(row.steps), "calories": _json_value(row.calories_burned), "activityType": row.activity_type} for row in sample.itertuples()],
        "heartRateCaloriesSample": [{"heartRate": _json_value(row.heart_rate_avg), "calories": _json_value(row.calories_burned), "activityType": row.activity_type} for row in sample.itertuples()],
        "histograms": {column: _histogram(frame[column], 50) for column in NUMERIC_COLUMNS},
    }


def _segmentation(frame: pd.DataFrame, models_dir: Path) -> dict[str, object]:
    users = frame.groupby("user_id", as_index=False).agg(
        avgSteps=("steps", "mean"), avgCalories=("calories_burned", "mean"), avgHeartRate=("heart_rate_avg", "mean")
    )
    users = users.rename(columns={"user_id": "userId"})
    method = "activity-band-fallback"
    model_path = models_dir / "user_segmentation.pkl"
    features_path = models_dir / "cluster_features.pkl"
    if model_path.exists() and features_path.exists():
        try:
            import joblib

            pipeline = joblib.load(model_path)
            features = joblib.load(features_path)
            users["cluster"] = pipeline.predict(users[features]).astype(int)
            method = "kmeans"
        except Exception:
            method = "activity-band-fallback"
    if method == "activity-band-fallback":
        cluster_count = min(5, len(users))
        ranks = users["avgSteps"].rank(method="first")
        users["cluster"] = (((ranks - 1) * cluster_count / len(users)).astype(int)).clip(upper=cluster_count - 1)

    clusters = []
    for cluster, group in users.groupby("cluster", sort=True):
        avg_steps = float(group["avgSteps"].mean())
        if avg_steps < 5000:
            label, message = "Sedentary", "Focus on basic engagement and motivation."
        elif avg_steps < 8000:
            label, message = "Lightly active", "Encourage consistency and gradual increases."
        elif avg_steps < 10000:
            label, message = "Active", "Meeting a strong everyday movement baseline."
        else:
            label, message = "High performer", "Provide advanced challenges and goals."
        clusters.append({"cluster": int(cluster), "users": int(len(group)), "avgSteps": avg_steps, "avgCalories": float(group.avgCalories.mean()), "avgHeartRate": float(group.avgHeartRate.mean()), "label": label, "message": message})
    users_payload = [{"userId": _json_value(row.userId), "avgSteps": float(row.avgSteps), "avgCalories": float(row.avgCalories), "avgHeartRate": float(row.avgHeartRate), "cluster": int(row.cluster)} for row in users.itertuples()]
    return {"method": method, "features": ["avgSteps", "avgCalories", "avgHeartRate"], "users": users_payload, "clusters": clusters}


def _data_quality(frame: pd.DataFrame) -> dict[str, object]:
    completeness = []
    for column in frame.columns:
        series = frame[column]
        completeness.append({"column": column, "missing": int(series.isna().sum()), "completeness": float((1 - series.isna().mean()) * 100), "dtype": str(series.dtype), "uniqueValues": int(series.nunique()), "sampleValue": str(series.iloc[0])[:50]})
    outliers = []
    for column in NUMERIC_COLUMNS:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        outliers.append({"metric": column, "count": count, "percentage": count / len(frame) * 100, "lowerBound": float(lower), "upperBound": float(upper)})
    validity_ranges = {"steps": (0, 50000), "calories_burned": (0, 5000), "heart_rate_avg": (40, 220), "sleep_hours": (0, 24)}
    validity = []
    for column, (minimum, maximum) in validity_ranges.items():
        count = int(((frame[column] < minimum) | (frame[column] > maximum)).sum())
        validity.append({"metric": column, "expectedMin": minimum, "expectedMax": maximum, "outOfRange": count, "validPercentage": (len(frame) - count) / len(frame) * 100})
    efficiency = frame.groupby("activity_type").agg(totalCalories=("calories_burned", "sum"), totalSteps=("steps", "sum"))
    return {
        "statistics": _summary_stats(frame),
        "completeness": completeness,
        "duplicates": {"count": int(frame.duplicated().sum()), "percentage": float(frame.duplicated().mean() * 100), "uniqueRecords": int(len(frame) - frame.duplicated().sum())},
        "validity": validity,
        "outliers": outliers,
        "findings": {
            "mostPopularActivity": frame["activity_type"].value_counts().index[0],
            "averageSteps": float(frame["steps"].mean()), "highPerformerPercentage": float((frame["steps"] > 10000).mean() * 100),
            "averageHeartRate": float(frame["heart_rate_avg"].mean()), "averageSleepHours": float(frame["sleep_hours"].mean()),
            "heartRateMin": float(frame["heart_rate_avg"].min()), "heartRateMax": float(frame["heart_rate_avg"].max()),
            "stepsCaloriesCorrelation": float(frame["steps"].corr(frame["calories_burned"])), "heartRateCaloriesCorrelation": float(frame["heart_rate_avg"].corr(frame["calories_burned"])),
            "mostEfficientActivity": (efficiency["totalCalories"] / efficiency["totalSteps"]).idxmax(),
        },
    }


def _evaluation(frame: pd.DataFrame) -> dict[str, object]:
    sample = frame.sample(n=min(len(frame), 2000), random_state=42)
    actual = sample["activity_type"].astype(str).str.strip().str.lower().to_numpy()
    predicted = np.array([predict_activity_baseline(row.steps, row.heart_rate_avg) for row in sample.itertuples()])
    labels = sorted(set(actual) | set(predicted))
    regression_predicted = np.array([predict_calories_baseline(row.steps, row.heart_rate_avg, row.sleep_hours, row.activity_type) for row in sample.itertuples()])
    return {
        "sampleSize": int(len(sample)),
        "classification": {"accuracy": float((actual == predicted).mean()), "labels": labels, "confusionMatrix": confusion_matrix(actual, predicted, labels=labels).tolist(), "report": classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0), "source": "baseline"},
        "regression": {"rmse": float(np.sqrt(mean_squared_error(sample["calories_burned"], regression_predicted))), "r2": float(r2_score(sample["calories_burned"], regression_predicted)), "pairs": [{"actual": float(actual_value), "predicted": float(predicted_value)} for actual_value, predicted_value in zip(sample["calories_burned"].iloc[:500], regression_predicted[:500])], "source": "baseline"},
    }


def export_artifacts(project_root: Path | None = None, output_dir: Path | None = None) -> Path:
    paths = get_runtime_paths() if project_root is None else {"processed_data_dir": project_root / "data_lake" / "processed" / "fitness_data", "models_dir": project_root / "artifacts"}
    source = paths["processed_data_dir"]
    if not source.exists():
        raise FileNotFoundError(f"Processed data not found at {source}")
    destination = output_dir or (project_root or get_runtime_paths()["project_root"]) / "frontend" / "public" / "data"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"])
    overview = {"generatedAt": pd.Timestamp.now(tz="UTC").isoformat(), "rowCount": int(len(frame)), "userCount": int(frame["user_id"].nunique()), "activityCount": int(frame["activity_type"].nunique()), "averageSteps": float(frame["steps"].mean()), "averageCalories": float(frame["calories_burned"].mean()), "averageHeartRate": float(frame["heart_rate_avg"].mean()), "averageSleepHours": float(frame["sleep_hours"].mean()), "dateRange": {"start": _date_string(frame["date"].min()), "end": _date_string(frame["date"].max()), "days": int((frame["date"].max() - frame["date"].min()).days)}, "activities": [{"activityType": activity, "count": int(count), "share": float(count / len(frame))} for activity, count in frame["activity_type"].value_counts().sort_index().items()]}
    health = {"numericColumns": NUMERIC_COLUMNS, "correlation": frame[NUMERIC_COLUMNS].corr().round(8).values.tolist(), "heartRateByActivity": [{"activityType": activity, "average": float(value)} for activity, value in frame.groupby("activity_type")["heart_rate_avg"].mean().items()], "sleepByActivity": [{"activityType": activity, "average": float(value)} for activity, value in frame.groupby("activity_type")["sleep_hours"].mean().items()], "heartRateHistogram": _histogram(frame["heart_rate_avg"], 30), "sleepHistogram": _histogram(frame["sleep_hours"], 20)}
    daily = frame.groupby("date").agg(steps=("steps", "mean"), calories=("calories_burned", "mean"), heartRate=("heart_rate_avg", "mean")).reset_index()
    weekdays = frame.groupby("day_of_week").agg(steps=("steps", "mean"), calories=("calories_burned", "mean"), count=("user_id", "count")).reindex(DAY_ORDER).reset_index()
    temporal = {"daily": [{"date": _date_string(row.date), "steps": float(row.steps), "calories": float(row.calories), "heartRate": float(row.heartRate)} for row in daily.itertuples()], "weekdays": [{"day": row.day_of_week, "steps": float(row.steps), "calories": float(row.calories), "count": int(row.count)} for row in weekdays.itertuples()]}
    metrics_path = paths["models_dir"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    model_metrics = {"activityClassifier": {"heldOutAccuracy": metrics.get("activity_classifier", {}).get("held_out_accuracy")}, "caloriesRegressor": {"heldOutR2": metrics.get("calories_regressor", {}).get("held_out_r2"), "heldOutRmse": metrics.get("calories_regressor", {}).get("held_out_rmse")}, "methodology": metrics.get("methodology")}
    artifacts = {"overview.json": overview, "activity_analysis.json": _activity_analysis(frame), "health_metrics.json": health, "temporal.json": temporal, "segmentation.json": _segmentation(frame, paths["models_dir"]), "data_quality.json": _data_quality(frame), "model_metrics.json": model_metrics, "model_eval_sample.json": _evaluation(frame)}
    for name, value in artifacts.items():
        _write_json(destination / name, value)
    frame.to_parquet(destination / "fitness.parquet", index=False, compression="zstd")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = export_artifacts(args.project_root, args.output_dir)
    print(f"Exported frontend artifacts to {result}")