from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_TREES = 50


def train_dashboard_models(df: pd.DataFrame, models_dir: Path | str) -> list[Path]:
    """Train and persist the classifier, regressor, and segmentation models."""
    destination = Path(models_dir)
    destination.mkdir(parents=True, exist_ok=True)

    class_features = ["steps", "calories_burned", "heart_rate_avg"]
    class_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=MODEL_TREES,
                max_depth=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    class_pipeline.fit(df[class_features], df["activity_type"])

    reg_features = ["steps", "heart_rate_avg", "sleep_hours", "activity_type"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["steps", "heart_rate_avg", "sleep_hours"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["activity_type"]),
        ]
    )
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=MODEL_TREES,
                max_depth=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    reg_pipeline.fit(df[reg_features], df["calories_burned"])

    user_df = df.groupby("user_id").agg({
        "steps": "mean",
        "calories_burned": "mean",
        "heart_rate_avg": "mean",
    }).rename(columns={
        "steps": "avg_steps",
        "calories_burned": "avg_calories",
        "heart_rate_avg": "avg_hr",
    })
    if user_df.empty:
        raise ValueError("No user profiles available for clustering.")

    cluster_features = ["avg_steps", "avg_calories", "avg_hr"]
    cluster_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=min(5, len(user_df)), random_state=42, n_init=20)),
    ])
    cluster_pipeline.fit(user_df[cluster_features])

    artifacts = {
        "activity_classifier.pkl": class_pipeline,
        "calories_regressor.pkl": reg_pipeline,
        "user_segmentation.pkl": cluster_pipeline,
        "cluster_features.pkl": cluster_features,
    }
    artifact_paths = []
    for filename, artifact in artifacts.items():
        path = destination / filename
        joblib.dump(artifact, path)
        artifact_paths.append(path)
    return artifact_paths