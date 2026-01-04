import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define paths
DATA_PATH = "data_lake/processed/fitness_data"
MODELS_DIR = "dashboard/models"

def train_segmentation_only():
    """Trains only the user segmentation model."""
    
    print("Loading data for segmentation...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data not found at {DATA_PATH}")
        return

    # Create models directory if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Data loaded. Preparing user profiles...")
    
    # Aggregating by user
    user_df = df.groupby('user_id').agg({
        'steps': 'mean',
        'calories_burned': 'mean',
        'heart_rate_avg': 'mean'
    }).rename(columns={
        'steps': 'avg_steps', 
        'calories_burned': 'avg_calories', 
        'heart_rate_avg': 'avg_hr'
    })
    
    cluster_features = ['avg_steps', 'avg_calories', 'avg_hr']
    
    print(f"Training KMeans on {len(user_df)} users...")
    
    # Use SimpleImputer strictly to fix the version mismatch issue
    cluster_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=5, random_state=42, n_init=20))
    ])
    
    cluster_pipeline.fit(user_df[cluster_features])
    joblib.dump(cluster_pipeline, os.path.join(MODELS_DIR, "user_segmentation.pkl"))
    
    # Also save the feature names validation
    joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))
    
    print("Clustering model saved successfully (user_segmentation.pkl).")

if __name__ == "__main__":
    train_segmentation_only()
