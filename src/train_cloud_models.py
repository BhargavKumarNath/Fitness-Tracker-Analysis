import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define paths
DATA_PATH = "data_lake/processed/fitness_data"
MODELS_DIR = "dashboard/models"

def train_cloud_models():
    """Trains lightweight dashboard models optimized for Streamlit Cloud (Free Tier)."""
    
    print("Loading data...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data not found at {DATA_PATH}")
        return

    # Create models directory if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Data loaded: {len(df)} records. Starting optimized training...")
    
    # --- 1. CLASSIFICATION MODEL ---
    print("\n[1/3] Training Lightweight Activity Classifier...")
    
    class_features = ['steps', 'calories_burned', 'heart_rate_avg']
    target_class = 'activity_type'
    
    X_class = df[class_features]
    y_class = df[target_class]
    
    # Optimized Classification Pipeline
    class_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # Lightweight Random Forest: restricted depth and estimators
        ('classifier', RandomForestClassifier(
            n_estimators=30,      # Reduced from 100
            max_depth=12,         # Prevent deep trees
            min_samples_leaf=10,  # Prune noise
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    class_pipeline.fit(X_class, y_class)
    
    class_path = os.path.join(MODELS_DIR, "activity_classifier.pkl")
    joblib.dump(class_pipeline, class_path, compress=3) # High compression
    
    size_mb = os.path.getsize(class_path) / (1024 * 1024)
    print(f"✅ Classifier saved: {size_mb:.1f} MB (Target: <50MB)")

    # --- 2. REGRESSION MODEL ---
    print("\n[2/3] Training Lightweight Calorie Regressor...")
        
    reg_features = ['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']
    target_reg = 'calories_burned'
    
    X_reg = df[reg_features]
    y_reg = df[target_reg]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['steps', 'heart_rate_avg', 'sleep_hours']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['activity_type'])
        ])
    
    # Optimized Regression Pipeline
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        # Lightweight Random Forest
        ('regressor', RandomForestRegressor(
            n_estimators=30,      # Reduced from 100
            max_depth=12,         # Prevent deep trees
            min_samples_leaf=10,  # Prune noise
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    reg_pipeline.fit(X_reg, y_reg)
    
    reg_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")
    joblib.dump(reg_pipeline, reg_path, compress=3)
    
    size_mb = os.path.getsize(reg_path) / (1024 * 1024)
    print(f"✅ Regressor saved: {size_mb:.1f} MB (Target: <50MB)")

    # --- 3. CLUSTERING MODEL (already small, but retraining for consistency) ---
    print("\n[3/3] Training Clustering Model...")
    
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
    
    cluster_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=5, random_state=42, n_init=20))
    ])
    
    cluster_pipeline.fit(user_df[cluster_features])
    
    # Save model and features
    joblib.dump(cluster_pipeline, os.path.join(MODELS_DIR, "user_segmentation.pkl"))
    joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))
    
    print("✅ User segmentation model saved.")
    print("\n🎉 All models optimized for Cloud deployment!")

if __name__ == "__main__":
    train_cloud_models()
