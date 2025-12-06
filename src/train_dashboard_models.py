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

# Define paths
DATA_PATH = "data_lake/processed/fitness_data"
MODELS_DIR = "dashboard/models"

def train_and_save_models():
    """Trains dashboard models and saves them as .pkl files."""
    
    print("Loading data...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data not found at {DATA_PATH}")
        return

    # Create models directory if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Data loaded. Preprocessing...")
    
    # 1. Classification Model (Activity Prediction)
    print("Training Classification Model (Activity Type)...")
    
    # Features for classification
    class_features = ['steps', 'calories_burned', 'heart_rate_avg']
    target_class = 'activity_type'
    
    X_class = df[class_features]
    y_class = df[target_class]
    
    # Simple pipeline
    class_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    class_pipeline.fit(X_class, y_class)
    joblib.dump(class_pipeline, os.path.join(MODELS_DIR, "activity_classifier.pkl"))
    print("Classification model saved.")

    # 2. Regression Model (Calories Prediction)
    print("Training Regression Model (Calories Burned)...")
        
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    reg_features = ['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']
    target_reg = 'calories_burned'
    
    X_reg = df[reg_features]
    y_reg = df[target_reg]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['steps', 'heart_rate_avg', 'sleep_hours']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['activity_type'])
        ])
    
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    reg_pipeline.fit(X_reg, y_reg)
    joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "calories_regressor.pkl"))
    print("Regression model saved.")


    # 3. Clustering Model (User Segments)
    print("Training Clustering Model (User Segments)...")
    # For clustering, we aggregate by user first
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
        ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    
    cluster_pipeline.fit(user_df[cluster_features])
    joblib.dump(cluster_pipeline, os.path.join(MODELS_DIR, "user_segmentation.pkl"))
    
    # Also save the feature names validation
    joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))
    
    print("Clustering model saved.")
    print("All models successfully trained and saved!")

if __name__ == "__main__":
    train_and_save_models()
