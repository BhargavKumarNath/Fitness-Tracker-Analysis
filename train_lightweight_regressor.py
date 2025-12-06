import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Paths
DATA_PATH = "data_lake/processed/fitness_data"
MODELS_DIR = "dashboard/models"

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
os.makedirs(MODELS_DIR, exist_ok=True)

print("Training LIGHTWEIGHT Calorie Regressor...")
X_reg = df[['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']]
y_reg = df['calories_burned']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['steps', 'heart_rate_avg', 'sleep_hours']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['activity_type'])
])

# KEY: Reduced complexity for cloud deployment
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=20,      # Reduced from 100
        max_depth=10,         # Limited depth
        min_samples_split=10, # Simplified trees
        random_state=42,
        n_jobs=-1
    ))
])

print("Fitting model (this will take 2-3 minutes)...")
reg_pipeline.fit(X_reg, y_reg)

# Save with compression
output_path = os.path.join(MODELS_DIR, "calories_regressor.pkl")
joblib.dump(reg_pipeline, output_path, compress=3)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\n✅ SUCCESS!")
print(f"Model saved to: {output_path}")
print(f"Model size: {file_size_mb:.1f} MB")
print(f"\nTarget: <500MB for Streamlit Cloud")

if file_size_mb > 500:
    print("⚠️ Still too large! Reduce n_estimators further.")
else:
    print("✓ This model should work on Streamlit Cloud!")