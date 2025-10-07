import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import sys
sys.path.append("/app")

st.set_page_config(page_title="Live Predictions", page_icon="🔮", layout="wide")

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("FitnessDashboard").master("local[*]").getOrCreate()

@st.cache_resource
def load_models():
    spark = get_spark_session()
    class_model = PipelineModel.load("/app/models/spark_classification_model")
    reg_model = PipelineModel.load("/app/models/spark_regression_model")
    return class_model, reg_model

st.title("🔮 Live Machine Learning Predictions")
spark = get_spark_session()
classification_model, regression_model = load_models()

# Prediction 1: Activity Type (Classification)
st.header("Predict Activity Type")
col1, col2, col3 = st.columns(3)
steps = col1.slider("Steps", 0, 25000, 5000)
calories = col2.slider("Calories Burned", 50, 2000, 400)
hr = col3.slider("Average Heart Rate", 60, 180, 120)

# Create a Spark DataFrame from the inputs
predict_data = [(steps, calories, hr, 7.0)] # Adding a default sleep_hours
predict_df = spark.createDataFrame(predict_data, ["steps", "calories_burned", "heart_rate_avg", "sleep_hours"])

# Make prediction
prediction = classification_model.transform(predict_df)
predicted_label_index = prediction.select("prediction").first()[0]
activity_labels = classification_model.stages[0].labels # Get labels from StringIndexer
predicted_activity = activity_labels[int(predicted_label_index)]

st.success(f"Predicted Activity: **{predicted_activity}**")


# Prediction 2: Calories Burned (Regression)
st.header("Predict Calories Burned")
col_reg1, col_reg2, col_reg3 = st.columns(3)
activity_reg = col_reg1.selectbox("Activity Type", activity_labels)
steps_reg = col_reg2.slider("Steps", 0, 25000, 8000, key="steps_reg")
hr_reg = col_reg3.slider("Average Heart Rate", 60, 180, 130, key="hr_reg")

# Create DataFrame for regression prediction
predict_reg_data = [(steps_reg, hr_reg, 7.5, activity_reg)]
predict_reg_df = spark.createDataFrame(predict_reg_data, ["steps", "heart_rate_avg", "sleep_hours", "activity_type"])

# Make prediction
prediction_reg = regression_model.transform(predict_reg_df)
predicted_calories = prediction_reg.select("prediction").first()[0]

st.success(f"Predicted Calories Burned: **{predicted_calories:,.0f}**")
