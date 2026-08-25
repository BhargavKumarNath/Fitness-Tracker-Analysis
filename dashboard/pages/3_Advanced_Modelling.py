import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.utils import (
    get_classifier_model,
    get_regressor_model,
    get_user_segments,
    load_dataset,
    predict_activity_baseline,
    predict_calories_baseline,
)

st.set_page_config(page_title="Advanced Modeling", page_icon="🧠", layout="wide")


def load_css(file_name):
    try:
        with open(file_name) as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css("dashboard/style.css")
st.title("🧠 Advanced Modeling")
st.markdown("Explore behavior segments and test predictions without waiting for large model downloads.")

df = load_dataset()
if df.empty:
    st.error("No processed data is available. Run the pipeline before opening this page.")
    st.stop()

df_sample = df.sample(n=min(len(df), 2000), random_state=42)
tab_segments, tab_classification, tab_regression, tab_comparison = st.tabs(
    ["🧬 User Segmentation", "🏃 Activity Classification", "🔥 Calorie Regression", "📊 Model Comparison"]
)

with tab_segments:
    st.header("User Segmentation")
    st.caption("Users are grouped by average steps, calories burned, and heart rate.")
    user_segments = get_user_segments(df_sample)
    if user_segments.empty:
        st.info("There are not enough user records to create segments.")
    else:
        left, right = st.columns(2)
        with left:
            st.metric("Users analyzed", f"{len(user_segments):,}")
        with right:
            st.metric("Groups created", user_segments["prediction"].nunique())

        fig = px.scatter_3d(
            user_segments,
            x="avg_steps",
            y="avg_calories",
            z="avg_hr",
            color="prediction",
            hover_data=["user_id"],
            title="User groups by average behavior",
        )
        st.plotly_chart(fig, use_container_width=True)
        summary = user_segments.groupby("prediction").agg(
            Users=("user_id", "count"),
            Avg_Steps=("avg_steps", "mean"),
            Avg_Calories=("avg_calories", "mean"),
            Avg_Heart_Rate=("avg_hr", "mean"),
        ).round(1)
        st.dataframe(summary, use_container_width=True)

with tab_classification:
    st.header("Activity Classification")
    st.caption("Evaluate the local classifier when available, or use the instant baseline for a working demo.")
    run_classification = st.button("Run classification evaluation", key="run_classification")
    if run_classification:
        class_model = get_classifier_model()
        features = df_sample[["steps", "calories_burned", "heart_rate_avg"]]
        actual = df_sample["activity_type"].astype(str).str.strip().str.lower()
        if class_model is not None:
            predicted = class_model.predict(features)
            source = "trained local classifier"
        else:
            predicted = [predict_activity_baseline(row.steps, row.heart_rate_avg) for row in features.itertuples()]
            source = "instant baseline"
        predicted = np.asarray(predicted).astype(str).astype(str)
        predicted = np.char.lower(np.char.strip(predicted))
        accuracy = (predicted == actual.to_numpy()).mean()
        st.success(f"Evaluation completed with the {source}.")
        st.metric("Accuracy on sample", f"{accuracy:.1%}")
        labels = sorted(set(actual.unique()) | set(predicted))
        matrix = confusion_matrix(actual, predicted, labels=labels)
        st.plotly_chart(
            px.imshow(matrix, x=labels, y=labels, text_auto=True, labels={"x": "Predicted", "y": "Actual"}, title="Confusion matrix"),
            use_container_width=True,
        )
        report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

with tab_regression:
    st.header("Calorie Regression")
    st.caption("Evaluate the local regressor when available, or use the instant baseline for a working demo.")
    run_regression = st.button("Run regression evaluation", key="run_regression")
    if run_regression:
        reg_model = get_regressor_model()
        features = df_sample[["steps", "heart_rate_avg", "sleep_hours", "activity_type"]]
        actual = df_sample["calories_burned"]
        if reg_model is not None:
            predicted = reg_model.predict(features)
            source = "trained local regressor"
        else:
            predicted = [
                predict_calories_baseline(row.steps, row.heart_rate_avg, row.sleep_hours, row.activity_type)
                for row in features.itertuples()
            ]
            source = "instant baseline"
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        st.success(f"Evaluation completed with the {source}.")
        first, second = st.columns(2)
        first.metric("RMSE on sample", f"{rmse:,.1f}")
        second.metric("R² on sample", f"{r2_score(actual, predicted):.3f}")
        chart = px.scatter(x=actual, y=predicted, labels={"x": "Actual calories", "y": "Predicted calories"}, title="Actual vs predicted")
        st.plotly_chart(chart, use_container_width=True)

with tab_comparison:
    st.header("Model Availability")
    st.caption("Model artifacts are optional runtime files. The dashboard remains usable without downloading them.")
    availability = pd.DataFrame([
        {"Capability": "User segmentation", "Default behavior": "Local activity bands", "Requires model": "No"},
        {"Capability": "Activity classification", "Default behavior": "Instant baseline", "Requires model": "No"},
        {"Capability": "Calorie prediction", "Default behavior": "Instant baseline", "Requires model": "No"},
    ])
    st.dataframe(availability, hide_index=True, use_container_width=True)
    st.info("For trained-model evaluation, place compatible artifacts in dashboard/models/ and rerun the selected evaluation.")
