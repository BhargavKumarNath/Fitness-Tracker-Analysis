import streamlit as st
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.utils import load_inference_models

# Set page config
st.set_page_config(page_title="Live Predictions", page_icon="🔮", layout="wide")

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("dashboard/style.css")
except:
    pass

class_model, reg_model = load_inference_models()

st.title("🔮 Live Machine Learning Predictions")

if class_model is None or reg_model is None:
    st.error("⚠️ Models not found. Please run 'src/train_dashboard_models.py' locally to generate models.")
else:
    # Use tabs for different prediction tasks
    tab1, tab2 = st.tabs(["🏃 Activity Prediction", "🔥 Calories Prediction"])

    with tab1:
        st.markdown("### Predict Activity Type")
        st.markdown("Enter your metrics to predict the type of activity you performed.")
        
        col1, col2, col3 = st.columns(3)
        steps = col1.slider("Steps", 0, 25000, 5000, key="class_steps")
        calories = col2.slider("Calories Burned", 50, 2000, 400, key="class_cal")
        hr = col3.slider("Average Heart Rate", 60, 180, 120, key="class_hr")
        
        if st.button("Predict Activity", type="primary"):
            # Prepare input dataframe
            input_data = pd.DataFrame([[steps, calories, hr]], columns=['steps', 'calories_burned', 'heart_rate_avg'])
            
            prediction = class_model.predict(input_data)[0]
            
            st.markdown("---")
            st.success(f"### Predicted Activity: **{prediction}**")

    with tab2:
        st.markdown("### Predict Calories Burned")
        st.markdown("Estimate calories burned based on your activity and stats.")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        activity_options = ['Walking', 'Running', 'Cycling', 'Yoga', 'HIIT', 'Strength Training'] 
        
        activity_reg = col_r1.selectbox("Activity Type", activity_options, key="reg_act")
        steps_reg = col_r2.slider("Steps", 0, 25000, 8000, key="reg_steps")
        hr_reg = col_r3.slider("Average Heart Rate", 60, 180, 130, key="reg_hr")
        sleep_reg = st.slider("Sleep Hours (Previous Night)", 4.0, 12.0, 7.5, key="reg_sleep")

        if st.button("Predict Calories", type="primary"):
            # Columns: ['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type']
            input_reg_data = pd.DataFrame([[steps_reg, hr_reg, sleep_reg, activity_reg]], 
                                          columns=['steps', 'heart_rate_avg', 'sleep_hours', 'activity_type'])
            
            cal_prediction = reg_model.predict(input_reg_data)[0]
            
            st.markdown("---")
            st.success(f"### Predicted Calories Burned: **{cal_prediction:,.0f} kcal**")
