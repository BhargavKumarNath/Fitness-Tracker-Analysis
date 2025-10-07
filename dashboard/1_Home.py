import streamlit as st

st.set_page_config(page_title="Fitness Tracker - Home", page_icon="🏃", layout="wide")

st.title("Welcome to the Fitness Tracker Data Project")
st.sidebar.success("Select a page above to explore")

st.markdown("""
    ### This is an end-to-end demonstration of a modern data science project.
    
    This multi-page application showcases the entire pipeline, from data ingestion and ETL to exploratory data analysis, machine learning, and finally, this interactive dashboard.
    
    **👈 Select a page from the sidebar** to dive into the different aspects of the project.
    
    ### Project Phases Showcased:
    - **Data Overview:** Explore the structure and a sample of our processed dataset.
    - **EDA and Clustering:** Discover insights from the data through visualizations and unsupervised learning.
    - **Live Predictions:** Interact directly with our trained machine learning models to make real-time predictions.
""")