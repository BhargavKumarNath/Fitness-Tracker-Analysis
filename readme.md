# Fitness Analytics with PySpark

![alt text](https://github.com/BhargavKumarNath/Fitness-Tracker-Analysis/actions/workflows/ci.yml/badge.svg)
![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

# 1. Project Vision
This repository demonstrates a production grade, end-to-end data science project. The core philosophy is to build a scalable, reliable, and automated system for analyzing fitness tracker data using a modern data engineering and MLOps stack.

The project ingests and processes a substantial dataset of **358, 497 records from 1,959 users** and showcase a full project lifecycle: from foundational ETL and advanced ML modeling with **PySpark** to robust MLOps workflow using Docker for reproducibility and **GitHub Actions** for continuous integration and testing.

# 2. Technical Architecture
The system is designed as a modular pipeline, ensuring scalability and maintainability.

![Alt text](plot/workflow.png)

# 3. Phase I: The Data Foundation - ETL with PySpark

The realibility of the entire system depends on a robust data foundation. We choose **Apache Spark (PySpark)** as our processing engine for its ability to handle large datasets in a distributed manner and its unified API for data processing and machine learning.

**The ETL Pipeline** (`src/etl_pipeline.py`)
* **Extraction:** The pipeline reads over 180 daily-partitioned Parquet files. We chose Parquet for its columnar storage format, which provides significant performance gains and storage efficiency.

* **Transformation:** The core logic is encapsulated in a testable transform_data function. Key transformations include:
    * **Schema Enforcement:** Casting string dates to DateType for proper time-series operations.

    * **Feature Engineering:** Creating high-value features like `calories_to_steps_ratio`, which proved critical for distinguishing between low-step, high-intensity activities (e.g., swimming) and high-step, low-intensity ones (e.g., walking).

* Loading: The transformed data is written to `data_lake/processed/`, partitioned by `year` and `month`. This partitioning scheme is a critical optimization, allowing Spark to perform partition pruning and drastically speed up queries that filter on specific time ranges.

# 4. Phase II: Advanced Analytics & Machine Learning
With a clean dataset, we used PySpark MLlib and Spark SQL to build predictive models and derive insights.

## Machine Learning Models

1. **User Segmentation (K-Means Clustering):** An unsupervised model identified three statistically distinct user personas, enabling targeted analysis and engagement strategies.

2. **Activity Prediction (Logistic Regression):** A multi-class classification pipeline was trained to predict activity_type. The model achieved a strong **84% accuracy** on the test set.

3. **Calorie Prediction (Linear Regression):** A regression pipeline, featuring a `OneHotEncoder` to handle categorical features, was built to predict `calories_burned`. The model was highly successful, achieving an **R² of 0.91**, indicating that our features explain 91% of the variance in calorie expenditure.

4. **Activity Recommendation (ALS):** A collaborative filtering model was trained to provide personalized activity recommendations, using the frequency of activities as an implicit user rating.

# 5. Phase III: Real-Time Processing with Spark Structured Streaming

To demonstrate a more advanced, production-ready capability, we implemented a real-time analytics pipeline.

* **Architecture:** A data generator script simulates a live stream of user activity.

* **Stateful Aggregation:** We used Spark Structured Streaming to consume this data, treating the stream as an unbounded table. The core of the analysis is a stateful aggregation using **30-second tumbling windows**.

* **Watermarking:** To ensure the long-running stream is resilient and does not run out of memory, a **1-minute watermark** is applied. This allows Spark to correctly handle late-arriving data and discard old state that is no longer needed. This is a critical feature for production streaming jobs.
