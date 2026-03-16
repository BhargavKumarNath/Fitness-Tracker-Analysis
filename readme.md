# Fitness Analytics Intelligence Platform

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)

## 1. Project Vision & Impact

This repository contains a **production grade Data Science and Machine Learning platform** built to process, analyze, and serve insights from high-volume fitness tracker data. 

Rather than a static analysis script, this project is architected as an end to end, resilient system designed to demonstrate how raw, disparate telemetry data is transformed into actionable intelligence using industry standard Data Engineering and MLOps practices.

**Key Outcomes Delivered:**
*   **Scalable Data Ingestion:** Distributed ETL pipelines utilizing PySpark for declarative transformations over partitioned Parquet data lakes.
*   **Decoupled Architecture:** Separation of concerns across data storage, feature engineering, model training, and interactive serving.
*   **Reproducible Intelligence:** Containerized execution environments, automated testing, and CI/CD pipelines to prevent "works on my machine" anti-patterns.
*   **Actionable ML Serving:** A fully decoupled interactive consumption layer (Streamlit) integrating pre-trained models for zero-latency inference.

<br>

## [Launch Live Intelligence Dashboard](https://fitness-tracker-analysis.streamlit.app/)

<br>

## 2. System Design & Architecture

The system is defined by a monolithic Batch Processing pipeline coupled with a decoupled ML inference layer. Below is the **Data Lifecycle**, illustrating how raw telemetry is curated and served:

### 2.1 The Data Lifecycle

*   **1. Ingestion Layer:** External telemetry from fitness trackers is batch-exported as raw, daily-partitioned Parquet files into the Data Lake.
*   **2. Transformation Engine:** A PySpark ETL pipeline performs data cleaning, schema enforcement, and feature engineering. It operates on a full-overwrite semantic and materializes the output as a partitioned analytical layer (acting as a lightweight Feature Store). 
*   **3. Inference & Serving Service:** The trained Scikit-Learn pipelines are serialized locally. The serving layer (Streamlit) maps user inputs to this local Model Registry, running real-time, in-memory inference without calling external microservices.
*   **4. Monitoring & Feedback Loop:** (Constraint) The current system lacks automated telemetry on model drift or data quality monitoring. Future iterations require an active feedback loop where Streamlit prediction logs are ingested back into the raw Data Lake.

### 2.2 Architecture Diagram

![Architecture Diagram](system_design.svg)

### 2.3 Technical Constraints & Engineering Trade-offs

To optimize for deployment latency and infrastructure costs, specific architectural trade-offs were made:

1.  **Batch Processing vs. Real-Time Streaming:** We opted for a nightly PySpark batch ETL pipeline over Spark Structured Streaming. This constraint accepts a 24-hour freshness SLA to drastically reduce compute costs and infrastructure complexity.
2.  **Overwrite Semantics vs. Delta Lake:** The Transformation Engine overwrites entire partitions instead of utilizing `UPSERT`/`MERGE` operations on a Delta Lake. This is viable at the current megabyte scale but necessitates a transition to Delta formats at terabyte scale.
3.  **Local Model Registry vs. Cloud Tracking:** Models are tracked via serialized `.pkl` files within the Docker image (`dashboard/models`) rather than a managed MLflow/S3 registry. This tight coupling simplifies local containerized deployment at the cost of independent model lifecycle management.
4.  **Hardcoded Fragility:** By skipping environment variables for file paths, the ETL scripts rely on the strict directory configuration provided by the `Dockerfile` volume mounts.

---

## 3. Engineering Rigor & Methodologies

This implementation strictly adheres to advanced Data Science principles, ensuring models are not only performant in training but robust in production environments.

### 3.1 Data Engineering Strategy
*   **Storage Optimization:** Raw data is persisted as Parquet files. Parquet's columnar format guarantees high compression ratios, while heavy reliance on **Predicate Pushdown** significantly minimizes I/O overhead during distributed Spark reads.
*   **Partition Pruning:** The processed data lake is structured hierarchically by `year` and `month`. Downstream queries and ML data loaders selectively scan required partitions, ensuring the system can scale to multi-terabyte datasets without linear performance degradation.
*   **Idempotent ETL pipelines:** Pipeline execution is strictly idempotent, enabling reliable reruns in the event of upstream failures or late-arriving data.

### 3.2 Feature Engineering Rationale
Domain knowledge dictates that standalone metrics often fail to capture nuanced physiological states.
*   **Derived Metrics:** We engineered the `calories_to_steps_ratio` to capture activity intensity. This high-variance feature is critical; 500 calories burned over 2,000 steps indicates fundamentally different metabolic exertion (e.g., swimming/weightlifting) compared to 500 calories over 10,000 steps (walking). 
*   **Temporal Features:** Extraction of `day_of_week` isolates behavioral seasonality crucial for user segmentation.

### 3.3 Model Selection & Validation
We employ scikit-learn pipelines to construct modular architectures, averting data leakage and training-serving skew.

1.  **Activity Classification (Random Forest Classifier):**
    *   **Rationale:** Chosen for its robustness to non-linear relationships and immunity to feature scaling discrepancies compared to distance-based algorithms.
    *   **Metric Selection:** We track **Accuracy (84%)** given the relatively balanced class distribution across the 7 activity types. Precision and Recall mapping is utilized internally for threshold tuning on misclassified minority events.

2.  **Caloric Expenditure Prediction (Random Forest Regressor):**
    *   **Rationale:** An ensemble tree approach effectively captures the complex interactions between heart rate, activity type, and step count.
    *   **Metric Selection:** **$R^2$ (0.91)** and RMSE were prioritized to directly quantify the variance in metabolic burn successfully captured by the model vs the baseline mean prediction.

3.  **User Profiling (K-Means Clustering):**
    *   **Rationale:** Segmenting users based on aggregated behavioral metrics (avg steps, heart rate, calories) allows for targeted UI/UX adaptations.
    *   **Validation:** Silhouette scores and elbow-method inertia plots validate the optimal $k=5$ cluster configuration.

### 3.4 Production Readiness & MLOps
*   **Pipeline Encapsulation:** All model preprocessing steps (e.g., `SimpleImputer` for median imputation, `StandardScaler`, `OneHotEncoder`) are bundled into unified `Pipeline` objects. This guarantees that raw telemetry passed during live Streamlit inference undergoes the exact transformation graph as the training data, structurally eliminating training-serving skew.
*   **Containerization:** The execution environment is locked via Docker (`openjdk:11-jre-slim` + Python 3 binaries).
*   **CI/CD Constraints:** GitHub Actions enforce quality gates. Commits trigger immutable build verifications, execute `pytest` suites validating atomic ETL functions (e.g., zero-division handling), and run integration tests verifying the end-to-end Spark job functionality.

---

## 4. Local Deployment & Execution

**Prerequisites:** Docker Engine installed and running.

### Step 1: Initialize the Environment
Build the portable Docker image containing Spark binaries and Python dependencies:
```bash
docker build -t fitness-tracker-app .
```

### Step 2: Execute the Intelligence Pipeline
Run the ETL processes and train the ML models inside the isolated container:

```bash
# Spawn an interactive container mounted to the local codebase
docker run -it --rm -p 8888:8888 --name fitness-dev \
    --mount type=bind,source="$(pwd)",target=/app \
    fitness-tracker-app bash

# Execute the Data Engineering Pipeline
python src/etl_pipeline.py

# Execute the Machine Learning Pipeline
python src/train_dashboard_models.py
```

### Step 3: Serve the Application
Launch the decoupled Streamlit dashboard to interact with the processed data and serialized models:
```bash
docker run --rm -p 8501:8501 --name fitness-dashboard \
    --mount type=bind,source="$(pwd)",target=/app \
    fitness-tracker-app streamlit run /app/dashboard/1_Overview.py
```

Navigate to `http://localhost:8501` to view the platform.
