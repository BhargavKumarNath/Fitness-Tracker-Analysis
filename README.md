# Fitness Tracker Analysis

Fitness Tracker Analysis turns daily fitness exports into a dataset that people can actually use. It cleans and enriches the raw records, trains a small set of predictive models, and presents the results in a Streamlit dashboard.

**Live dashboard:** [fitness-tracker-analysis.streamlit.app](https://fitness-tracker-analysis.streamlit.app/)

The project grew from a practical engineering problem: fitness data is easy to collect but surprisingly awkward to operate. Files arrive in nested directories, exports can use different formats, and the code that trains a model often makes different assumptions from the code that serves it. This repository brings those steps into one repeatable batch workflow.

## The Problem

A raw fitness export is not yet an analytical product. Before it can support a chart or a prediction, several things must line up:

- Files must be found reliably regardless of the directory from which a command is run.
- Valid parquet data must be separated from unrelated files in the same data lake.
- Dates and derived measures must be calculated consistently.
- Processed data must be written in a form that can be queried by time period.
- Training and inference must apply the same preprocessing.
- A small test dataset must not fail simply because it has fewer users than a default clustering configuration.
- Large generated model files must stay out of source control.

Without those boundaries, a pipeline can appear to work while remaining difficult to reproduce. A developer may train models from one path, a dashboard may load data from another, and a seemingly harmless rerun may leave behind incompatible outputs. The difficult part is not just choosing an estimator; it is making the entire path from file to prediction predictable.

## The Solution

The repository uses a straightforward batch design with explicit module boundaries:

1. The extractor discovers nested `.parquet` files and reads them through one PyArrow dataset scan. Other files, including CSV exports, are ignored.
2. The Spark transformation parses dates and adds the derived fields used by analysis.
3. The loader writes the result as parquet partitions by year and month.
4. The training module builds three scikit-learn pipelines and saves them for the dashboard.
5. The Streamlit application reads the processed data and uses the saved pipelines for exploration and inference.

`src/run_pipeline.py` connects these stages. The individual modules can also be tested independently, which makes failures easier to locate and changes safer to review.

```text
raw parquet files
        |
        v
src/etl/extract.py
        |
        v
src/etl/transform.py  (PySpark feature engineering)
        |
        v
src/etl/load.py  -->  data_lake/processed/fitness_data/
        |
        v
src/models/training.py
        |
        v
dashboard/models/*.pkl  -->  Streamlit dashboard
```

## Methodology

### Input and configuration

Paths are resolved from the repository root by `src/config.py`. Set `FITNESS_TRACKER_ROOT` when running against another project copy. The input may be partitioned into directories such as `year=2023/month=04/day=01`; the extractor searches below that root rather than depending on a fixed list of folders.

### Transformation

The Spark transformation adds:

- `day_of_week`, derived from `date`;
- `calories_to_steps_ratio`, calculated as calories burned divided by steps;
- a zero value for records with no steps, avoiding division-by-zero failures;
- `year` and `month`, which become output partition columns.

### Output

Processed records are written to `data_lake/processed/fitness_data/` as parquet. The current batch job uses overwrite semantics, making a rerun easy to reason about at the project’s current scale. It is not an incremental or transactional data lake implementation.

### Model handoff

The training code stores preprocessing with each estimator. The dashboard therefore receives a complete prediction pipeline rather than a model that requires a second, separately maintained feature transformation.

## Models and Why They Are Used

### Activity prediction

A Random Forest classifier predicts `activity_type` from steps, calories burned, and average heart rate. It is a reasonable choice for this tabular data because activity labels can depend on nonlinear combinations of measurements. Median imputation and standardization happen inside the saved pipeline.

### Calorie prediction

A Random Forest regressor predicts `calories_burned` from steps, average heart rate, sleep hours, and activity type. Numeric columns are imputed and scaled; the activity label is one-hot encoded. Unknown activity categories are ignored during encoding so inference does not fail when a new label appears.

### User segmentation

The segmentation model first reduces daily records to one row per user using average steps, calories, and heart rate. After imputation and standardization, K-Means assigns behavioral groups. The requested maximum is five clusters, but the implementation caps that value at the number of users so small datasets remain usable.

The two Random Forest models use 50 trees, a maximum depth of 20, and all available CPU workers. This keeps training bounded and reduces the size of generated artifacts. The model files are runtime outputs and are intentionally ignored by Git.

## Repository Layout

```text
Fitness-Tracker-Analysis/
├── data_lake/
│   ├── raw/synthetic_user_data/       raw fitness exports
│   └── processed/fitness_data/        partitioned ETL output
├── dashboard/
│   ├── 1_Overview.py                  Streamlit entrypoint
│   ├── pages/                         dashboard views
│   └── utils.py                       data and model loading
├── notebooks/                         exploratory reference notebook
├── src/
│   ├── etl/
│   │   ├── extract.py                 parquet input
│   │   ├── transform.py                Spark features
│   │   ├── load.py                    parquet output
│   │   └── run.py                     ETL orchestration
│   ├── models/training.py             model construction and persistence
│   ├── config.py                      runtime paths
│   └── run_pipeline.py                end-to-end orchestration
├── tests/                             unit and integration tests
├── archive/legacy/                    retired or experimental scripts
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated model binaries, local streaming input, Python environments, caches, and `project_refactor.md` are not source files and are excluded from version control.

## Getting Started

### Requirements

- Python 3.9 or newer
- Java suitable for the installed PySpark version
- Enough disk space for local data and generated model artifacts

### Install

```bash
git clone <repository-url>
cd Fitness-Tracker-Analysis
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

By default, paths resolve from the repository root. To use a different project location:

```bash
export FITNESS_TRACKER_ROOT=/path/to/fitness-project
```

### Run the pipeline

```bash
python -m src.run_pipeline
```

The command reads raw parquet files, writes processed partitions, and creates these local artifacts:

```text
dashboard/models/activity_classifier.pkl
dashboard/models/calories_regressor.pkl
dashboard/models/user_segmentation.pkl
dashboard/models/cluster_features.pkl
```

### Start the dashboard

```bash
streamlit run dashboard/1_Overview.py
```

Open `http://localhost:8501`. If a model file is missing, the dashboard can attempt to download it using the identifiers configured in `dashboard/utils.py`. Generating the artifacts locally is preferable for repeatable offline runs.

### Run the tests

```bash
python -m pytest -q
```

The integration tests use temporary input and output directories, so they do not require the large local model files.

### Run with Docker

```bash
docker build -t fitness-tracker-app .
docker run --rm \
  --mount type=bind,source="$(pwd)",target=/app \
  fitness-tracker-app
```

To start the dashboard from the image:

```bash
docker run --rm -p 8501:8501 \
  --mount type=bind,source="$(pwd)",target=/app \
  fitness-tracker-app \
  streamlit run /app/dashboard/1_Overview.py --server.address=0.0.0.0
```

## Results and Benchmarks

The following measurements were collected in the project virtual environment on 2026-08-25.

| Check | Result |
| --- | ---: |
| Automated tests | 13 passed |
| Raw dataset measured | 358,497 rows across 183 parquet files |
| Extraction before Arrow scan | 0.441 s |
| Extraction after Arrow scan | 0.055 s |
| Extraction improvement | approximately 88% |
| Model training benchmark | 0.438 s for 10,000 rows |
| ETL smoke test | 7.13 s, 124 MB peak RSS |
| Training smoke test | 1.09 s, 224 MB peak RSS |
| End-to-end smoke output | 2 rows, 1 partition, 4 artifacts |

The extraction result was measured on the full repository dataset. Model training was measured on a bounded sample because full Random Forest artifacts can consume several gigabytes of disk and memory. The benchmark is an engineering reference, not a production service-level objective.

The validation run also confirmed that malformed or unsupported inputs fail at clear boundaries: missing parquet input raises an error, mixed CSV/parquet directories remain readable, invalid model names are rejected, and small user populations do not exceed the K-Means sample limit.

## Scope and Next Steps

This is a tested batch workflow. It does not currently provide Spark Structured Streaming, incremental or transactional updates, automated data-quality monitoring, model-drift monitoring, a managed model registry, or committed CI configuration.

The most useful next improvements would be incremental partition processing, formal data-quality checks, coverage reporting in CI, and versioned model artifacts instead of local files and download URLs.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
