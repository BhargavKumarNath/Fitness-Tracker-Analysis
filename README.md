# Fitness Tracker Analysis

A batch pipeline that cleans synthetic fitness tracker data (steps, calories, heart rate, sleep, activity type), trains a few models on it, and serves the results as a static dashboard. No backend, no live API. The Python side runs once and writes files; the frontend just reads them.

## Motivation

Raw fitness exports are annoying to work with in practice. Parquet files end up nested at random directory depths, sometimes with a stray CSV sitting next to them. Any per-step calculation (calories per step, for example) needs a division-by-zero guard for rest days. And clustering code that assumes thousands of users breaks the moment you run it against a 3-row test fixture. This repo exists to handle that mess once, in code that's tested, instead of redoing it by hand in a notebook every time.

## Architecture

```
raw parquet (data_lake/raw)
  -> extract.py     PyArrow multi-file dataset scan
  -> transform.py   Spark: day_of_week, calories_to_steps_ratio
  -> load.py        partitioned parquet, by year/month
  -> training.py    classifier + regressor + KMeans, with held-out metrics
  -> export_frontend_artifacts.py
  -> frontend/public/data/*.json, fitness.parquet
```

`python -m src.run_pipeline` runs extract, transform, load, and training as one step. Everything after the processed parquet file is a deterministic function of static data, there's no live ingestion and nothing needs a running server.

## Models

Three plain sklearn pipelines, nothing exotic:

- **Activity classifier**: RandomForestClassifier (50 trees, max depth 20), predicts `activity_type` from steps, calories, and heart rate.
- **Calorie regressor**: RandomForestRegressor, same hyperparameters, predicts calories from steps, heart rate, sleep hours, and activity type (one-hot encoded, unseen categories are ignored rather than crashing).
- **Segmentation**: KMeans on each user's average steps/calories/heart rate. Cluster count is capped at `min(5, num_users)` so it doesn't error out on small datasets.

Held-out accuracy and R² come from cloning each pipeline, fitting it on an 80/20 split, and scoring the held-out 20%. The model that actually ships is still fit on all the data; the split only exists to produce an honest number. Below 20 rows, or a class with fewer than 2 examples, it reports `null` instead of a meaningless score. Current numbers on the full dataset:

| Metric | Value |
|---|---|
| Activity classifier accuracy | 89.3% |
| Calorie regressor R² | 0.92 |
| Calorie regressor RMSE | 126 |

There's also a plain rule-based baseline (`src/predictions.py`) with no model behind it at all, used for instant predictions in the frontend and as the reference point the trained models are compared against in the evaluation export. If a trained segmentation model isn't available, `src/serving.py` falls back to a deterministic step-rank banding instead of failing.

## Frontend

`frontend/` is a statically exported Next.js app, deployed on Vercel, with no API routes. Every page renders from precomputed JSON written by the export step. Two things happen in the browser instead of at build time: filtering queries the parquet file directly via DuckDB-WASM, and the 3D user segmentation view (`react-three-fiber`) is lazy-loaded so it's not part of the main bundle. Exact schemas for each JSON artifact are in `frontend/DATA_CONTRACTS.md`.

## Getting Started

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m src.run_pipeline    # ETL + training
python -m pytest -q           # tests

python scripts/export_frontend_artifacts.py
cd frontend
npm install
npm run typecheck
npm test -- --run
npm run build
```

You need a JDK on your PATH for PySpark. If you're running against a data directory that isn't this checkout, set `FITNESS_TRACKER_ROOT` to point at it.

Docker just runs the Spark ETL job:

```bash
docker build -t fitness-tracker-app .
docker run --rm --mount type=bind,source="$(pwd)",target=/app fitness-tracker-app
```

## Project Structure

```
data_lake/            raw and processed parquet
src/etl/              extract, transform, load
src/models/           training
src/predictions.py    rule-based baseline, shared with the frontend
src/serving.py        dataset/model loading, segmentation fallback
artifacts/            trained models + metrics.json
scripts/              frontend export step
frontend/             Next.js app
tests/
archive/legacy/       old scripts, not used by anything
```

## Limitations

- The dataset is synthetic and fixed: 182 days, 358,497 rows, 1,959 users. This is a batch job you rerun from scratch, not something built for streaming or incremental updates.
- `archive/legacy/` is old code kept for reference. Nothing imports it.
- Tests are run manually (`pytest`), there's no CI configured yet.

## License

MIT, see [LICENSE](LICENSE).
