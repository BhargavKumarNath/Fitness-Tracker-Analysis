# Frontend Data Contracts

This document is the contract between the batch export step and the static Next.js application. JSON files are UTF-8, use camelCase keys, and contain finite JSON numbers only. Dates are ISO `YYYY-MM-DD` strings. The processed parquet remains the row-level source for client-side filtering.

## Artifact inventory

| Artifact | Approximate size | Consumers | Source / coverage |
| --- | ---: | --- | --- |
| `overview.json` | < 2 KB | Overview KPI cards, activity distribution | Full dataset |
| `activity_analysis.json` | < 150 KB | Activity comparison, sampled scatter, distributions | Per-activity aggregates plus deterministic samples |
| `health_metrics.json` | < 25 KB | Correlation heatmap, heart-rate and sleep charts | Full-dataset aggregates and histogram bins |
| `temporal.json` | < 30 KB | Daily and weekday trends | Full-dataset group-bys |
| `segmentation.json` | < 400 KB | User point cloud, cluster summary, narratives | One profile per user plus cluster centroids |
| `data_quality.json` | < 25 KB | Statistics, completeness, validity, outliers, findings | Full dataset |
| `model_metrics.json` | < 5 KB | Model comparison and overview | `artifacts/metrics.json` |
| `model_eval_sample.json` | < 100 KB | Precomputed modelling evaluation | Deterministic 2,000-row metrics sample, 500 plotted pairs, seed 42 |
| `fitness.parquet` | ~9 MB | Optional DuckDB-WASM filters | Compacted processed dataset |

## Shared types

The TypeScript interfaces below are the canonical shape. The frontend keeps these definitions in `frontend/lib/types.ts`; this document is the human-readable contract and is intentionally kept next to the generated data directory.

```ts
type DateString = string; // ISO YYYY-MM-DD
type ActivityType = string; // verified values come from the exported categories array

interface OverviewArtifact {
  generatedAt: string;
  rowCount: number;
  userCount: number;
  activityCount: number;
  averageSteps: number;
  averageCalories: number;
  averageHeartRate: number;
  averageSleepHours: number;
  dateRange: { start: DateString; end: DateString; days: number };
  activities: Array<{ activityType: ActivityType; count: number; share: number }>;
}

interface ActivityAnalysisArtifact {
  categories: ActivityType[];
  summary: Array<{
    activityType: ActivityType; count: number;
    steps: { mean: number; median: number; std: number };
    calories: { mean: number; median: number; std: number };
    heartRate: { mean: number; median: number; std: number };
    sleepHours: { mean: number; median: number; std: number };
  }>;
  stepsCaloriesSample: Array<{ steps: number; calories: number; activityType: ActivityType }>;
  heartRateCaloriesSample: Array<{ heartRate: number; calories: number; activityType: ActivityType }>;
  histograms: Record<string, Array<{ start: number; end: number; count: number }>>;
}

interface HealthMetricsArtifact {
  numericColumns: string[];
  correlation: number[][];
  heartRateByActivity: Array<{ activityType: ActivityType; average: number }>;
  sleepByActivity: Array<{ activityType: ActivityType; average: number }>;
  heartRateHistogram: Array<{ start: number; end: number; count: number }>;
  sleepHistogram: Array<{ start: number; end: number; count: number }>;
}

interface TemporalArtifact {
  daily: Array<{ date: DateString; steps: number; calories: number; heartRate: number }>;
  weekdays: Array<{ day: string; steps: number; calories: number; count: number }>;
}

interface SegmentationArtifact {
  method: "kmeans" | "activity-band-fallback";
  features: ["avgSteps", "avgCalories", "avgHeartRate"];
  users: Array<{ userId: string | number; avgSteps: number; avgCalories: number; avgHeartRate: number; cluster: number }>;
  clusters: Array<{ cluster: number; users: number; avgSteps: number; avgCalories: number; avgHeartRate: number; label: string; message: string }>;
}

interface DataQualityArtifact {
  statistics: Array<{ metric: string; mean: number; median: number; std: number; min: number; max: number; q1: number; q3: number; iqr: number; skewness: number; kurtosis: number }>;
  completeness: Array<{ column: string; missing: number; completeness: number; dtype: string; uniqueValues: number; sampleValue: string }>;
  duplicates: { count: number; percentage: number; uniqueRecords: number };
  validity: Array<{ metric: string; expectedMin: number; expectedMax: number; outOfRange: number; validPercentage: number }>;
  outliers: Array<{ metric: string; count: number; percentage: number; lowerBound: number; upperBound: number }>;
  findings: { mostPopularActivity: ActivityType; averageSteps: number; highPerformerPercentage: number; averageHeartRate: number; averageSleepHours: number; heartRateMin: number; heartRateMax: number; stepsCaloriesCorrelation: number; heartRateCaloriesCorrelation: number; mostEfficientActivity: ActivityType };
}

interface ModelMetricsArtifact {
  activityClassifier: { heldOutAccuracy: number | null };
  caloriesRegressor: { heldOutR2: number | null; heldOutRmse: number | null };
  methodology?: string;
}

interface ModelEvalSampleArtifact {
  sampleSize: number;
  classification: { accuracy: number; labels: string[]; confusionMatrix: number[][]; report: Record<string, Record<string, number>>; source: "baseline" | "trained-model" };
  regression: { rmse: number; r2: number; pairs: Array<{ actual: number; predicted: number }>; source: "baseline" | "trained-model" };
}
```

## Visual coverage

- Overview KPI cards, activity distribution, architecture/model summary: `overview.json`, `model_metrics.json`.
- Exploratory overview and activity tabs: `overview.json`, `activity_analysis.json`.
- Health metrics tab: `health_metrics.json`.
- Temporal tab: `temporal.json`.
- User segmentation tab: `segmentation.json`; the export records whether it used the trained K-Means pipeline or the existing deterministic fallback.
- Data insights statistics, quality, validity, outliers, and findings: `data_quality.json`.
- Advanced modelling evaluation and comparison: `model_metrics.json`, `model_eval_sample.json`.
- Live inference: TypeScript baseline functions with the same thresholds and factors as `src/predictions.py`.

## Decisions and deviations

- The export step uses pandas/NumPy because they are already installed and preserve the pipeline's exact numerical behavior. Polars/DuckDB remain suitable follow-up optimizations.
- No row-level JSON is shipped. Only deterministic aggregates/samples are JSON; `fitness.parquet` is the optional interactive source.
- The optional model files are not required. When unavailable, evaluation exports use the same baseline functions the frontend ships.
- The roadmap’s Swiss design requirement takes precedence over its earlier dark-gradient/glassmorphism suggestion: the product uses a disciplined light editorial canvas, a near-black ink color, blue accent, restrained red status color, and no decorative gradients.