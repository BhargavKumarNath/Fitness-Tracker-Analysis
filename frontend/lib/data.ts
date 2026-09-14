import overview from "@/public/data/overview.json";
import dataQuality from "@/public/data/data_quality.json";
import modelMetrics from "@/public/data/model_metrics.json";
import type { DataQualityArtifact, ModelMetricsArtifact, OverviewArtifact } from "@/lib/types";

export const overviewData = overview as OverviewArtifact;
export const dataQualityData = dataQuality as DataQualityArtifact;
export const modelMetricsData = modelMetrics as ModelMetricsArtifact;