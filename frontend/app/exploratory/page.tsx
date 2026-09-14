"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { ActivitySummaryChart, CorrelationGrid, HealthChart, TemporalChart } from "@/components/ExploratoryCharts";
import { ChartCard } from "@/components/ChartCard";
import { DuckDbFilter } from "@/components/DuckDbFilter";
import activityAnalysis from "@/public/data/activity_analysis.json";
import healthMetrics from "@/public/data/health_metrics.json";
import temporal from "@/public/data/temporal.json";
import segmentation from "@/public/data/segmentation.json";
import type { ActivityAnalysisArtifact, HealthMetricsArtifact, SegmentationArtifact, TemporalArtifact } from "@/lib/types";
import type { FilteredActivitySummary } from "@/lib/duckdb";

const ClusterScene = dynamic(() => import("@/components/ClusterScene").then((module) => module.ClusterScene), { ssr: false, loading: () => <div className="flex h-[460px] items-center justify-center bg-[#ebe9e2] text-sm text-muted">Preparing point cloud...</div> });
const activities = activityAnalysis as ActivityAnalysisArtifact;
const health = healthMetrics as HealthMetricsArtifact;
const days = temporal as TemporalArtifact;
const clusters = segmentation as SegmentationArtifact;

export default function ExploratoryPage() {
	const [filtered, setFiltered] = useState<FilteredActivitySummary[] | undefined>();
	return <div className="mx-auto max-w-[1440px] px-5 py-14 md:px-10 md:py-24"><p className="font-mono text-xs uppercase tracking-[0.18em] text-cobalt">02 / Exploratory analysis</p><div className="mt-5 flex flex-col justify-between gap-8 border-b border-ink pb-10 md:flex-row md:items-end"><div><h1 className="font-display text-5xl tracking-[-0.04em] md:text-7xl">Explore</h1><p className="mt-5 max-w-xl text-lg leading-8 text-muted">The cohort&apos;s movement, recovery, and temporal rhythms, composed from deterministic aggregates.</p></div><DuckDbFilter categories={activities.categories} onResult={setFiltered} /></div><div className="mt-10 grid gap-4 lg:grid-cols-2"><ChartCard eyebrow="Activity comparison" title="Average steps and calories"><ActivitySummaryChart data={activities.summary} filtered={filtered} /></ChartCard><ChartCard eyebrow="Health metric" title="Average heart rate by activity"><HealthChart data={health.heartRateByActivity} /></ChartCard><ChartCard eyebrow="Temporal pattern" title="Daily movement and heart rate"><TemporalChart data={days.daily} /></ChartCard><ChartCard eyebrow="Correlation" title="Signals moving together"><CorrelationGrid data={health.correlation} labels={health.numericColumns} /><p className="mt-4 text-xs text-muted">Rows and columns: {health.numericColumns.join(", ")}</p></ChartCard></div><section className="mt-16 border-t border-line pt-10"><div className="flex flex-col justify-between gap-4 md:flex-row md:items-end"><div><p className="font-mono text-xs uppercase tracking-[0.18em] text-cobalt">User segmentation / {clusters.method === "kmeans" ? "K-Means" : "deterministic fallback"}</p><h2 className="mt-3 font-display text-3xl tracking-[-0.03em]">A point cloud of behavior</h2><p className="mt-3 max-w-xl text-sm leading-7 text-muted">Drag to orbit through {clusters.users.length.toLocaleString()} user profiles. Position encodes average steps, calories, and heart rate.</p></div><span className="font-mono text-xs text-muted">WEBGL / INTERACTIVE</span></div><div className="mt-8 grid gap-4 lg:grid-cols-[1.25fr_0.75fr]"><ClusterScene users={clusters.users} /><div className="space-y-2">{clusters.clusters.map((cluster) => <article key={cluster.cluster} className="border border-line p-4"><div className="flex items-baseline justify-between"><h3 className="font-display text-lg">{cluster.label}</h3><span className="font-mono text-xs text-muted">{cluster.users} users</span></div><p className="mt-2 text-sm text-muted">{cluster.message}</p><p className="mt-4 font-mono text-xs text-cobalt">{Math.round(cluster.avgSteps).toLocaleString()} avg steps / {Math.round(cluster.avgHeartRate)} bpm</p></article>)}</div></div></section></div>;
}