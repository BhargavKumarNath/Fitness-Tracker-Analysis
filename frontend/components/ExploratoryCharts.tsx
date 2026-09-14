"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ActivityAnalysisArtifact, HealthMetricsArtifact, TemporalArtifact } from "@/lib/types";
import type { FilteredActivitySummary } from "@/lib/duckdb";

export function ActivitySummaryChart({ data, filtered }: { data: ActivityAnalysisArtifact["summary"]; filtered?: FilteredActivitySummary[] }) {
  const rows = filtered?.length ? filtered.map((row) => ({ activityType: row.activityType, steps: row.averageSteps, calories: row.averageCalories })) : data.map((row) => ({ activityType: row.activityType, steps: row.steps.mean, calories: row.calories.mean }));
  return <div role="img" aria-label="Average steps and calories by activity"><ResponsiveContainer width="100%" height={290}><BarChart data={rows} margin={{ bottom: 34, left: -20 }}><CartesianGrid stroke="#d9d7d0" vertical={false} /><XAxis dataKey="activityType" angle={-24} textAnchor="end" height={58} tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><YAxis tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: "1px solid #d9d7d0", borderRadius: 0, background: "#f4f3ef", fontSize: 12 }} /><Bar dataKey="steps" name="Avg steps" fill="#1746a2" /><Bar dataKey="calories" name="Avg calories" fill="#e24a3b" /></BarChart></ResponsiveContainer></div>;
}

export function TemporalChart({ data }: { data: TemporalArtifact["daily"] }) {
  return <div role="img" aria-label="Daily average steps and heart rate over time"><ResponsiveContainer width="100%" height={290}><LineChart data={data} margin={{ left: -20 }}><CartesianGrid stroke="#d9d7d0" vertical={false} /><XAxis dataKey="date" tick={{ fill: "#686861", fontSize: 10 }} tickLine={false} axisLine={false} minTickGap={32} /><YAxis tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: "1px solid #d9d7d0", borderRadius: 0, background: "#f4f3ef", fontSize: 12 }} /><Line type="monotone" dataKey="steps" stroke="#1746a2" dot={false} strokeWidth={2} /><Line type="monotone" dataKey="heartRate" stroke="#e24a3b" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div>;
}

export function HealthChart({ data }: { data: HealthMetricsArtifact["heartRateByActivity"] }) {
  return <ResponsiveContainer width="100%" height={290}><BarChart data={data} margin={{ bottom: 34, left: -20 }}><CartesianGrid stroke="#d9d7d0" vertical={false} /><XAxis dataKey="activityType" angle={-24} textAnchor="end" height={58} tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><YAxis tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: "1px solid #d9d7d0", borderRadius: 0, background: "#f4f3ef", fontSize: 12 }} /><Bar dataKey="average" name="Avg heart rate" fill="#e24a3b" /></BarChart></ResponsiveContainer>;
}

export function CorrelationGrid({ data, labels }: { data: number[][]; labels: string[] }) {
  return <div className="grid grid-cols-4 gap-px border border-line bg-line">{data.flatMap((row, rowIndex) => row.map((value, columnIndex) => <div key={`${rowIndex}-${columnIndex}`} className="flex aspect-square flex-col justify-between p-2 text-[10px]" style={{ backgroundColor: `rgba(23, 70, 162, ${Math.min(0.8, Math.abs(value) * 0.8 + 0.06)})`, color: Math.abs(value) > 0.45 ? "#fff" : "#171717" }}><span>{rowIndex === 0 ? labels[columnIndex] : ""}</span><strong className="font-mono text-xs">{value.toFixed(2)}</strong></div>))}</div>;
}