"use client";

import { useState } from "react";
import { queryActivitySummary, type FilteredActivitySummary } from "@/lib/duckdb";

export function DuckDbFilter({ categories, onResult }: { categories: string[]; onResult: (result: FilteredActivitySummary[] | undefined) => void }) {
  const [activity, setActivity] = useState("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  async function update(value: string) {
    setActivity(value);
    setError(undefined);
    if (value === "all") { onResult(undefined); return; }
    setLoading(true);
    try {
      onResult(await queryActivitySummary(value));
    } catch {
      setError("Parquet filter unavailable. Showing full-dataset aggregates instead.");
      onResult(undefined);
    } finally {
      setLoading(false);
    }
  }
  return <label className="flex items-center gap-3 text-sm"><span className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Parquet filter</span><select value={activity} onChange={(event) => update(event.target.value)} className="border border-line bg-paper px-3 py-2 text-sm outline-none focus:border-cobalt" aria-label="Filter activity charts by activity type"><option value="all">All activities</option>{categories.map((category) => <option key={category} value={category}>{category.replaceAll("_", " ")}</option>)}</select>{loading && <span className="text-xs text-muted">Querying...</span>}{error && <span role="alert" className="text-xs text-signal">{error}</span>}</label>;
}