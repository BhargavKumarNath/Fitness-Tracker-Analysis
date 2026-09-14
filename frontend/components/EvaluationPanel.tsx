"use client";

import { useEffect, useState } from "react";
import type { ModelEvalSampleArtifact } from "@/lib/types";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { Skeleton } from "@/components/Skeleton";

export function EvaluationPanel({ data }: { data: ModelEvalSampleArtifact }) {
  const [ready, setReady] = useState(false);
  useEffect(() => { const timer = window.setTimeout(() => setReady(true), 350); return () => window.clearTimeout(timer); }, []);
  if (!ready) return <div className="grid gap-4 md:grid-cols-2" aria-busy="true"><Skeleton className="h-52 w-full" /><Skeleton className="h-52 w-full" /></div>;
  return <div className="space-y-4"><div className="grid gap-4 md:grid-cols-3"><div className="border border-line p-5"><p className="text-sm text-muted">Classification accuracy</p><p className="mt-6 font-display text-3xl">{(data.classification.accuracy * 100).toFixed(1)}%</p></div><div className="border border-line p-5"><p className="text-sm text-muted">Regression RMSE</p><p className="mt-6 font-display text-3xl">{data.regression.rmse.toFixed(1)}</p></div><div className="border border-line p-5"><p className="text-sm text-muted">Evaluation rows</p><p className="mt-6 font-display text-3xl">{data.sampleSize.toLocaleString()}</p></div></div><div className="border border-line p-5"><p className="mb-5 font-mono text-xs uppercase tracking-[0.14em] text-cobalt">Confusion matrix / {data.classification.source}</p><ConfusionMatrix labels={data.classification.labels} matrix={data.classification.confusionMatrix} /></div></div>;
}