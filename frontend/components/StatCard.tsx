"use client";

import { motion, useReducedMotion } from "framer-motion";

export function StatCard({ label, value, detail }: { label: string; value: string; detail?: string }) {
  const reducedMotion = useReducedMotion();
  return <motion.article initial={reducedMotion ? undefined : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="border border-line bg-paper p-5"><p className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">{label}</p><p className="mt-8 font-display text-4xl tracking-[-0.04em]">{value}</p>{detail && <p className="mt-2 text-xs text-muted">{detail}</p>}</motion.article>;
}