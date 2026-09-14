export function ChartCard({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return <section className="border border-line p-5 md:p-6"><p className="font-mono text-[11px] uppercase tracking-[0.14em] text-cobalt">{eyebrow}</p><h2 className="mt-2 font-display text-xl tracking-[-0.02em]">{title}</h2><div className="mt-6">{children}</div></section>;
}