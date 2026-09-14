export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="border border-dashed border-line px-6 py-12 text-center"><p className="font-display text-lg">{title}</p><p className="mx-auto mt-2 max-w-md text-sm text-muted">{detail}</p></div>;
}