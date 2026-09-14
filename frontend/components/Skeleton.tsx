export function Skeleton({ className = "h-6 w-full" }: { className?: string }) {
  return <span className={`block animate-pulse bg-[#e5e3dc] ${className}`} aria-hidden="true" />;
}