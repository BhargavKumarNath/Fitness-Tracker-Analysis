"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, ArrowUpRight } from "lucide-react";
import { motion } from "framer-motion";

const routes = [
  { href: "/", label: "Overview" },
  { href: "/exploratory", label: "Explore" },
  { href: "/insights", label: "Insights" },
  { href: "/modeling", label: "Models" },
  { href: "/predict", label: "Predict" },
];

export function SiteShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-paper/95">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-6 px-5 py-5 md:px-10">
          <Link href="/" className="group flex items-center gap-3" aria-label="Fitness Tracker Analysis home">
            <span className="flex h-9 w-9 items-center justify-center bg-ink text-paper"><Activity size={18} strokeWidth={1.5} /></span>
            <span className="font-display text-sm font-medium uppercase tracking-[0.14em]">Fitness<br />Tracker Analysis</span>
          </Link>
          <nav aria-label="Primary navigation" className="hidden items-center gap-1 md:flex">
            {routes.map((route) => {
              const active = route.href === "/" ? pathname === "/" : pathname.startsWith(route.href);
              return <Link key={route.href} href={route.href} className={`px-3 py-2 text-sm transition-colors ${active ? "text-cobalt" : "text-muted hover:text-ink"}`} aria-current={active ? "page" : undefined}>{route.label}</Link>;
            })}
          </nav>
          <Link href="/predict" className="hidden items-center gap-2 border border-ink px-4 py-2 text-sm font-medium transition-colors hover:bg-ink hover:text-paper sm:flex">Run a prediction <ArrowUpRight size={15} /></Link>
        </div>
        <nav aria-label="Mobile navigation" className="flex gap-1 overflow-x-auto border-t border-line px-5 py-2 md:hidden">
          {routes.map((route) => <Link key={route.href} href={route.href} className={`whitespace-nowrap px-3 py-2 text-sm ${pathname === route.href ? "text-cobalt" : "text-muted"}`}>{route.label}</Link>)}
        </nav>
      </header>
      <motion.main initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35 }}>{children}</motion.main>
      <footer className="mx-auto flex max-w-[1440px] justify-between border-t border-line px-5 py-8 text-xs text-muted md:px-10">
        <span>Static analytical publication / 2023 cohort</span>
        <span className="font-mono">v1.0 / DATA READY</span>
      </footer>
    </div>
  );
}