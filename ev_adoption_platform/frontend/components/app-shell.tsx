"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BatteryCharging,
  Blocks,
  ChartNoAxesCombined,
  DatabaseZap,
  Gauge,
  Menu,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  UsersRound,
  X
} from "lucide-react";
import { useState } from "react";

export type PageKey =
  | "overview"
  | "live-scoring"
  | "segments"
  | "market-dna"
  | "simulator"
  | "model"
  | "data-quality";

const navigation = [
  { key: "overview", label: "Overview", href: "/", icon: Blocks },
  { key: "live-scoring", label: "Live scoring", href: "/live-scoring", icon: Activity },
  { key: "segments", label: "Segments", href: "/segments", icon: UsersRound },
  { key: "market-dna", label: "Market DNA", href: "/market-dna", icon: ChartNoAxesCombined },
  { key: "simulator", label: "Simulator", href: "/simulator", icon: SlidersHorizontal },
  { key: "model", label: "Model", href: "/model", icon: Gauge },
  { key: "data-quality", label: "Data quality", href: "/data-quality", icon: ShieldCheck }
] as const;

export const pageCopy: Record<PageKey, { title: string; subtitle: string; eyebrow: string }> = {
  overview: { title: "Adoption command center", subtitle: "Where buyer intent, market signals, and policy opportunities meet.", eyebrow: "Decision overview" },
  "live-scoring": { title: "Scoring operations", subtitle: "A live view of incoming profiles moving through the serving layer.", eyebrow: "Operations" },
  segments: { title: "Buyer segments", subtitle: "Find the audiences where EV intent is concentrated and actionable.", eyebrow: "Audience intelligence" },
  "market-dna": { title: "Market DNA", subtitle: "See how the current scoring population differs from the historical reference.", eyebrow: "Population signals" },
  simulator: { title: "Adoption simulator", subtitle: "Test how access, incentives, and buyer context change purchase likelihood.", eyebrow: "Scenario lab" },
  model: { title: "Model story", subtitle: "From an explainable baseline to the best-performing neural ensemble.", eyebrow: "ML performance" },
  "data-quality": { title: "Data health", subtitle: "A focused view of warehouse readiness and the exceptions that matter.", eyebrow: "Pipeline trust" }
};

export function AppShell({
  page,
  children,
  onRefresh
}: {
  page: PageKey;
  children: React.ReactNode;
  onRefresh: () => void;
}) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const copy = pageCopy[page];

  return (
    <main className="app-root">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className={`app-sidebar ${mobileOpen ? "is-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><BatteryCharging size={21} strokeWidth={1.8} /></div>
          <div><strong>EV Buyer Intelligence</strong><span>Growth analytics</span></div>
          <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={20} /></button>
        </div>
        <nav className="side-nav" aria-label="Dashboard navigation">
          <span className="side-nav-label">Workspace</span>
          {navigation.map((item) => {
            const active = pathname === item.href;
            return (
              <Link key={item.href} href={item.href} className={active ? "active" : ""} onClick={() => setMobileOpen(false)}>
                <item.icon size={18} strokeWidth={1.65} /><span>{item.label}</span>{active && <i />}
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          <DatabaseZap size={17} strokeWidth={1.7} />
          <div><strong>Warehouse connected</strong><span>DuckDB serving layer</span></div>
        </div>
      </aside>
      {mobileOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

      <section className="app-stage">
        <header className="app-header">
          <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="page-heading">
            <span>{copy.eyebrow}</span>
            <h1>{copy.title}</h1>
            <p>{copy.subtitle}</p>
          </div>
          <div className="header-actions">
            <Link href="/home" className="site-link">View site</Link>
            <div className="system-state"><i /> Online</div>
            <button className="icon-button" onClick={onRefresh} aria-label="Refresh dashboard" title="Refresh dashboard"><RefreshCw size={17} /></button>
          </div>
        </header>
        <div id="main-content" className="page-content">{children}</div>
      </section>
    </main>
  );
}
