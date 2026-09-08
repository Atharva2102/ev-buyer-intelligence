"use client";

import Image from "next/image";
import { motion, useReducedMotion } from "motion/react";
import { ArrowUpRight, ChevronRight } from "lucide-react";
import { vehicleImages } from "../lib/site";
import type { SegmentMetric } from "../lib/api";

export function Reveal({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion();
  return (
    <motion.div className={className} initial={reduce ? false : { opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] }}>
      {children}
    </motion.div>
  );
}

export function MetricTile({ label, value, detail, icon, progress, delay = 0 }: { label: string; value: string; detail: string; icon: React.ReactNode; progress: number; delay?: number }) {
  return (
    <Reveal className="metric-tile" delay={delay}>
      <div className="metric-top"><span className="metric-icon">{icon}</span><ArrowUpRight size={16} /></div>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-foot"><span>{detail}</span><div className="micro-line"><i style={{ transform: `scaleX(${Math.max(0.08, Math.min(progress, 1))})` }} /></div></div>
    </Reveal>
  );
}

export function SectionTitle({ title, action }: { title: string; action?: React.ReactNode }) {
  return <div className="section-title"><h2>{title}</h2>{action}</div>;
}

export function VehicleCard({ segment, selected = false, onClick }: { segment: SegmentMetric; selected?: boolean; onClick?: () => void }) {
  const image = vehicleImages[segment.segment];
  const content = (
    <>
      <div className="vehicle-card-head"><div><h3>{segment.segment}</h3><span>{segment.population.toLocaleString()} profiles</span></div><ChevronRight size={18} /></div>
      {image && <Image src={image} alt={`Unbranded electric ${segment.segment.toLowerCase()} side profile`} width={720} height={420} className="vehicle-art" />}
      <div className="vehicle-stats"><div><span>Purchase intent</span><strong>{(segment.avg_probability * 100).toFixed(1)}%</strong></div><div><span>High intent</span><strong>{segment.high_intent_buyers.toLocaleString()}</strong></div></div>
    </>
  );
  return onClick ? <button className={`vehicle-card ${selected ? "selected" : ""}`} onClick={onClick}>{content}</button> : <article className="vehicle-card">{content}</article>;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="empty-state"><span className="empty-pulse" /><p>{message}</p></div>;
}

export function Skeleton({ rows = 4 }: { rows?: number }) {
  return <div className="skeleton-stack" aria-label="Loading data">{Array.from({ length: rows }).map((_, index) => <i key={index} style={{ width: `${94 - index * 7}%` }} />)}</div>;
}
