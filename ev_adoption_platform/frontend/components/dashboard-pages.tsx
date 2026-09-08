"use client";

import Image from "next/image";
import {
  Activity, ArrowUpRight, BatteryCharging, Check, CircleGauge, Clock3, DatabaseZap,
  Gauge, GitCompareArrows, Layers3, Network, Route, ShieldCheck, Target, TrendingUp,
  UsersRound, Zap
} from "lucide-react";
import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PageKey } from "./app-shell";
import { EmptyState, MetricTile, Reveal, SectionTitle, Skeleton, VehicleCard } from "./visuals";
import { vehicleImages } from "../lib/site";
import {
  CustomerScore, DriftMetric, LiveFeed, OverviewMetric, PolicyScenario, PredictionRequest,
  PredictionResult, QualityMetric, SegmentMetric, postJson
} from "../lib/api";

type ViewProps = {
  page: PageKey;
  loading: boolean;
  metricMap: Record<string, OverviewMetric>;
  live: LiveFeed | null;
  segments: SegmentMetric[];
  carSegments: SegmentMetric[];
  customers: CustomerScore[];
  scenarios: PolicyScenario[];
  drift: DriftMetric[];
  quality: QualityMetric[];
  activeSegment: string;
  onSegmentChange: (value: string) => void;
};

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
const money = (value: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

export function DashboardView(props: ViewProps) {
  if (props.page === "overview") return <OverviewPage {...props} />;
  if (props.page === "live-scoring") return <LiveScoringPage {...props} />;
  if (props.page === "segments") return <SegmentsPage {...props} />;
  if (props.page === "market-dna") return <MarketDnaPage {...props} />;
  if (props.page === "simulator") return <SimulatorPage {...props} />;
  if (props.page === "model") return <ModelPage />;
  return <DataQualityPage {...props} />;
}

function OverviewPage({ loading, metricMap, live, scenarios, segments, carSegments }: ViewProps) {
  const top = segments[0];
  return (
    <div className="page-stack">
      <section className="metrics-grid">
        <MetricTile label="Customers scored" value={metricMap.customers_scored?.metric_label ?? "--"} detail="Current scoring population" progress={0.86} icon={<DatabaseZap />} />
        <MetricTile label="Average intent" value={metricMap.avg_ev_purchase_probability?.metric_label ?? "--"} detail="Across all scored buyers" progress={0.5} icon={<CircleGauge />} delay={0.05} />
        <MetricTile label="High-intent buyers" value={metricMap.high_intent_buyers?.metric_label ?? "--"} detail="Priority and high bands" progress={0.65} icon={<Target />} delay={0.1} />
        <MetricTile label="Historical adoption" value={metricMap.historical_adoption_rate?.metric_label ?? "--"} detail="Original reference cohort" progress={0.175} icon={<TrendingUp />} delay={0.15} />
      </section>

      <section className="overview-focus">
        <Reveal className="decision-brief">
          <span className="panel-kicker">Recommended focus</span>
          <h2>Remove the access barrier before changing the message.</h2>
          <p>Subsidies and charging access produce the strongest estimated lift across the current population.</p>
          <div className="decision-meta">
            <div><strong>{scenarios[0] ? num(scenarios[0].incremental_expected_buyers) : "--"}</strong><span>potential incremental buyers</span></div>
            <div><strong>{top ? pct(top.avg_probability) : "--"}</strong><span>top segment intent</span></div>
          </div>
        </Reveal>
        <PolicyOpportunity scenarios={scenarios} />
      </section>

      <section>
        <SectionTitle title="Intent by vehicle type" action={<span className="section-note">Real population segments</span>} />
        {loading ? <Skeleton rows={4} /> : <div className="vehicle-grid">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} />)}</div>}
      </section>

      <section className="overview-lower">
        <LiveQueue live={live} compact />
        <div className="segment-snapshot">
          <SectionTitle title="Audience signal" />
          <div className="snapshot-orbit"><div><span>Strongest segment</span><strong>{top?.segment ?? "Loading"}</strong><p>{top ? `${num(top.population)} profiles with ${pct(top.avg_probability)} average purchase intent.` : "Waiting for segment data."}</p></div><div className="orbit-score">{top ? pct(top.lift_vs_average, 1) : "--"}<span>lift</span></div></div>
        </div>
      </section>
    </div>
  );
}

function PolicyOpportunity({ scenarios }: { scenarios: PolicyScenario[] }) {
  const max = Math.max(...scenarios.map((item) => item.incremental_expected_buyers), 1);
  return (
    <Reveal className="policy-visual" delay={0.08}>
      <SectionTitle title="Policy opportunity" />
      <div className="policy-bars">{scenarios.map((item, index) => <div className="policy-row" key={item.scenario}><span>{item.scenario.replace(" for customers", "")}</span><div><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: item.incremental_expected_buyers / max }} transition={{ duration: 0.7, delay: 0.12 + index * 0.08 }} /><strong>+{num(item.incremental_expected_buyers)}</strong></div></div>)}</div>
    </Reveal>
  );
}

function LiveScoringPage({ live, customers }: ViewProps) {
  const priority = customers.filter((row) => row.adoption_band === "Priority").length;
  const avg = live?.rows.length ? live.rows.reduce((sum, row) => sum + row.ev_purchase_probability, 0) / live.rows.length : 0;
  return (
    <div className="page-stack">
      <section className="ops-strip">
        <MetricTile label="Batch status" value="Healthy" detail={live?.batch_id ?? "Waiting for service"} progress={0.94} icon={<Activity />} />
        <MetricTile label="Refresh cadence" value="5 sec" detail="Serving feed interval" progress={0.72} icon={<Clock3 />} delay={0.05} />
        <MetricTile label="Batch intent" value={pct(avg)} detail="Average visible score" progress={avg} icon={<Gauge />} delay={0.1} />
        <MetricTile label="Priority queue" value={String(priority)} detail="Top profiles loaded" progress={priority / Math.max(customers.length, 1)} icon={<Zap />} delay={0.15} />
      </section>
      <section className="live-layout"><LiveQueue live={live} /><BandDistribution rows={live?.rows ?? []} /></section>
      <section><SectionTitle title="Next best profiles" action={<span className="section-note">Sorted by model score</span>} /><div className="profile-grid">{customers.slice(0, 6).map((customer, index) => <ProfileCard key={customer.id} customer={customer} index={index} />)}</div></section>
    </div>
  );
}

function LiveQueue({ live, compact = false }: { live: LiveFeed | null; compact?: boolean }) {
  return (
    <div className={`live-queue ${compact ? "compact" : ""}`}>
      <SectionTitle title="Scoring queue" action={<span className="live-badge"><i /> Receiving</span>} />
      <div className="batch-meta"><span>{live?.batch_id ?? "Waiting for batch"}</span><span>{live ? new Date(live.generated_at).toLocaleTimeString() : "--"}</span></div>
      {live ? <div className="feed-list">{live.rows.slice(0, compact ? 5 : 8).map((row, index) => <motion.div key={`${live.batch_id}-${row.id}`} initial={false} style={{ animationDelay: `${index * 35}ms` }}><span className={`band-mark ${row.adoption_band.toLowerCase()}`} /><div><strong>Profile {row.id}</strong><span>{row.City_Type} / {row.Current_Car_Type} / {row.Range_Anxiety_Level} anxiety</span></div><b>{pct(row.ev_purchase_probability)}</b></motion.div>)}</div> : <EmptyState message="The scoring service is waiting for the next batch." />}
    </div>
  );
}

function BandDistribution({ rows }: { rows: LiveFeed["rows"] }) {
  const counts = ["Priority", "High", "Watch", "Low"].map((band) => ({ band, count: rows.filter((row) => row.adoption_band === band).length }));
  const strong = rows.length ? (counts[0].count + counts[1].count) / rows.length : 0;
  return <div className="band-panel"><SectionTitle title="Batch composition" /><div className="donut-wrap"><div className="score-donut" style={{ "--score": `${strong * 360}deg` } as React.CSSProperties}><strong>{pct(strong, 0)}</strong><span>high intent</span></div></div><div className="band-legend">{counts.map((item) => <div key={item.band}><i className={item.band.toLowerCase()} /><span>{item.band}</span><strong>{item.count}</strong></div>)}</div></div>;
}

function ProfileCard({ customer, index }: { customer: CustomerScore; index: number }) {
  return <Reveal className="profile-card" delay={index * 0.04}><div className="profile-card-top"><span>#{customer.id}</span><strong>{pct(customer.ev_purchase_probability)}</strong></div><div className="profile-avatar">{customer.City_Type.charAt(0)}{customer.Current_Car_Type.charAt(0)}</div><h3>{customer.City_Type} {customer.Current_Car_Type} owner</h3><p>{money(customer.Annual_Income_USD)} income / {customer.Daily_Commute_km.toFixed(0)} km commute</p><div className="profile-signals"><span>{customer.Home_Charging_Possible === "Yes" ? "Home charging" : "No home charging"}</span><span>{customer.Subsidy_Available === "Yes" ? "Subsidy" : "No subsidy"}</span></div></Reveal>;
}

function SegmentsPage({ segments, carSegments, activeSegment, onSegmentChange, customers }: ViewProps) {
  const labels: Record<string, string> = { buyer_segment: "Buyer profiles", city_home_charging_segment: "City and charging", subsidy_anxiety_segment: "Subsidy and anxiety" };
  return (
    <div className="page-stack">
      <section><SectionTitle title="Vehicle-class pulse" action={<span className="section-note">Current scoring population</span>} /><div className="vehicle-grid">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} />)}</div></section>
      <section className="segment-layout">
        <div className="ranking-panel"><div className="segment-tabs">{Object.entries(labels).map(([value, label]) => <button key={value} className={activeSegment === value ? "active" : ""} onClick={() => onSegmentChange(value)}>{label}</button>)}</div><SectionTitle title="Top opportunities" />{segments.slice(0, 5).map((segment) => <div className="rank-row" key={segment.segment}><span className="rank-number">{String(segment.rank_in_type).padStart(2, "0")}</span><div><strong>{segment.segment}</strong><span>{num(segment.population)} buyers</span></div><div className="rank-score"><strong>{pct(segment.avg_probability)}</strong><span>{pct(segment.lift_vs_average)} lift</span></div></div>)}</div>
        <div className="segment-context"><span className="panel-kicker">What stands out</span><h2>{segments[0]?.segment ?? "Loading segment"}</h2><p>The top-ranked group combines buyer context with the strongest observed model intent in this view.</p><ContextStat icon={<UsersRound />} value={segments[0] ? num(segments[0].high_intent_buyers) : "--"} label="high-intent buyers" /><ContextStat icon={<TrendingUp />} value={segments[0] ? money(segments[0].avg_income) : "--"} label="average income" /><ContextStat icon={<Route />} value={segments[0] ? `${segments[0].avg_commute_km.toFixed(1)} km` : "--"} label="average commute" /></div>
      </section>
      <section><SectionTitle title="Priority profiles" /><div className="profile-grid">{customers.slice(0, 4).map((customer, index) => <ProfileCard key={customer.id} customer={customer} index={index} />)}</div></section>
    </div>
  );
}

function ContextStat({ icon, value, label }: { icon: React.ReactNode; value: string; label: string }) { return <div className="context-stat">{icon}<div><strong>{value}</strong><span>{label}</span></div></div>; }

function MarketDnaPage({ drift, quality }: ViewProps) {
  const top = [...drift].sort((a, b) => b.absolute_delta - a.absolute_delta);
  const numericMax = Math.max(...top.filter((x) => x.metric_type.includes("numeric")).map((x) => x.absolute_delta), 1);
  const normalized = top.map((row) => ({ ...row, visual: row.metric_type.includes("numeric") ? Math.min(row.absolute_delta / numericMax, 1) : Math.min(row.absolute_delta / 0.2, 1) }));
  const originalMissing = quality.filter((row) => row.dataset === "original_reference" && row.missing_count > 0).length;
  return <div className="page-stack"><section className="dna-hero"><div><span className="panel-kicker">The headline</span><h2>The current market travels less and charges differently.</h2><p>Commute distance creates the clearest behavioral shift. Charging access and vehicle mix follow behind it.</p></div><div className="dna-stat"><strong>8.95 km</strong><span>lower average commute than the historical reference</span></div><div className="dna-stat"><strong>{originalMissing}</strong><span>original fields with missing values</span></div></section><section className="dna-layout"><div className="drift-panel"><SectionTitle title="Ranked population shifts" action={<span className="section-note">Relative visual scale</span>} /><div className="drift-list">{normalized.slice(0, 9).map((row, index) => <div key={row.column_name}><span>{row.column_name.replaceAll("_", " ")}</span><div><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: row.visual }} transition={{ delay: index * 0.05, duration: 0.6 }} /></div><strong>{row.metric_type.includes("numeric") ? row.absolute_delta.toFixed(2) : pct(row.absolute_delta)}</strong></div>)}</div></div><div className="dna-notes"><SectionTitle title="Read the shift" /><Insight icon={<Route />} title="Shorter daily travel" text="The competition population commutes about nine kilometers less on average." /><Insight icon={<BatteryCharging />} title="Access has moved" text="Home charging and nearby infrastructure differ enough to affect targeting assumptions." /><Insight icon={<GitCompareArrows />} title="Reference, not training fuel" text="The original dataset is retained for comparison after augmentation did not improve validation." /></div></section></div>;
}

function Insight({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) { return <div className="insight"><span>{icon}</span><div><h3>{title}</h3><p>{text}</p></div></div>; }

const defaultProfile: PredictionRequest = { age: 35, annual_income_usd: 90000, daily_commute_km: 25, number_of_cars_owned: 1, charging_stations_near_home: 3, charging_stations_near_work: 4, environmental_concern_level: 7, city_type: "Urban", current_car_type: "Sedan", home_charging_possible: "Yes", subsidy_available: "Yes", range_anxiety_level: "Low" };

function SimulatorPage({ scenarios, carSegments }: ViewProps) {
  const [profile, setProfile] = useState(defaultProfile);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [scoring, setScoring] = useState(false);
  async function score() { setScoring(true); try { setPrediction(await postJson<PredictionResult, PredictionRequest>("/predict", profile)); } finally { setScoring(false); } }
  useEffect(() => { score(); }, []);
  const positive = prediction?.top_positive_factors.filter(Boolean) ?? [];
  const barriers = prediction?.top_barriers.filter(Boolean) ?? [];
  return <div className="page-stack"><section><SectionTitle title="Choose a vehicle context" /><div className="vehicle-grid selector">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} selected={profile.current_car_type === segment.segment} onClick={() => setProfile({ ...profile, current_car_type: segment.segment })} />)}</div></section><section className="simulator-layout"><div className="control-panel"><SectionTitle title="Buyer profile" /><div className="control-grid"><RangeControl label="Age" value={profile.age} min={18} max={80} onChange={(v) => setProfile({ ...profile, age: v })} /><RangeControl label="Annual income" value={profile.annual_income_usd} min={25000} max={220000} step={5000} format={money} onChange={(v) => setProfile({ ...profile, annual_income_usd: v })} /><RangeControl label="Daily commute" value={profile.daily_commute_km} min={0} max={120} format={(v) => `${v} km`} onChange={(v) => setProfile({ ...profile, daily_commute_km: v })} /><RangeControl label="Environmental concern" value={profile.environmental_concern_level} min={1} max={10} onChange={(v) => setProfile({ ...profile, environmental_concern_level: v })} /></div><div className="select-grid"><SelectControl label="City" value={profile.city_type} options={["Urban", "Suburban", "Rural"]} onChange={(v) => setProfile({ ...profile, city_type: v })} /><SelectControl label="Home charging" value={profile.home_charging_possible} options={["Yes", "No"]} onChange={(v) => setProfile({ ...profile, home_charging_possible: v })} /><SelectControl label="Subsidy" value={profile.subsidy_available} options={["Yes", "No"]} onChange={(v) => setProfile({ ...profile, subsidy_available: v })} /><SelectControl label="Range anxiety" value={profile.range_anxiety_level} options={["Low", "Medium", "High"]} onChange={(v) => setProfile({ ...profile, range_anxiety_level: v })} /></div><button className="primary-action" onClick={score} disabled={scoring}>{scoring ? "Scoring profile" : "Score buyer profile"}<ArrowUpRight size={17} /></button></div><div className="prediction-stage"><div className="selected-vehicle">{vehicleImages[profile.current_car_type] && <Image src={vehicleImages[profile.current_car_type]} alt={`Unbranded electric ${profile.current_car_type.toLowerCase()}`} width={720} height={420} />}</div><div className="prediction-gauge" style={{ "--probability": `${(prediction?.ev_purchase_probability ?? 0) * 360}deg` } as React.CSSProperties}><div><strong>{prediction ? pct(prediction.ev_purchase_probability) : "--"}</strong><span>purchase likelihood</span></div></div><div className="prediction-band"><span>Adoption band</span><strong>{prediction?.adoption_band ?? "Waiting"}</strong></div><div className="factor-columns"><FactorGroup title="Positive signals" items={positive} positive /><FactorGroup title="Adoption barriers" items={barriers} /></div></div></section><section><SectionTitle title="Population interventions" /><div className="scenario-grid">{scenarios.map((item, index) => <Reveal className="scenario-card" key={item.scenario} delay={index * 0.05}><span>0{index + 1}</span><h3>{item.scenario}</h3><strong>+{num(item.incremental_expected_buyers)}</strong><p>estimated incremental buyers across {num(item.affected_customers)} affected profiles</p></Reveal>)}</div></section></div>;
}

function RangeControl({ label, value, min, max, step = 1, format = num, onChange }: { label: string; value: number; min: number; max: number; step?: number; format?: (v: number) => string; onChange: (v: number) => void }) { return <label className="range-control"><span><b>{label}</b><strong>{format(value)}</strong></span><input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} /></label>; }
function SelectControl({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) { return <label className="select-control"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>; }
function FactorGroup({ title, items, positive = false }: { title: string; items: Array<string | null>; positive?: boolean }) { const filtered = items.filter(Boolean); return <div className={`factor-group ${positive ? "positive" : ""}`}><span>{title}</span>{filtered.length ? filtered.map((item) => <div key={item}><Check size={14} />{item}</div>) : <p>No strong signals detected.</p>}</div>; }

function ModelPage() {
  const rows = [{ model: "Logistic", auc: 0.938106 }, { model: "XGBoost", auc: 0.941804 }, { model: "LightGBM", auc: 0.941897 }, { model: "Tree blend", auc: 0.942261 }, { model: "TabM", auc: 0.94475 }];
  return <div className="page-stack"><section className="model-hero"><div><span className="panel-kicker">Winning approach</span><h2>TabM rank average</h2><p>A neural tabular ensemble delivered the strongest generalization and the project&apos;s best leaderboard result.</p><div className="model-badges"><span><ShieldCheck /> Strict OOF</span><span><Layers3 /> 5 folds</span><span><Target /> ROC AUC</span></div></div><div className="auc-display"><span>Public leaderboard</span><strong>0.94480</strong><small>Top 300 finish</small></div></section><section className="model-layout"><div className="chart-panel"><SectionTitle title="Performance progression" /><div className="model-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{ left: -18, right: 18, top: 24, bottom: 8 }}><defs><linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#39d7a5" stopOpacity={0.28} /><stop offset="100%" stopColor="#39d7a5" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#1f2a38" vertical={false} /><XAxis dataKey="model" stroke="#718096" tick={{ fontSize: 11 }} /><YAxis domain={[0.937, 0.946]} stroke="#718096" tickFormatter={(v) => Number(v).toFixed(3)} tick={{ fontSize: 11 }} /><Tooltip contentStyle={{ background: "#111923", border: "1px solid #263545", borderRadius: 6 }} formatter={(v) => Number(v).toFixed(6)} /><Area type="monotone" dataKey="auc" stroke="#39d7a5" strokeWidth={3} fill="url(#aucFill)" dot={{ r: 4, fill: "#07100e", strokeWidth: 2 }} /></AreaChart></ResponsiveContainer></div></div><div className="experiment-list"><SectionTitle title="Experiment trail" />{rows.map((row, index) => <div key={row.model} className={index === rows.length - 1 ? "winner" : ""}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{row.model}</strong><small>{index === 0 ? "Explainable baseline" : index === 4 ? "Best neural ensemble" : "Controlled iteration"}</small></div><b>{row.auc.toFixed(5)}</b></div>)}</div></section><section><SectionTitle title="Trust by design" /><div className="governance-flow"><Insight icon={<Network />} title="Honest validation" text="Every training row is scored only by a model that did not train on its target." /><Insight icon={<GitCompareArrows />} title="Original data as context" text="Historical data supports drift analysis after augmentation failed to improve CV." /><Insight icon={<DatabaseZap />} title="Serving boundary" text="The product consumes scored outputs through a warehouse and API layer." /></div></section></div>;
}

function DataQualityPage({ quality }: ViewProps) {
  const datasets = ["competition_train", "competition_test", "original_reference", "scored_current"];
  const summaries = datasets.map((dataset) => { const rows = quality.filter((item) => item.dataset === dataset); return { dataset, fields: rows.length, rows: rows[0]?.row_count ?? 0, missing: rows.filter((item) => item.missing_count > 0).length, maxMissing: Math.max(...rows.map((item) => item.missing_rate), 0) }; });
  const exceptions = quality.filter((item) => item.missing_count > 0).sort((a, b) => b.missing_rate - a.missing_rate).slice(0, 8);
  return <div className="page-stack"><section className="health-grid">{summaries.map((item, index) => <Reveal className="health-card" key={item.dataset} delay={index * 0.05}><div className="health-icon"><ShieldCheck /></div><span>{item.dataset.replaceAll("_", " ")}</span><strong>{item.missing === 0 ? "Ready" : `${item.missing} exceptions`}</strong><p>{num(item.rows)} rows / {item.fields} monitored fields</p><div className="health-line"><i style={{ transform: `scaleX(${1 - item.maxMissing})` }} /></div></Reveal>)}</section><section className="quality-layout"><div className="heatmap-panel"><SectionTitle title="Warehouse coverage" /><div className="quality-heatmap">{quality.slice(0, 48).map((item) => <div key={`${item.dataset}-${item.column_name}`} title={`${item.dataset}: ${item.column_name}, ${pct(item.missing_rate)} missing`} style={{ opacity: Math.max(0.22, 1 - item.missing_rate * 7) }}><span>{item.column_name.replaceAll("_", " ")}</span></div>)}</div></div><div className="exception-panel"><SectionTitle title="Exceptions to review" />{exceptions.length ? exceptions.map((item) => <div key={`${item.dataset}-${item.column_name}`}><span className="warning-mark" /><div><strong>{item.column_name.replaceAll("_", " ")}</strong><small>{item.dataset.replaceAll("_", " ")}</small></div><b>{pct(item.missing_rate)}</b></div>) : <EmptyState message="No missing-value exceptions were found." />}</div></section></div>;
}
