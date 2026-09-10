"use client";

import {
  Activity, ArrowUpRight, BatteryCharging, Check, CircleGauge, Clock3, DatabaseZap,
  Gauge, GitCompareArrows, Info, Layers3, Network, Radio, Route, ShieldCheck, Target,
  TrendingUp, UsersRound, Zap
} from "lucide-react";
import { useState } from "react";
import { motion } from "motion/react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PageKey } from "./app-shell";
import { EmptyState, MetricTile, Reveal, SectionTitle, Skeleton, VehicleCard } from "./visuals";
import {
  CustomerScore, DriftMetric, OverviewMetric, PolicyScenario, PredictionRequest,
  PredictionResult, QualityMetric, ScoringOperations, SegmentMetric, postJson
} from "../lib/api";

type ViewProps = {
  page: PageKey;
  loading: boolean;
  metricMap: Record<string, OverviewMetric>;
  operations: ScoringOperations | null;
  demoEnabled: boolean;
  segments: SegmentMetric[];
  carSegments: SegmentMetric[];
  customers: CustomerScore[];
  scenarios: PolicyScenario[];
  drift: DriftMetric[];
  quality: QualityMetric[];
  activeSegment: string;
  onSegmentChange: (value: string) => void;
  onDemoToggle: () => void;
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

function PageGuide({ question, summary, read }: { question: string; summary: string; read: string }) {
  return (
    <Reveal className="page-guide">
      <div className="guide-icon"><Info /></div>
      <div><span>Question this page answers</span><h2>{question}</h2></div>
      <p>{summary}</p>
      <div className="guide-read"><span>How to read it</span><p>{read}</p></div>
    </Reveal>
  );
}

function OverviewPage({ loading, metricMap, operations, scenarios, segments, carSegments }: ViewProps) {
  const top = segments[0];
  return (
    <div className="page-stack">
      <PageGuide question="Where should a strategy team investigate first?" summary="This overview combines saved model scores across 286,571 synthetic buyer profiles. It sizes the audience, summarizes predicted intent, and compares four directional intervention scenarios." read="Use the KPIs for scale, then compare interventions by estimated additional high-intent buyers. These are modeled scenarios, not proof that a policy will cause purchases." />
      <section className="metrics-grid">
        <MetricTile label="Profiles scored" value={metricMap.customers_scored?.metric_label ?? "--"} detail="Synthetic scoring population" progress={0.86} icon={<DatabaseZap />} />
        <MetricTile label="Average intent" value={metricMap.avg_ev_purchase_probability?.metric_label ?? "--"} detail="Across all scored profiles" progress={0.5} icon={<CircleGauge />} delay={0.05} />
        <MetricTile label="High-intent profiles" value={metricMap.high_intent_buyers?.metric_label ?? "--"} detail="Priority and high bands" progress={0.65} icon={<Target />} delay={0.1} />
        <MetricTile label="Historical adoption" value={metricMap.historical_adoption_rate?.metric_label ?? "--"} detail="Original reference cohort" progress={0.175} icon={<TrendingUp />} delay={0.15} />
      </section>

      <section className="overview-focus">
        <Reveal className="decision-brief">
          <span className="panel-kicker">Recommended focus</span>
          <h2>Prioritize incentives and charging access for further testing.</h2>
          <p>In this scenario model, offering subsidies reaches the largest estimated opportunity. Charging access ranks next. The result suggests where to investigate first, not a guaranteed outcome.</p>
          <div className="decision-meta">
            <div><strong>{scenarios[0] ? num(scenarios[0].incremental_expected_buyers) : "--"}</strong><span>profiles with modeled intent lift</span></div>
            <div><strong>{top ? pct(top.avg_probability) : "--"}</strong><span>top segment intent</span></div>
          </div>
        </Reveal>
        <PolicyOpportunity scenarios={scenarios} />
      </section>

      <section>
        <SectionTitle title="Intent by vehicle type" action={<span className="section-note">Scored synthetic profiles</span>} />
        <p className="panel-explainer section-copy">Compare current vehicle classes by average model score and the number of profiles above the high-intent threshold.</p>
        {loading ? <Skeleton rows={4} /> : <div className="vehicle-grid">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} />)}</div>}
      </section>

      <section className="overview-lower">
        <OperationsFeed operations={operations} compact />
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
      <p className="panel-explainer">Each bar estimates how many additional profiles could cross into stronger purchase intent after a fixed probability adjustment. Longer bars indicate larger modeled reach.</p>
      <div className="policy-bars">{scenarios.map((item, index) => <div className="policy-row" key={item.scenario}><span>{item.scenario.replace(" for customers", "")}</span><div><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: item.incremental_expected_buyers / max }} transition={{ duration: 0.7, delay: 0.12 + index * 0.08 }} /><strong>+{num(item.incremental_expected_buyers)}</strong></div></div>)}</div>
    </Reveal>
  );
}

function LiveScoringPage({ operations, demoEnabled, onDemoToggle }: ViewProps) {
  const total = operations?.total_requests ?? 0;
  const failureProgress = total ? 1 - (operations?.failed_requests ?? 0) / total : 1;
  const streamHeadline = demoEnabled
    ? "Synthetic arrivals are being scored"
    : total
      ? "Monitoring recorded scoring activity"
      : "Waiting for the first scoring event";
  return (
    <div className="page-stack">
      <PageGuide question="Is the scoring service working, and what is it processing?" summary="This is an operations view of API requests, not a claim that real customers are arriving live. Simulator actions are user-triggered; the optional demo stream samples synthetic profiles and labels them separately." read="Watch success rate and latency for service health. Use event source and prediction mix to understand traffic composition. Metrics cover the selected 60-minute window." />
      <section className="ops-strip">
        <MetricTile label="Requests processed" value={num(total)} detail="Last 60 minutes" progress={Math.min(total / 60, 1)} icon={<Activity />} />
        <MetricTile label="Request rate" value={`${(operations?.requests_per_minute ?? 0).toFixed(1)}/min`} detail="Measured event throughput" progress={Math.min((operations?.requests_per_minute ?? 0) / 10, 1)} icon={<Radio />} delay={0.05} />
        <MetricTile label="Average latency" value={`${(operations?.avg_latency_ms ?? 0).toFixed(1)} ms`} detail={`p95 ${(operations?.p95_latency_ms ?? 0).toFixed(1)} ms`} progress={Math.max(0.08, 1 - (operations?.avg_latency_ms ?? 0) / 500)} icon={<Clock3 />} delay={0.1} />
        <MetricTile label="Successful requests" value={pct(operations?.success_rate ?? 1, 1)} detail={`${operations?.failed_requests ?? 0} failed events`} progress={failureProgress} icon={<Zap />} delay={0.15} />
      </section>
      <section className="stream-console">
        <div><span className="panel-kicker">Event source</span><h2>{streamHeadline}</h2><p>Simulator submissions are actual API events created by dashboard users. Demo arrivals sample synthetic profiles and are labeled throughout the system.</p></div>
        <button className={`stream-toggle ${demoEnabled ? "active" : ""}`} onClick={onDemoToggle} role="switch" aria-checked={demoEnabled}><span><i /></span><b>{demoEnabled ? "Stop demo stream" : "Start demo stream"}</b></button>
        <div className="runtime-stamp"><DatabaseZap /><span>{operations?.storage_backend === "dynamodb" ? "DynamoDB event store" : "Local SQLite event store"}</span></div>
      </section>
      <section className="live-layout"><OperationsFeed operations={operations} /><BandDistribution operations={operations} /></section>
      <section className="ops-detail-grid">
        <div className="runtime-panel"><SectionTitle title="Serving model" /><div className="runtime-model"><span>Active artifact</span><strong>{operations?.model_version ?? "Waiting for API"}</strong><p>{operations?.model_kind === "lightgbm" ? "A trained LightGBM artifact produces each recorded probability." : "The API is using its explainable fallback scorer."}</p></div></div>
        <div className="source-panel"><SectionTitle title="Event provenance" /><div className="source-split"><div><span>Simulator</span><strong>{operations?.source_counts.simulator ?? 0}</strong><i style={{ transform: `scaleX(${total ? (operations?.source_counts.simulator ?? 0) / total : 0})` }} /></div><div><span>Demo stream</span><strong>{operations?.source_counts.demo_stream ?? 0}</strong><i style={{ transform: `scaleX(${total ? (operations?.source_counts.demo_stream ?? 0) / total : 0})` }} /></div></div></div>
      </section>
    </div>
  );
}

function OperationsFeed({ operations, compact = false }: { operations: ScoringOperations | null; compact?: boolean }) {
  const events = operations?.recent_events ?? [];
  return (
    <div className={`live-queue ${compact ? "compact" : ""}`}>
      <SectionTitle title={compact ? "Recent scoring activity" : "Recorded scoring events"} action={<span className="live-badge"><i /> Updates every 5s</span>} />
      <div className="batch-meta"><span>Last {operations?.window_minutes ?? 60} minutes</span><span>{operations ? new Date(operations.generated_at).toLocaleTimeString() : "--"}</span></div>
      {events.length ? <div className="feed-list">{events.slice(0, compact ? 5 : 8).map((event, index) => <motion.div key={event.event_id} initial={{ opacity: 0, y: -7 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.025 }}><span className={`band-mark ${(event.adoption_band ?? "failed").toLowerCase()}`} /><div><strong>Event {event.event_id.slice(0, 8)}</strong><span>{event.source === "demo_stream" ? "Demo" : "Simulator"} / {event.city_type ?? "Unknown"} / {event.current_car_type ?? "Unknown"} / {event.latency_ms.toFixed(1)} ms</span></div><b>{event.probability === null ? "Failed" : pct(event.probability)}</b></motion.div>)}</div> : <EmptyState message="No scoring events yet. Score a buyer or start the labeled demo stream." />}
    </div>
  );
}

function BandDistribution({ operations }: { operations: ScoringOperations | null }) {
  const counts = ["Priority", "High", "Watch", "Low"].map((band) => ({ band, count: operations?.band_distribution[band] ?? 0 }));
  const successful = operations?.successful_requests ?? 0;
  const strong = successful ? (counts[0].count + counts[1].count) / successful : 0;
  return <div className="band-panel"><SectionTitle title="Prediction mix" /><div className="donut-wrap"><div className="score-donut" style={{ "--score": `${strong * 360}deg` } as React.CSSProperties}><strong>{pct(strong, 0)}</strong><span>high intent</span></div></div><div className="band-legend">{counts.map((item) => <div key={item.band}><i className={item.band.toLowerCase()} /><span>{item.band}</span><strong>{item.count}</strong></div>)}</div></div>;
}

function ProfileCard({ customer, index }: { customer: CustomerScore; index: number }) {
  return <Reveal className="profile-card" delay={index * 0.04}><div className="profile-card-top"><span>#{customer.id}</span><strong>{pct(customer.ev_purchase_probability)}</strong></div><div className="profile-avatar">{customer.City_Type.charAt(0)}{customer.Current_Car_Type.charAt(0)}</div><h3>{customer.City_Type} {customer.Current_Car_Type} owner</h3><p>{money(customer.Annual_Income_USD)} income / {customer.Daily_Commute_km.toFixed(0)} km commute</p><div className="profile-signals"><span>{customer.Home_Charging_Possible === "Yes" ? "Home charging" : "No home charging"}</span><span>{customer.Subsidy_Available === "Yes" ? "Subsidy" : "No subsidy"}</span></div></Reveal>;
}

function SegmentsPage({ segments, carSegments, activeSegment, onSegmentChange, customers }: ViewProps) {
  const labels: Record<string, string> = { buyer_segment: "Buyer profiles", city_home_charging_segment: "City and charging", subsidy_anxiety_segment: "Subsidy and anxiety" };
  return (
    <div className="page-stack">
      <PageGuide question="Which groups have the highest modeled EV purchase intent?" summary="Profiles are grouped by vehicle class, city and charging access, or subsidy and range anxiety. Rankings help a market team find concentrated opportunities for deeper research." read="Purchase intent is the group average model probability. Lift compares that average with the full scored population. A high rank describes association, not causation." />
      <section><SectionTitle title="Vehicle-class pulse" action={<span className="section-note">Current scoring population</span>} /><div className="vehicle-grid">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} />)}</div></section>
      <section className="segment-layout">
        <div className="ranking-panel"><div className="segment-tabs">{Object.entries(labels).map(([value, label]) => <button key={value} className={activeSegment === value ? "active" : ""} onClick={() => onSegmentChange(value)}>{label}</button>)}</div><SectionTitle title="Top opportunities" /><p className="panel-explainer">Sorted by average predicted probability within the selected grouping.</p>{segments.slice(0, 5).map((segment) => <div className="rank-row" key={segment.segment}><span className="rank-number">{String(segment.rank_in_type).padStart(2, "0")}</span><div><strong>{segment.segment}</strong><span>{num(segment.population)} profiles</span></div><div className="rank-score"><strong>{pct(segment.avg_probability)}</strong><span>{pct(segment.lift_vs_average)} lift</span></div></div>)}</div>
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
  return <div className="page-stack"><PageGuide question="How does the scoring population differ from the original reference data?" summary="Market DNA compares feature distributions between the synthetic competition data and the original 10,000-row EV adoption dataset that inspired it." read="Longer bars mean a larger difference between datasets, but numeric and categorical rows use different units. Drift warns that assumptions may not transfer; it does not say which dataset is better." /><section className="dna-hero"><div><span className="panel-kicker">The clearest difference</span><h2>The scoring population travels less and has a different charging profile.</h2><p>Commute distance shows the largest numeric shift. Charging access and vehicle mix also differ enough to matter when interpreting model outputs.</p></div><div className="dna-stat"><strong>8.95 km</strong><span>lower average commute than the historical reference</span></div><div className="dna-stat"><strong>{originalMissing}</strong><span>original fields with missing values</span></div></section><section className="dna-layout"><div className="drift-panel"><SectionTitle title="Ranked population shifts" action={<span className="section-note">Relative visual scale</span>} /><p className="panel-explainer">Numeric rows show absolute differences in feature means. Categorical rows show distribution distance as a percentage.</p><div className="drift-list">{normalized.slice(0, 9).map((row, index) => <div key={row.column_name}><span>{row.column_name.replaceAll("_", " ")}</span><div><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: row.visual }} transition={{ delay: index * 0.05, duration: 0.6 }} /></div><strong>{row.metric_type.includes("numeric") ? row.absolute_delta.toFixed(2) : pct(row.absolute_delta)}</strong></div>)}</div></div><div className="dna-notes"><SectionTitle title="Read the shift" /><Insight icon={<Route />} title="Shorter daily travel" text="The competition population commutes about nine kilometers less on average." /><Insight icon={<BatteryCharging />} title="Access has moved" text="Home charging and nearby infrastructure differ enough to affect targeting assumptions." /><Insight icon={<GitCompareArrows />} title="Reference, not training fuel" text="The original dataset is retained for comparison after augmentation did not improve validation." /></div></section></div>;
}

function Insight({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) { return <div className="insight"><span>{icon}</span><div><h3>{title}</h3><p>{text}</p></div></div>; }

const defaultProfile: PredictionRequest = { age: 35, annual_income_usd: 90000, daily_commute_km: 25, number_of_cars_owned: 1, charging_stations_near_home: 3, charging_stations_near_work: 4, environmental_concern_level: 7, gender: "Other", city_type: "Urban", current_car_type: "Sedan", home_charging_possible: "Yes", subsidy_available: "Yes", range_anxiety_level: "Low" };

function SimulatorPage({ scenarios, carSegments }: ViewProps) {
  const [profile, setProfile] = useState(defaultProfile);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  async function score() { setScoring(true); setScoreError(null); try { setPrediction(await postJson<PredictionResult, PredictionRequest>("/predict", profile)); } catch { setScoreError("The scoring service could not process this profile."); } finally { setScoring(false); } }
  const positive = prediction?.top_positive_factors.filter(Boolean) ?? [];
  const barriers = prediction?.top_barriers.filter(Boolean) ?? [];
  return <div className="page-stack"><PageGuide question="How would the serving model score one hypothetical buyer profile?" summary="Adjust the profile, choose a current vehicle class, and submit it to the FastAPI scoring endpoint. A trained LightGBM model returns a probability and stores the request as an operations event." read="The result is a model estimate, not an individual recommendation. Signal labels summarize selected profile inputs and are not SHAP explanations." /><section><SectionTitle title="Choose a vehicle context" /><p className="panel-explainer section-copy">Vehicle class is one input to the model. The percentages on these cards summarize the full scored population, not the profile below.</p><div className="vehicle-grid selector">{carSegments.map((segment) => <VehicleCard key={segment.segment} segment={segment} selected={profile.current_car_type === segment.segment} onClick={() => setProfile({ ...profile, current_car_type: segment.segment })} />)}</div></section><section className="simulator-layout"><div className="control-panel"><SectionTitle title="Buyer profile" /><div className="control-grid"><RangeControl label="Age" value={profile.age} min={18} max={80} onChange={(v) => setProfile({ ...profile, age: v })} /><RangeControl label="Annual income" value={profile.annual_income_usd} min={25000} max={220000} step={5000} format={money} onChange={(v) => setProfile({ ...profile, annual_income_usd: v })} /><RangeControl label="Daily commute" value={profile.daily_commute_km} min={0} max={120} format={(v) => `${v} km`} onChange={(v) => setProfile({ ...profile, daily_commute_km: v })} /><RangeControl label="Environmental concern" value={profile.environmental_concern_level} min={1} max={10} onChange={(v) => setProfile({ ...profile, environmental_concern_level: v })} /></div><div className="select-grid"><SelectControl label="Gender" value={profile.gender} options={["Female", "Male", "Other"]} onChange={(v) => setProfile({ ...profile, gender: v })} /><SelectControl label="City" value={profile.city_type} options={["Urban", "Suburban", "Rural"]} onChange={(v) => setProfile({ ...profile, city_type: v })} /><SelectControl label="Home charging" value={profile.home_charging_possible} options={["Yes", "No"]} onChange={(v) => setProfile({ ...profile, home_charging_possible: v })} /><SelectControl label="Subsidy" value={profile.subsidy_available} options={["Yes", "No"]} onChange={(v) => setProfile({ ...profile, subsidy_available: v })} /><SelectControl label="Range anxiety" value={profile.range_anxiety_level} options={["Low", "Medium", "High"]} onChange={(v) => setProfile({ ...profile, range_anxiety_level: v })} /></div><button className="primary-action" onClick={score} disabled={scoring}>{scoring ? "Scoring profile" : "Score and record event"}<ArrowUpRight size={17} /></button>{scoreError && <p className="score-error">{scoreError}</p>}<p className="event-note">Each submitted score becomes a timestamped operations event.</p></div><div className={`prediction-stage ${prediction ? "has-prediction" : ""}`}><motion.div className="vehicle-context-pill" key={profile.current_car_type} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}><span>Selected vehicle context</span><strong>{profile.current_car_type}</strong></motion.div><div className="prediction-gauge" style={{ "--probability": `${(prediction?.ev_purchase_probability ?? 0) * 360}deg` } as React.CSSProperties}><div><strong>{prediction ? pct(prediction.ev_purchase_probability) : "--"}</strong><span>purchase likelihood</span></div></div><div className="prediction-band"><span>Model result</span><strong>{prediction?.adoption_band ?? "Submit a profile"}</strong>{prediction && <small>{prediction.model_kind === "lightgbm" ? "LightGBM" : "Fallback"} / {prediction.latency_ms.toFixed(1)} ms / event {prediction.event_id.slice(0, 8)}</small>}</div><div className="factor-columns"><FactorGroup title="Positive profile signals" items={positive} positive /><FactorGroup title="Potential barriers" items={barriers} /></div></div></section><section><SectionTitle title="Population-level scenarios" /><p className="panel-explainer section-copy">Unlike the single-profile score above, these cards apply fixed directional assumptions across eligible groups in the full population. They are planning prompts, not causal forecasts.</p><div className="scenario-grid">{scenarios.map((item, index) => <Reveal className="scenario-card" key={item.scenario} delay={index * 0.05}><span>0{index + 1}</span><h3>{item.scenario}</h3><strong>+{num(item.incremental_expected_buyers)}</strong><p>estimated incremental buyers across {num(item.affected_customers)} affected profiles</p></Reveal>)}</div></section></div>;
}

function RangeControl({ label, value, min, max, step = 1, format = num, onChange }: { label: string; value: number; min: number; max: number; step?: number; format?: (v: number) => string; onChange: (v: number) => void }) { return <label className="range-control"><span><b>{label}</b><strong>{format(value)}</strong></span><input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} /></label>; }
function SelectControl({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) { return <label className="select-control"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>; }
function FactorGroup({ title, items, positive = false }: { title: string; items: Array<string | null>; positive?: boolean }) { const filtered = items.filter(Boolean); return <div className={`factor-group ${positive ? "positive" : ""}`}><span>{title}</span>{filtered.length ? filtered.map((item) => <div key={item}><Check size={14} />{item}</div>) : <p>No strong signals detected.</p>}</div>; }

function ModelPage() {
  const rows = [{ model: "Logistic", auc: 0.938106 }, { model: "XGBoost", auc: 0.941804 }, { model: "LightGBM", auc: 0.941897 }, { model: "Tree blend", auc: 0.942261 }, { model: "TabM", auc: 0.94475 }];
  return <div className="page-stack"><PageGuide question="Which modeling approach performed best, and can we trust the comparison?" summary="This page traces the project from a logistic baseline through boosted trees and into TabM, the neural tabular model that produced the strongest public leaderboard result." read="ROC AUC measures ranking quality: higher is better, and 0.5 is random. OOF means every training prediction came from a fold that did not train on that row. The interactive API uses a separate LightGBM serving artifact." /><section className="model-hero"><div><span className="panel-kicker">Winning competition approach</span><h2>TabM rank average</h2><p>A neural tabular ensemble delivered the project&apos;s strongest public leaderboard result. It ranks likely buyers well; it does not prove purchase causality.</p><div className="model-badges"><span><ShieldCheck /> Strict OOF</span><span><Layers3 /> 5 folds</span><span><Target /> ROC AUC</span></div></div><div className="auc-display"><span>Public leaderboard</span><strong>0.94480</strong><small>Top 300 finish</small></div></section><section className="model-layout"><div className="chart-panel"><SectionTitle title="Performance progression" /><p className="panel-explainer">Each point is a controlled model iteration. Small AUC gains matter because the models already rank most positive cases correctly.</p><div className="model-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{ left: -18, right: 18, top: 24, bottom: 8 }}><defs><linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#39d7a5" stopOpacity={0.28} /><stop offset="100%" stopColor="#39d7a5" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#1f2a38" vertical={false} /><XAxis dataKey="model" stroke="#718096" tick={{ fontSize: 11 }} /><YAxis domain={[0.937, 0.946]} stroke="#718096" tickFormatter={(v) => Number(v).toFixed(3)} tick={{ fontSize: 11 }} /><Tooltip contentStyle={{ background: "#111923", border: "1px solid #263545", borderRadius: 6 }} formatter={(v) => Number(v).toFixed(6)} /><Area type="monotone" dataKey="auc" stroke="#39d7a5" strokeWidth={3} fill="url(#aucFill)" dot={{ r: 4, fill: "#07100e", strokeWidth: 2 }} /></AreaChart></ResponsiveContainer></div></div><div className="experiment-list"><SectionTitle title="Experiment trail" />{rows.map((row, index) => <div key={row.model} className={index === rows.length - 1 ? "winner" : ""}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{row.model}</strong><small>{index === 0 ? "Explainable baseline" : index === 4 ? "Best neural ensemble" : "Controlled iteration"}</small></div><b>{row.auc.toFixed(5)}</b></div>)}</div></section><section><SectionTitle title="Trust by design" /><div className="governance-flow"><Insight icon={<Network />} title="Honest validation" text="Every training row is scored only by a model that did not train on its target." /><Insight icon={<GitCompareArrows />} title="Original data as context" text="Historical data supports drift analysis after augmentation failed to improve CV." /><Insight icon={<DatabaseZap />} title="Serving boundary" text="The dashboard uses saved TabM scores; interactive requests use a versioned LightGBM model." /></div></section></div>;
}

function DataQualityPage({ quality }: ViewProps) {
  const datasets = ["competition_train", "competition_test", "original_reference", "scored_current"];
  const summaries = datasets.map((dataset) => { const rows = quality.filter((item) => item.dataset === dataset); return { dataset, fields: rows.length, rows: rows[0]?.row_count ?? 0, missing: rows.filter((item) => item.missing_count > 0).length, maxMissing: Math.max(...rows.map((item) => item.missing_rate), 0) }; });
  const exceptions = quality.filter((item) => item.missing_count > 0).sort((a, b) => b.missing_rate - a.missing_rate).slice(0, 8);
  return <div className="page-stack"><PageGuide question="Is the warehouse complete enough to support the dashboard?" summary="Data health checks row counts, field coverage, missing values, and cardinality across the competition data, original reference, and scored customer table." read="A darker heatmap cell means more missing data. Exceptions deserve review, but missingness is not automatically an error: the original source contains expected numeric gaps that LightGBM can handle." /><section className="health-grid">{summaries.map((item, index) => <Reveal className="health-card" key={item.dataset} delay={index * 0.05}><div className="health-icon"><ShieldCheck /></div><span>{item.dataset.replaceAll("_", " ")}</span><strong>{item.missing === 0 ? "Ready" : `${item.missing} exceptions`}</strong><p>{num(item.rows)} rows / {item.fields} monitored fields</p><div className="health-line"><i style={{ transform: `scaleX(${1 - item.maxMissing})` }} /></div></Reveal>)}</section><section className="quality-layout"><div className="heatmap-panel"><SectionTitle title="Warehouse coverage" /><p className="panel-explainer">Each cell represents one monitored field. Hover to see its dataset and missing-value rate.</p><div className="quality-heatmap">{quality.slice(0, 48).map((item) => <div key={`${item.dataset}-${item.column_name}`} title={`${item.dataset}: ${item.column_name}, ${pct(item.missing_rate)} missing`} style={{ opacity: Math.max(0.22, 1 - item.missing_rate * 7) }}><span>{item.column_name.replaceAll("_", " ")}</span></div>)}</div></div><div className="exception-panel"><SectionTitle title="Exceptions to review" />{exceptions.length ? exceptions.map((item) => <div key={`${item.dataset}-${item.column_name}`}><span className="warning-mark" /><div><strong>{item.column_name.replaceAll("_", " ")}</strong><small>{item.dataset.replaceAll("_", " ")}</small></div><b>{pct(item.missing_rate)}</b></div>) : <EmptyState message="No missing-value exceptions were found." />}</div></section></div>;
}
