"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell, PageKey } from "./app-shell";
import { DashboardView } from "./dashboard-pages";
import {
  CustomerScore,
  DriftMetric,
  OverviewMetric,
  PolicyScenario,
  QualityMetric,
  ScoringOperations,
  SegmentMetric,
  getJson,
  postJson
} from "../lib/api";

export default function Dashboard({ page = "overview" }: { page?: PageKey }) {
  const [overview, setOverview] = useState<OverviewMetric[]>([]);
  const [segments, setSegments] = useState<SegmentMetric[]>([]);
  const [carSegments, setCarSegments] = useState<SegmentMetric[]>([]);
  const [customers, setCustomers] = useState<CustomerScore[]>([]);
  const [scenarios, setScenarios] = useState<PolicyScenario[]>([]);
  const [drift, setDrift] = useState<DriftMetric[]>([]);
  const [quality, setQuality] = useState<QualityMetric[]>([]);
  const [operations, setOperations] = useState<ScoringOperations | null>(null);
  const [demoEnabled, setDemoEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeSegment, setActiveSegment] = useState("buyer_segment");

  async function loadStaticData(segmentType = activeSegment) {
    try {
      const data = await Promise.all([
        getJson<OverviewMetric[]>("/metrics/overview"),
        getJson<SegmentMetric[]>(`/segments?segment_type=${segmentType}&limit=12`),
        getJson<SegmentMetric[]>("/segments?segment_type=Current_Car_Type&limit=4"),
        getJson<CustomerScore[]>("/customers?limit=14"),
        getJson<PolicyScenario[]>("/policy-simulation"),
        getJson<DriftMetric[]>("/drift"),
        getJson<QualityMetric[]>("/data-quality")
      ]);
      setOverview(data[0]);
      setSegments(data[1]);
      setCarSegments(data[2]);
      setCustomers(data[3]);
      setScenarios(data[4]);
      setDrift(data[5]);
      setQuality(data[6]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The analytics API is unavailable.");
    } finally {
      setLoading(false);
    }
  }

  async function loadOperations() {
    try {
      setOperations(await getJson<ScoringOperations>("/scoring-operations?limit=12&window_minutes=60"));
    } catch {
      setOperations(null);
    }
  }

  useEffect(() => {
    loadStaticData();
    loadOperations();
    const timer = window.setInterval(loadOperations, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!demoEnabled) return;
    let active = true;
    async function emitDemoEvent() {
      try {
        await postJson("/demo-score", {});
        if (active) await loadOperations();
      } catch {
        if (active) setDemoEnabled(false);
      }
    }
    emitDemoEvent();
    const timer = window.setInterval(emitDemoEvent, 2500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [demoEnabled]);

  const metricMap = useMemo(
    () => Object.fromEntries(overview.map((item) => [item.metric_name, item])),
    [overview]
  );

  function refresh() {
    setLoading(true);
    loadStaticData();
    loadOperations();
  }

  function changeSegment(value: string) {
    setActiveSegment(value);
    loadStaticData(value);
  }

  return (
    <AppShell page={page} onRefresh={refresh}>
      {error && (
        <div className="api-error">
          <strong>Dashboard data could not load.</strong>
          <span>{error} Start the ETL and API from ev_adoption_platform.</span>
        </div>
      )}
      <DashboardView
        page={page}
        loading={loading}
        metricMap={metricMap}
        operations={operations}
        demoEnabled={demoEnabled}
        segments={segments}
        carSegments={carSegments}
        customers={customers}
        scenarios={scenarios}
        drift={drift}
        quality={quality}
        activeSegment={activeSegment}
        onSegmentChange={changeSegment}
        onDemoToggle={() => setDemoEnabled((enabled) => !enabled)}
      />
    </AppShell>
  );
}
