const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type OverviewMetric = {
  metric_name: string;
  metric_value: number;
  metric_label: string;
};

export type SegmentMetric = {
  segment: string;
  segment_type: string;
  population: number;
  avg_probability: number;
  high_intent_buyers: number;
  avg_income: number;
  avg_commute_km: number;
  lift_vs_average: number;
  rank_in_type: number;
};

export type CustomerScore = {
  id: number;
  Age: number;
  Annual_Income_USD: number;
  Daily_Commute_km: number;
  Current_Car_Type: string;
  City_Type: string;
  Home_Charging_Possible: string;
  Subsidy_Available: string;
  Range_Anxiety_Level: string;
  ev_purchase_probability: number;
  adoption_band: string;
};

export type PolicyScenario = {
  scenario: string;
  affected_customers: number;
  baseline_expected_buyers: number;
  scenario_expected_buyers: number;
  incremental_expected_buyers: number;
  avg_probability_lift: number;
};

export type DriftMetric = {
  column_name: string;
  metric_type: string;
  original_value: number | null;
  competition_value: number | null;
  absolute_delta: number;
};

export type QualityMetric = {
  dataset: string;
  column_name: string;
  row_count: number;
  missing_count: number;
  missing_rate: number;
  unique_count: number;
};

export type LiveFeed = {
  generated_at: string;
  batch_id: string;
  rows: Array<{
    id: number;
    ev_purchase_probability: number;
    adoption_band: string;
    City_Type: string;
    Current_Car_Type: string;
    Range_Anxiety_Level: string;
    score_source: string;
  }>;
};

export type PredictionRequest = {
  age: number;
  annual_income_usd: number;
  daily_commute_km: number;
  number_of_cars_owned: number;
  charging_stations_near_home: number;
  charging_stations_near_work: number;
  environmental_concern_level: number;
  city_type: string;
  current_car_type: string;
  home_charging_possible: string;
  subsidy_available: string;
  range_anxiety_level: string;
};

export type PredictionResult = {
  ev_purchase_probability: number;
  adoption_band: string;
  top_positive_factors: Array<string | null>;
  top_barriers: Array<string | null>;
};

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function postJson<TResponse, TBody>(path: string, body: TBody): Promise<TResponse> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json() as Promise<TResponse>;
}
