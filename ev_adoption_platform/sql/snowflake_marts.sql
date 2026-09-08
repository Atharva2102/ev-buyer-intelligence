CREATE OR REPLACE VIEW mart.overview_metrics AS
SELECT
    'customers_scored' AS metric_name,
    COUNT(*)::FLOAT AS metric_value,
    TO_VARCHAR(COUNT(*)) AS metric_label
FROM serving.scored_current_customers
UNION ALL
SELECT
    'avg_ev_purchase_probability',
    AVG(ev_purchase_probability),
    TO_VARCHAR(ROUND(AVG(ev_purchase_probability) * 100, 2)) || '%'
FROM serving.scored_current_customers
UNION ALL
SELECT
    'high_intent_buyers',
    SUM(IFF(ev_purchase_probability >= 0.35, 1, 0))::FLOAT,
    TO_VARCHAR(SUM(IFF(ev_purchase_probability >= 0.35, 1, 0)))
FROM serving.scored_current_customers
UNION ALL
SELECT
    'historical_adoption_rate',
    AVG(will_buy_ev),
    TO_VARCHAR(ROUND(AVG(will_buy_ev) * 100, 2)) || '%'
FROM staging.historical_adoption;

CREATE OR REPLACE VIEW mart.segment_metrics AS
SELECT
    buyer_segment AS segment,
    'buyer_segment' AS segment_type,
    COUNT(*) AS population,
    AVG(ev_purchase_probability) AS avg_probability,
    SUM(IFF(ev_purchase_probability >= 0.35, 1, 0)) AS high_intent_buyers,
    AVG(annual_income_usd) AS avg_income,
    AVG(daily_commute_km) AS avg_commute_km,
    AVG(ev_purchase_probability)
        - (SELECT AVG(ev_purchase_probability) FROM serving.scored_current_customers)
        AS lift_vs_average
FROM serving.scored_current_customers
GROUP BY buyer_segment;

CREATE OR REPLACE VIEW mart.data_quality AS
SELECT
    dataset,
    column_name,
    row_count,
    missing_count,
    missing_count / NULLIF(row_count, 0) AS missing_rate,
    unique_count
FROM staging.column_quality_checks;

CREATE OR REPLACE VIEW mart.policy_simulation AS
SELECT
    scenario,
    affected_customers,
    baseline_expected_buyers,
    scenario_expected_buyers,
    scenario_expected_buyers - baseline_expected_buyers AS incremental_expected_buyers,
    avg_probability_lift
FROM serving.policy_simulation_results;
