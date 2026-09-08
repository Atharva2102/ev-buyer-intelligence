# AWS and Snowflake Blueprint

This project runs locally with DuckDB, but the pipeline is structured so it can be explained or extended as a cloud analytics platform.

## Cloud Mapping

| Local Layer | AWS / Snowflake Equivalent | Purpose |
|---|---|---|
| Repository CSV files | S3 `raw/` prefix | Immutable source files |
| `etl/build_warehouse.py` | ECS task, Lambda container, or Glue Python shell job | Batch ingestion and validation |
| DuckDB tables | Snowflake staging and mart schemas | SQL analytics layer |
| `warehouse/marts/*.csv` | S3 `curated/` prefix or Snowflake external unloads | Dashboard-ready outputs |
| FastAPI | ECS/Fargate service or Lambda behind API Gateway | Metric and scoring API |
| Next.js frontend | Amplify, S3 + CloudFront, or Vercel | Analytics application |
| Console logs | CloudWatch Logs | Pipeline observability |

## Suggested AWS Layout

```text
s3://ev-buyer-intelligence/raw/
  train.csv
  test.csv
  original_ev_adoption.csv

s3://ev-buyer-intelligence/curated/
  historical_adoption.parquet
  scored_current_customers.parquet
  mart_segment_metrics.parquet
  mart_policy_simulation.parquet
  mart_data_quality.parquet
  mart_drift_metrics.parquet
```

## Batch Flow

1. Upload source files to S3 `raw/`.
2. EventBridge starts a scheduled ETL/scoring task.
3. The task validates schema, normalizes columns, computes derived features, and attaches model probabilities.
4. Curated marts are written to S3 and/or Snowflake.
5. The API reads serving tables and exposes dashboard endpoints.
6. CloudWatch captures row counts, missingness warnings, and run status.

## Snowflake Schemas

```sql
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS serving;
```

Recommended table ownership:

- `raw`: exact landing data.
- `staging`: standardized IDs, target labels, feature columns, and derived features.
- `mart`: analyst-facing business metrics.
- `serving`: model scores consumed by operational applications.

## Portfolio Talking Points

- Local DuckDB keeps the project runnable without paid cloud resources.
- The schema still mirrors enterprise warehouse practices.
- The dashboard consumes marts through an API instead of reading notebook outputs directly.
- The original dataset is treated as historical reference data, supporting drift checks and data-quality monitoring.
