# EV Buyer Intelligence Platform

An end-to-end portfolio data product built from the EV purchase prediction work. It turns the Kaggle competition assets and the original EV adoption dataset into a realistic analytics platform for an EV manufacturer's growth and strategy team.

## What It Demonstrates

- Data engineering: raw ingestion, schema normalization, validation, warehouse marts, and scored datasets.
- Analytics: segment KPIs, distribution drift, adoption barriers, and reusable business metrics.
- Machine learning: model score consumption, probability bands, scenario simulation, and model monitoring surfaces.
- Dashboard engineering: custom Next.js dashboard instead of Streamlit, with dense app-style navigation and live metric panels.

## Architecture

```text
Raw CSV files
  -> Python ETL and validation
  -> DuckDB warehouse marts
  -> FastAPI analytics API
  -> Next.js dashboard
```

The local DuckDB warehouse is intentionally modeled like a Snowflake-style analytics layer. The same zones can later map to AWS:

- `raw`: S3 raw CSV bucket
- `staging`: cleaned and standardized Parquet/Snowflake tables
- `mart`: dashboard-ready warehouse views
- `serving`: scored customer table consumed by API/dashboard

See `infra/aws_snowflake_blueprint.md` and `sql/snowflake_marts.sql` for the cloud/warehouse version of the design.

## Quick Start

From the repository root:

```powershell
cd ev_adoption_platform
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python etl\\build_warehouse.py
uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd ev_adoption_platform\\frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Data Sources

- Competition train/test data from the current repository.
- Original reference dataset: `OG Dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv`.
- Model score source: the best available local submission file, preferring TabM outputs if present and falling back to the best local tree submission.

## Portfolio Story

The original dataset is used as a historical survey/reference population, not as a naive training-row boost. Earlier experiments showed direct row augmentation and original-prior target encoding did not improve CV, which is documented as part of the platform's model governance story.
