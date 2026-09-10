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
  -> ECS Fargate ETL and validation
  -> S3 Parquet and DuckDB marts
  -> Lambda container FastAPI and LightGBM API
  -> GitHub Pages Next.js application
  -> DynamoDB scoring events and CloudWatch logs
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

Open `http://localhost:3000` for the landing page and
`http://localhost:3000/dashboard` for the analytics workspace.

## Public Deployment

The frontend uses a Next.js static export so the landing page and all dashboard
routes can be published together through GitHub Pages. The FastAPI service runs
as an AWS Lambda container exposed through a Function URL. Each cold start
downloads the latest DuckDB warehouse from S3 into Lambda's temporary storage.

Production routes follow this shape:

```text
/              portfolio landing page
/dashboard     decision overview
/simulator     interactive LightGBM scoring
/model         validation and model evidence
```

The Pages workflow is `.github/workflows/deploy-pages.yml`. Set the repository
variable `NEXT_PUBLIC_API_BASE` to the Lambda Function URL. `NEXT_PUBLIC_SITE_URL`
and `PAGES_BASE_PATH` configure canonical metadata and repository-path hosting.

Deploy and verify the AWS API from `ev_adoption_platform/`:

```powershell
.\cloud\scripts\deploy_api_lambda.ps1
```

This builds and pushes the Lambda-compatible API image, creates the function and
public Function URL, and checks `/health`. The batch schedule remains disabled.
See `cloud/README.md` for IAM, MFA, and App Runner fallback notes.

## Data Sources

- Competition train/test data from the current repository.
- Original reference dataset: `OG Dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv`.
- Model score source: the best available local submission file, preferring TabM outputs if present and falling back to the best local tree submission.

## Portfolio Story

The original dataset is used as a historical survey/reference population, not as a naive training-row boost. Earlier experiments showed direct row augmentation and original-prior target encoding did not improve CV, which is documented as part of the platform's model governance story.
