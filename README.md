# EV Buyer Intelligence

An end-to-end analytics portfolio project that turns EV adoption data into buyer scoring, market intelligence, and policy scenario planning.

## Platform

- **Data engineering:** reproducible Python ETL into DuckDB analytics marts
- **API:** FastAPI endpoints for metrics, segments, scoring, drift, and data quality
- **Machine learning:** competition-tested tree ensembles and TabM experiments
- **Product analytics:** a responsive Next.js dashboard with Recharts and Motion
- **Cloud design:** AWS ingestion and Snowflake warehouse architecture documentation

The active application lives in [`ev_adoption_platform/`](ev_adoption_platform/). The modeling history, experiment scripts, and project write-up live in [`kaggle_competition_workspace/`](kaggle_competition_workspace/).

## Run Locally

Build the warehouse and start the API:

```powershell
cd ev_adoption_platform
pip install -r requirements.txt
python etl\build_warehouse.py
uvicorn api.main:app --reload --port 8000
```

Start the frontend in a second terminal:

```powershell
cd ev_adoption_platform\frontend
npm install
npm run dev
```

Open `http://localhost:3000/home` for the product site or `http://localhost:3000` for the analytics workspace.

Raw datasets and generated warehouse files are intentionally excluded from version control. See [`ev_adoption_platform/README.md`](ev_adoption_platform/README.md) for expected data locations and architecture details.
