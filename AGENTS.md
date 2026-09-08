# Repository Guidelines

## Project Structure & Module Organization

Whatever task you can do yourself please do yourself this inclues staring apps and verifying them.

This repository has two top-level areas:

- `ev_adoption_platform/`: active portfolio application.
- `kaggle_competition_workspace/`: archived Kaggle experiments, submissions, notebooks, and source data.

Inside `ev_adoption_platform/`:

- `etl/build_warehouse.py` builds the local DuckDB warehouse and CSV marts.
- `api/main.py` exposes FastAPI endpoints for dashboard metrics and prediction simulation.
- `frontend/app/` contains Next.js route pages.
- `frontend/components/` contains reusable React dashboard components.
- `frontend/lib/` contains API client types and helpers.
- `infra/` and `sql/` document the AWS/Snowflake-style architecture.

Generated warehouse outputs live under `ev_adoption_platform/warehouse/` and should be treated as rebuildable artifacts.

## Build, Test, and Development Commands

From `ev_adoption_platform/`:

```powershell
pip install -r requirements.txt
python etl\build_warehouse.py
uvicorn api.main:app --reload --port 8000
```

From `ev_adoption_platform/frontend/`:

```powershell
npm install
npm run dev
npm run build
```

- `build_warehouse.py` rebuilds DuckDB marts from the archived Kaggle/original data.
- `uvicorn` starts the API at `http://127.0.0.1:8000`.
- `npm run dev` starts the dashboard at `http://localhost:3000`.
- `npm run build` validates the production Next.js build.

## Coding Style & Naming Conventions

Use Python type hints where practical and keep ETL functions small and explicit. Prefer `snake_case` for Python functions, variables, and warehouse columns.

Use TypeScript for frontend code. React components use `PascalCase`; local variables, hooks, and helpers use `camelCase`. Keep dashboard copy generic and avoid real automaker brand names unless real brand data is added.

## Testing Guidelines

There is no formal test suite yet. Before handing off changes, run:

```powershell
python etl\build_warehouse.py
npm run build
```

Also smoke-test key endpoints:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://localhost:3000/model -UseBasicParsing
```

Future tests should cover ETL schema validation, API response shapes, and dashboard route rendering.

## Commit & Pull Request Guidelines

No Git history is present, so use clear conventional-style commits such as:

- `feat: add interactive adoption simulator`
- `fix: correct score-file discovery after archive cleanup`
- `docs: update warehouse architecture notes`

Pull requests should include a short summary, screenshots for UI changes, commands run, and any data/model assumptions changed.

## Security & Configuration Tips

Do not commit API keys, cloud credentials, or private datasets. Keep cloud-specific values in environment variables. Local DuckDB and generated mart files are reproducible from the ETL script.
