from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = APP_ROOT / "warehouse" / "ev_adoption.duckdb"


app = FastAPI(title="EV Buyer Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    age: int = Field(35, ge=18, le=90)
    annual_income_usd: float = Field(90000, ge=0)
    daily_commute_km: float = Field(25, ge=0)
    number_of_cars_owned: int = Field(1, ge=0, le=8)
    charging_stations_near_home: int = Field(3, ge=0, le=40)
    charging_stations_near_work: int = Field(4, ge=0, le=40)
    environmental_concern_level: int = Field(6, ge=1, le=10)
    city_type: str = "Urban"
    current_car_type: str = "Sedan"
    home_charging_possible: str = "Yes"
    subsidy_available: str = "Yes"
    range_anxiety_level: str = "Low"


def query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Warehouse not found. Run `python etl/build_warehouse.py` first.",
        )
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        result = con.execute(sql, params or []).fetchdf()
    return result.replace({np.nan: None}).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if DB_PATH.exists() else "warehouse_missing",
        "warehouse_path": str(DB_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics/overview")
def overview_metrics() -> list[dict[str, Any]]:
    return query("SELECT * FROM mart_overview_metrics ORDER BY metric_name")


@app.get("/segments")
def segments(
    segment_type: str = Query("buyer_segment"),
    limit: int = Query(12, ge=1, le=100),
) -> list[dict[str, Any]]:
    return query(
        """
        SELECT *
        FROM mart_segment_metrics
        WHERE segment_type = ?
        ORDER BY avg_probability DESC
        LIMIT ?
        """,
        [segment_type, limit],
    )


@app.get("/customers")
def customers(
    band: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    if band:
        return query(
            """
            SELECT id, Age, Annual_Income_USD, Daily_Commute_km, Current_Car_Type,
                   City_Type, Home_Charging_Possible, Subsidy_Available,
                   Range_Anxiety_Level, ev_purchase_probability, adoption_band
            FROM scored_current_customers
            WHERE adoption_band = ?
            ORDER BY ev_purchase_probability DESC
            LIMIT ?
            """,
            [band, limit],
        )
    return query(
        """
        SELECT id, Age, Annual_Income_USD, Daily_Commute_km, Current_Car_Type,
               City_Type, Home_Charging_Possible, Subsidy_Available,
               Range_Anxiety_Level, ev_purchase_probability, adoption_band
        FROM scored_current_customers
        ORDER BY ev_purchase_probability DESC
        LIMIT ?
        """,
        [limit],
    )


@app.get("/data-quality")
def data_quality(dataset: str | None = Query(None)) -> list[dict[str, Any]]:
    if dataset:
        return query(
            """
            SELECT *
            FROM mart_data_quality
            WHERE dataset = ?
            ORDER BY missing_rate DESC, column_name
            """,
            [dataset],
        )
    return query(
        """
        SELECT *
        FROM mart_data_quality
        ORDER BY dataset, missing_rate DESC, column_name
        """
    )


@app.get("/drift")
def drift() -> list[dict[str, Any]]:
    return query(
        """
        SELECT *
        FROM mart_drift_metrics
        ORDER BY absolute_delta DESC
        """
    )


@app.get("/policy-simulation")
def policy_simulation() -> list[dict[str, Any]]:
    return query(
        """
        SELECT *
        FROM mart_policy_simulation
        ORDER BY incremental_expected_buyers DESC
        """
    )


@app.get("/live-feed")
def live_feed(limit: int = Query(20, ge=5, le=100)) -> dict[str, Any]:
    offset = int(datetime.now(timezone.utc).timestamp()) % 20000
    rows = query(
        """
        SELECT id, ev_purchase_probability, adoption_band, City_Type,
               Current_Car_Type, Range_Anxiety_Level, score_source
        FROM scored_current_customers
        ORDER BY id
        LIMIT ? OFFSET ?
        """,
        [limit, offset],
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": f"batch-{offset:05d}",
        "rows": rows,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    score = (
        -4.2
        + 0.000018 * payload.annual_income_usd
        + 0.13 * payload.environmental_concern_level
        - 0.012 * payload.daily_commute_km
        + 0.04 * (payload.charging_stations_near_home + payload.charging_stations_near_work)
        + 0.55 * (payload.home_charging_possible == "Yes")
        + 0.75 * (payload.subsidy_available == "Yes")
        + 0.16 * (payload.current_car_type == "SUV")
        - 0.55 * (payload.range_anxiety_level == "Medium")
        - 1.15 * (payload.range_anxiety_level == "High")
        + 0.16 * (payload.city_type == "Urban")
        - 0.006 * max(payload.age - 45, 0)
    )
    probability = float(1 / (1 + np.exp(-score)))
    if probability >= 0.65:
        band = "Priority"
    elif probability >= 0.35:
        band = "High"
    elif probability >= 0.15:
        band = "Watch"
    else:
        band = "Low"

    return {
        "ev_purchase_probability": probability,
        "adoption_band": band,
        "top_positive_factors": [
            "subsidy availability" if payload.subsidy_available == "Yes" else None,
            "home charging access" if payload.home_charging_possible == "Yes" else None,
            "environmental concern" if payload.environmental_concern_level >= 7 else None,
            "higher income" if payload.annual_income_usd >= 90000 else None,
            "SUV owner segment" if payload.current_car_type == "SUV" else None,
        ],
        "top_barriers": [
            "high range anxiety" if payload.range_anxiety_level == "High" else None,
            "no home charging" if payload.home_charging_possible != "Yes" else None,
            "no subsidy" if payload.subsidy_available != "Yes" else None,
            "long commute" if payload.daily_commute_km >= 55 else None,
        ],
    }
