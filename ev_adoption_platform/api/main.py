from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.event_store import create_event_store
from api.scoring import ServingModel, adoption_band


APP_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("WAREHOUSE_PATH", APP_ROOT / "warehouse" / "ev_adoption.duckdb"))
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
logger = logging.getLogger("ev_buyer_api")


app = FastAPI(title="EV Buyer Intelligence API", version="0.2.0")
event_store = create_event_store(APP_ROOT)
serving_model = ServingModel(APP_ROOT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
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
    gender: str = "Other"
    city_type: str = "Urban"
    current_car_type: str = "Sedan"
    home_charging_possible: str = "Yes"
    subsidy_available: str = "Yes"
    range_anxiety_level: str = "Low"


class PredictionResult(BaseModel):
    event_id: str
    occurred_at: str
    source: str
    model_version: str
    model_kind: str
    latency_ms: float
    event_recorded: bool
    ev_purchase_probability: float
    adoption_band: str
    top_positive_factors: list[str | None]
    top_barriers: list[str | None]


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
    warehouse_source_path = DB_PATH.with_suffix(f"{DB_PATH.suffix}.source.json")
    warehouse_source = None
    if warehouse_source_path.exists():
        warehouse_source = json.loads(warehouse_source_path.read_text(encoding="utf-8"))
    try:
        event_store_health = event_store.health()
    except Exception as error:
        logger.warning("Event-store health check failed", exc_info=error)
        event_store_health = {
            "status": "unavailable",
            "backend": event_store.backend_name,
            "error": type(error).__name__,
        }
    warehouse_ready = DB_PATH.exists()
    return {
        "status": "ok" if warehouse_ready else "warehouse_missing",
        "warehouse_path": str(DB_PATH),
        "warehouse_size_bytes": DB_PATH.stat().st_size if warehouse_ready else 0,
        "warehouse_source": warehouse_source,
        "event_store": event_store_health,
        "model_version": serving_model.version,
        "model_kind": serving_model.kind,
        "cors_origins": CORS_ORIGINS,
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
    operations = scoring_operations(limit=limit, window_minutes=1440)
    return {
        "generated_at": operations["generated_at"],
        "batch_id": "recorded-events",
        "rows": operations["recent_events"],
    }


def score_payload(payload: PredictionRequest, source: str) -> PredictionResult:
    started = time.perf_counter()
    event_id = str(uuid.uuid4())
    occurred_at = datetime.now(timezone.utc).isoformat()
    values = payload.model_dump()
    try:
        probability = serving_model.predict(values)
        band = adoption_band(probability)
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        event_store.put(
            {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "source": source,
                "status": "failed",
                "probability": None,
                "adoption_band": None,
                "latency_ms": latency_ms,
                "model_version": serving_model.version,
                "city_type": payload.city_type,
                "current_car_type": payload.current_car_type,
                "range_anxiety_level": payload.range_anxiety_level,
                "error_message": type(error).__name__,
            }
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    event = {
        "event_id": event_id,
        "occurred_at": occurred_at,
        "source": source,
        "status": "succeeded",
        "probability": probability,
        "adoption_band": band,
        "latency_ms": latency_ms,
        "model_version": serving_model.version,
        "city_type": payload.city_type,
        "current_car_type": payload.current_car_type,
        "range_anxiety_level": payload.range_anxiety_level,
        "error_message": None,
    }
    event_recorded = True
    try:
        event_store.put(event)
    except Exception as error:
        logger.warning("Could not persist scoring event %s", event_id, exc_info=error)
        event_recorded = False

    return PredictionResult(
        event_id=event_id,
        occurred_at=occurred_at,
        source=source,
        model_version=serving_model.version,
        model_kind=serving_model.kind,
        latency_ms=latency_ms,
        event_recorded=event_recorded,
        ev_purchase_probability=probability,
        adoption_band=band,
        top_positive_factors=[
            "subsidy availability" if payload.subsidy_available == "Yes" else None,
            "home charging access" if payload.home_charging_possible == "Yes" else None,
            "environmental concern" if payload.environmental_concern_level >= 7 else None,
            "higher income" if payload.annual_income_usd >= 90000 else None,
            "SUV owner segment" if payload.current_car_type == "SUV" else None,
        ],
        top_barriers=[
            "high range anxiety" if payload.range_anxiety_level == "High" else None,
            "no home charging" if payload.home_charging_possible != "Yes" else None,
            "no subsidy" if payload.subsidy_available != "Yes" else None,
            "long commute" if payload.daily_commute_km >= 55 else None,
        ],
    )


@app.post("/predict", response_model=PredictionResult)
def predict(payload: PredictionRequest) -> PredictionResult:
    return score_payload(payload, "simulator")


@app.post("/demo-score", response_model=PredictionResult)
def demo_score() -> PredictionResult:
    count = query("SELECT COUNT(*) AS count FROM training_reference")[0]["count"]
    offset = random.randrange(int(count))
    row = query(
        """
        SELECT Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
               Charging_Stations_Near_Home, Charging_Stations_Near_Work,
               Environmental_Concern_Level, Gender, City_Type, Current_Car_Type,
               Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level
        FROM training_reference
        LIMIT 1 OFFSET ?
        """,
        [offset],
    )[0]
    payload = PredictionRequest(
        age=int(row["Age"]),
        annual_income_usd=float(row["Annual_Income_USD"]),
        daily_commute_km=float(row["Daily_Commute_km"]),
        number_of_cars_owned=int(row["Number_of_Cars_Owned"]),
        charging_stations_near_home=int(row["Charging_Stations_Near_Home"]),
        charging_stations_near_work=int(row["Charging_Stations_Near_Work"]),
        environmental_concern_level=int(row["Environmental_Concern_Level"]),
        gender=str(row["Gender"]),
        city_type=str(row["City_Type"]),
        current_car_type=str(row["Current_Car_Type"]),
        home_charging_possible=str(row["Home_Charging_Possible"]),
        subsidy_available=str(row["Subsidy_Available"]),
        range_anxiety_level=str(row["Range_Anxiety_Level"]),
    )
    return score_payload(payload, "demo_stream")


@app.get("/scoring-operations")
def scoring_operations(
    limit: int = Query(12, ge=1, le=100),
    window_minutes: int = Query(60, ge=1, le=10080),
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=window_minutes)
    events = event_store.recent(since, limit=2000)
    succeeded = [event for event in events if event["status"] == "succeeded"]
    failed = [event for event in events if event["status"] == "failed"]
    latencies = [float(event["latency_ms"]) for event in succeeded]
    probabilities = [float(event["probability"]) for event in succeeded]
    oldest = min(
        (datetime.fromisoformat(str(event["occurred_at"])) for event in events),
        default=now,
    )
    active_minutes = max((now - oldest).total_seconds() / 60, 1)
    bands = {
        band: sum(event.get("adoption_band") == band for event in succeeded)
        for band in ["Priority", "High", "Watch", "Low"]
    }
    sources = {
        source: sum(event.get("source") == source for event in events)
        for source in ["simulator", "demo_stream"]
    }
    return {
        "generated_at": now.isoformat(),
        "window_minutes": window_minutes,
        "storage_backend": event_store.backend_name,
        "model_version": serving_model.version,
        "model_kind": serving_model.kind,
        "total_requests": len(events),
        "successful_requests": len(succeeded),
        "failed_requests": len(failed),
        "success_rate": len(succeeded) / len(events) if events else 1.0,
        "requests_per_minute": len(events) / active_minutes,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "avg_probability": float(np.mean(probabilities)) if probabilities else 0.0,
        "band_distribution": bands,
        "source_counts": sources,
        "recent_events": events[:limit],
    }
