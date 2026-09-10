from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etl.build_warehouse import add_derived_features


FIELD_MAP = {
    "age": "Age",
    "annual_income_usd": "Annual_Income_USD",
    "daily_commute_km": "Daily_Commute_km",
    "number_of_cars_owned": "Number_of_Cars_Owned",
    "charging_stations_near_home": "Charging_Stations_Near_Home",
    "charging_stations_near_work": "Charging_Stations_Near_Work",
    "environmental_concern_level": "Environmental_Concern_Level",
    "gender": "Gender",
    "city_type": "City_Type",
    "current_car_type": "Current_Car_Type",
    "home_charging_possible": "Home_Charging_Possible",
    "subsidy_available": "Subsidy_Available",
    "range_anxiety_level": "Range_Anxiety_Level",
}


class ServingModel:
    def __init__(self, app_root: Path) -> None:
        model_dir = Path(os.getenv("MODEL_DIR", app_root / "models"))
        model_path = model_dir / "serving_lgbm.txt"
        metadata_path = model_dir / "serving_lgbm_metadata.json"
        self.booster = None
        self.metadata: dict[str, Any] = {}
        if model_path.exists() and metadata_path.exists():
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(model_path))
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "explainable-v1"))

    @property
    def kind(self) -> str:
        return "lightgbm" if self.booster is not None else "explainable_fallback"

    def predict(self, payload: dict[str, Any]) -> float:
        if self.booster is None:
            return self._fallback(payload)

        row = {target: payload[source] for source, target in FIELD_MAP.items()}
        frame = add_derived_features(pd.DataFrame([row]))
        for column, categories in self.metadata["categories"].items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        frame = frame[self.metadata["feature_columns"]]
        return float(np.clip(self.booster.predict(frame)[0], 1e-6, 1 - 1e-6))

    @staticmethod
    def _fallback(payload: dict[str, Any]) -> float:
        score = (
            -4.2
            + 0.000018 * payload["annual_income_usd"]
            + 0.13 * payload["environmental_concern_level"]
            - 0.012 * payload["daily_commute_km"]
            + 0.04 * (
                payload["charging_stations_near_home"]
                + payload["charging_stations_near_work"]
            )
            + 0.55 * (payload["home_charging_possible"] == "Yes")
            + 0.75 * (payload["subsidy_available"] == "Yes")
            + 0.16 * (payload["current_car_type"] == "SUV")
            - 0.55 * (payload["range_anxiety_level"] == "Medium")
            - 1.15 * (payload["range_anxiety_level"] == "High")
            + 0.16 * (payload["city_type"] == "Urban")
            - 0.006 * max(payload["age"] - 45, 0)
        )
        return float(1 / (1 + np.exp(-score)))


def adoption_band(probability: float) -> str:
    if probability >= 0.65:
        return "Priority"
    if probability >= 0.35:
        return "High"
    if probability >= 0.15:
        return "Watch"
    return "Low"
