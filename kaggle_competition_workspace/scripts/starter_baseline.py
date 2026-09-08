from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
ID_COL = "id"

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "submission_xgb_baseline.csv"


NUMERIC_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Total_Charging_Stations",
    "Charging_Station_Difference",
    "Charging_per_Commute",
    "Income_per_Car",
]

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "City_Home_Charging",
    "Subsidy_Range_Anxiety",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add small, interpretable features that should matter for EV buying."""
    df = df.copy()

    total_charging = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    df["Total_Charging_Stations"] = total_charging
    df["Charging_Station_Difference"] = (
        df["Charging_Stations_Near_Work"] - df["Charging_Stations_Near_Home"]
    )
    df["Charging_per_Commute"] = total_charging / np.maximum(
        df["Daily_Commute_km"], 1
    )
    df["Income_per_Car"] = df["Annual_Income_USD"] / np.maximum(
        df["Number_of_Cars_Owned"], 1
    )

    df["City_Home_Charging"] = (
        df["City_Type"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)
    )
    df["Subsidy_Range_Anxiety"] = (
        df["Subsidy_Available"].astype(str)
        + "_"
        + df["Range_Anxiety_Level"].astype(str)
    )

    return df


def build_model(model_params: dict | None = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", "passthrough", NUMERIC_COLS),
        ]
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    if model_params:
        params.update(model_params)

    model = XGBClassifier(
        **params,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    print("Loading data...")
    train = add_features(pd.read_csv(TRAIN_PATH))
    test = add_features(pd.read_csv(TEST_PATH))

    features = CATEGORICAL_COLS + NUMERIC_COLS
    x = train[features]
    y = train[TARGET].map({"No": 0, "Yes": 1})
    x_test = test[features]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Training XGBoost baseline...")
    model = build_model()
    model.fit(x_train, y_train)

    valid_pred = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"Validation ROC AUC: {auc:.6f}")

    print("Refitting on all training data...")
    final_model = build_model()
    final_model.fit(x, y)

    test_pred = final_model.predict_proba(x_test)[:, 1]
    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH.name} with {len(submission):,} rows.")


if __name__ == "__main__":
    main()
