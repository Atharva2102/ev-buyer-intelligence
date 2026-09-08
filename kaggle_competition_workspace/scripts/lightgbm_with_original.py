from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID_COL = "id"

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ORIGINAL_PATH = DATA_DIR / "OG Dataset" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
SUBMISSION_PATH = DATA_DIR / "submission_lgbm_with_original.csv"

BASE_FEATURE_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

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
    "is_original",
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

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
        df["City_Type"].astype("string").fillna("Missing")
        + "_"
        + df["Home_Charging_Possible"].astype("string").fillna("Missing")
    )
    df["Subsidy_Range_Anxiety"] = (
        df["Subsidy_Available"].astype("string").fillna("Missing")
        + "_"
        + df["Range_Anxiety_Level"].astype("string").fillna("Missing")
    )

    return df


def load_original() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_PATH)
    print(f"Original CSV: {ORIGINAL_PATH}", flush=True)
    print("Original columns:", flush=True)
    for column in original.columns:
        print(f"  {column}", flush=True)

    if TARGET not in original.columns:
        target_candidates = [
            column for column in original.columns if column.lower() == TARGET.lower()
        ]
        if len(target_candidates) != 1:
            raise ValueError(f"Could not find original target column for {TARGET}.")
        original = original.rename(columns={target_candidates[0]: TARGET})

    missing_features = [
        column for column in BASE_FEATURE_COLS if column not in original.columns
    ]
    print(
        "Missing competition feature columns in original:",
        missing_features if missing_features else "None",
        flush=True,
    )

    for column in missing_features:
        original[column] = np.nan

    original[TARGET] = original[TARGET].map({"No": 0, "Yes": 1})
    if original[TARGET].isna().any():
        raise ValueError("Original target mapping produced NaN values.")

    return original


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    print("Loading competition train/test...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    original = load_original()

    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1})
    if train[TARGET].isna().any():
        raise ValueError("Competition target mapping produced NaN values.")

    train["is_original"] = 0
    test["is_original"] = 0
    original["is_original"] = 1

    train = add_features(train)
    test = add_features(test)
    original = add_features(original)

    train_x = train[FEATURE_COLS].copy()
    test_x = test[FEATURE_COLS].copy()
    original_x = original[FEATURE_COLS].copy()

    train_y = train[TARGET].astype(int)
    original_y = original[TARGET].astype(int)

    unify_categorical_levels(train_x, test_x, original_x)

    return train_x, train_y, test[[ID_COL]].copy(), test_x, original_x, original_y


def unify_categorical_levels(*frames: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLS:
        categories = sorted(
            set().union(
                *[
                    set(frame[column].astype("string").fillna("Missing").unique())
                    for frame in frames
                ]
            )
        )
        for frame in frames:
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("Missing")
                .astype(pd.CategoricalDtype(categories=categories))
            )


def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=3000,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def run_cv(
    label: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    original_x: pd.DataFrame | None = None,
    original_y: pd.Series | None = None,
) -> tuple[float, np.ndarray]:
    print(f"\n=== {label} ===", flush=True)
    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_pred = np.zeros(len(train_x))
    test_pred = np.zeros(len(test_x))
    fold_scores: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, train_y), start=1):
        x_train = train_x.iloc[train_idx].copy()
        y_train = train_y.iloc[train_idx].copy()
        x_valid = train_x.iloc[valid_idx].copy()
        y_valid = train_y.iloc[valid_idx].copy()

        if original_x is not None and original_y is not None:
            x_train = pd.concat([x_train, original_x], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, original_y], axis=0, ignore_index=True)

        print(
            f"Fold {fold}: train rows={len(x_train):,}, "
            f"validation competition rows={len(x_valid):,}",
            flush=True,
        )

        model = build_model()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[
                early_stopping(stopping_rounds=100, verbose=False),
                log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)
        oof_pred[valid_idx] = valid_pred

        test_pred += model.predict_proba(test_x)[:, 1] / N_SPLITS
        print(
            f"Fold {fold} ROC AUC: {fold_auc:.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    oof_auc = roc_auc_score(train_y, oof_pred)
    print(f"{label} fold scores: {[round(score, 6) for score in fold_scores]}", flush=True)
    print(f"{label} mean fold ROC AUC: {np.mean(fold_scores):.6f}", flush=True)
    print(f"{label} OOF ROC AUC: {oof_auc:.6f}", flush=True)

    return oof_auc, test_pred


def main() -> None:
    train_x, train_y, test_ids, test_x, original_x, original_y = prepare_data()

    print("\nData checks:", flush=True)
    print(f"Competition train rows: {len(train_x):,}", flush=True)
    print(f"Test rows:              {len(test_x):,}", flush=True)
    print(f"Original rows:          {len(original_x):,}", flush=True)
    print(f"Features used:          {len(FEATURE_COLS)}", flush=True)
    print(f"Categorical features:   {CATEGORICAL_COLS}", flush=True)

    baseline_auc, _ = run_cv(
        label="Baseline native-categorical LightGBM without original",
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
    )
    with_original_auc, with_original_test_pred = run_cv(
        label="Native-categorical LightGBM with all original rows in training only",
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        original_x=original_x,
        original_y=original_y,
    )

    delta = with_original_auc - baseline_auc
    print("\n=== Comparison ===", flush=True)
    print(f"Baseline OOF AUC:      {baseline_auc:.6f}", flush=True)
    print(f"With original OOF AUC: {with_original_auc:.6f}", flush=True)
    print(f"Delta:                 {delta:+.6f}", flush=True)

    submission = pd.DataFrame(
        {
            ID_COL: test_ids[ID_COL],
            TARGET: with_original_test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nWrote {SUBMISSION_PATH.name}", flush=True)
    print(f"Submission rows: {len(submission):,}", flush=True)
    print(f"Submission columns: {list(submission.columns)}", flush=True)
    print(
        "Probabilities in [0, 1]: "
        f"{submission[TARGET].between(0, 1).all()}",
        flush=True,
    )
    print(submission[TARGET].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
