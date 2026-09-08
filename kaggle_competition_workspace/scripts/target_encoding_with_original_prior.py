from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
N_SPLITS = 5
ALPHA = 20.0
BASELINE_NATIVE_LGBM_AUC = 0.941897

TARGET = "Will_Buy_EV"
ID_COL = "id"
MISSING_VALUE = "Missing"

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ORIGINAL_PATH = DATA_DIR / "OG Dataset" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
SUBMISSION_PATH = DATA_DIR / "submission_lgbm_target_encoded.csv"

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

TARGET_ENCODING_KEYS = [
    ("Gender", ["Gender"]),
    ("City_Type", ["City_Type"]),
    ("Current_Car_Type", ["Current_Car_Type"]),
    ("Home_Charging_Possible", ["Home_Charging_Possible"]),
    ("Subsidy_Available", ["Subsidy_Available"]),
    ("Range_Anxiety_Level", ["Range_Anxiety_Level"]),
    ("City_Type__Home_Charging_Possible", ["City_Type", "Home_Charging_Possible"]),
    ("Subsidy_Available__Range_Anxiety_Level", ["Subsidy_Available", "Range_Anxiety_Level"]),
]

TE_COLS = [f"te_original_prior_{name}" for name, _ in TARGET_ENCODING_KEYS]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + TE_COLS


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
        clean_category_series(df["City_Type"])
        + "_"
        + clean_category_series(df["Home_Charging_Possible"])
    )
    df["Subsidy_Range_Anxiety"] = (
        clean_category_series(df["Subsidy_Available"])
        + "_"
        + clean_category_series(df["Range_Anxiety_Level"])
    )

    return df


def clean_category_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_VALUE)


def make_key_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = [clean_category_series(df[column]) for column in columns]
    key = values[0]
    for value in values[1:]:
        key = key + "__" + value
    return key


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


def compute_original_priors(
    original: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    priors = {}
    original_y = original[TARGET].astype(float)

    for name, columns in TARGET_ENCODING_KEYS:
        key = make_key_series(original, columns)
        stats = (
            pd.DataFrame({"key": key, "target": original_y})
            .groupby("key", observed=True)["target"]
            .agg(original_rate="mean", original_count="count")
        )
        priors[name] = stats

    return priors


def print_prior_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    priors: dict[str, pd.DataFrame],
) -> None:
    print("\nOriginal-prior coverage by key:", flush=True)
    for name, columns in TARGET_ENCODING_KEYS:
        competition_values = set(make_key_series(train, columns).unique()).union(
            set(make_key_series(test, columns).unique())
        )
        original_values = set(priors[name].index)
        found = len(competition_values & original_values)
        fallback = len(competition_values - original_values)
        print(
            f"  {name}: {found} found in original, "
            f"{fallback} fallback to competition global mean",
            flush=True,
        )


def add_target_encodings_from_stats(
    target_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    priors: dict[str, pd.DataFrame],
    global_mean: float,
) -> pd.DataFrame:
    encoded = target_df.copy()

    for name, columns in TARGET_ENCODING_KEYS:
        key = make_key_series(target_df, columns)
        fold_stats = stats_df_for_key(stats_df, columns)
        prior_stats = priors[name]

        fold_sum = key.map(fold_stats["fold_sum"]).astype(float).fillna(0.0)
        fold_count = key.map(fold_stats["fold_count"]).astype(float).fillna(0.0)
        prior_rate = key.map(prior_stats["original_rate"]).astype(float).fillna(
            global_mean
        )

        encoded[f"te_original_prior_{name}"] = (
            fold_sum + ALPHA * prior_rate
        ) / (fold_count + ALPHA)

    return encoded


def stats_df_for_key(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    key = make_key_series(df, columns)
    return (
        pd.DataFrame({"key": key, "target": df[TARGET].astype(float)})
        .groupby("key", observed=True)["target"]
        .agg(fold_sum="sum", fold_count="count")
    )


def add_inner_oof_target_encodings(
    train_fold: pd.DataFrame,
    priors: dict[str, pd.DataFrame],
    global_mean: float,
) -> pd.DataFrame:
    encoded = train_fold.copy()
    for column in TE_COLS:
        encoded[column] = np.nan

    inner_folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    y = train_fold[TARGET].astype(int)

    for inner_train_idx, inner_valid_idx in inner_folds.split(train_fold, y):
        inner_fit = train_fold.iloc[inner_train_idx]
        inner_valid = train_fold.iloc[inner_valid_idx]
        encoded_inner_valid = add_target_encodings_from_stats(
            target_df=inner_valid,
            stats_df=inner_fit,
            priors=priors,
            global_mean=global_mean,
        )
        encoded.iloc[inner_valid_idx, encoded.columns.get_indexer(TE_COLS)] = (
            encoded_inner_valid[TE_COLS].to_numpy()
        )

    if encoded[TE_COLS].isna().any().any():
        raise ValueError("Inner OOF target encoding produced NaN values.")

    return encoded


def unify_categorical_levels(*frames: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLS:
        categories = sorted(
            set().union(
                *[
                    set(clean_category_series(frame[column]).unique())
                    for frame in frames
                ]
            )
        )
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[column] = clean_category_series(frame[column]).astype(dtype)


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


def main() -> None:
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

    priors = compute_original_priors(original)
    print_prior_coverage(train, test, priors)

    global_mean = float(train[TARGET].mean())
    print(f"\nCompetition global target mean: {global_mean:.6f}", flush=True)
    print(f"Target encoding smoothing alpha: {ALPHA:g}", flush=True)

    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    y = train[TARGET].astype(int)
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores: list[float] = []

    test_encoded = add_target_encodings_from_stats(
        target_df=test,
        stats_df=train,
        priors=priors,
        global_mean=global_mean,
    )

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train, y), start=1):
        print(f"\nTraining fold {fold}/{N_SPLITS}...", flush=True)
        train_fold = train.iloc[train_idx].copy()
        valid_fold = train.iloc[valid_idx].copy()

        train_encoded = add_inner_oof_target_encodings(
            train_fold=train_fold,
            priors=priors,
            global_mean=global_mean,
        )
        valid_encoded = add_target_encodings_from_stats(
            target_df=valid_fold,
            stats_df=train_fold,
            priors=priors,
            global_mean=global_mean,
        )

        x_train = train_encoded[FEATURE_COLS].copy()
        y_train = train_encoded[TARGET].astype(int)
        x_valid = valid_encoded[FEATURE_COLS].copy()
        y_valid = valid_encoded[TARGET].astype(int)
        x_test = test_encoded[FEATURE_COLS].copy()

        unify_categorical_levels(x_train, x_valid, x_test)

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
        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS

        print(
            f"Fold {fold} ROC AUC: {fold_auc:.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    oof_auc = roc_auc_score(y, oof_pred)
    delta = oof_auc - BASELINE_NATIVE_LGBM_AUC

    print("\n=== Target Encoding With Original Prior Results ===", flush=True)
    print(f"Fold scores: {[round(score, 6) for score in fold_scores]}", flush=True)
    print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}", flush=True)
    print(f"OOF ROC AUC:       {oof_auc:.6f}", flush=True)
    print(f"Baseline OOF AUC:  {BASELINE_NATIVE_LGBM_AUC:.6f}", flush=True)
    print(f"Delta vs baseline: {delta:+.6f}", flush=True)

    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: test_pred,
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
