from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID_COL = "id"

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
SUBMISSION_PATH = "submission_chris_xgb_starter_reproduction.csv"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns]).copy()
    home = (df["Home_Charging_Possible"] == "Yes").astype(int)
    subsidy = (df["Subsidy_Available"] == "Yes").astype(int)

    x["worry_score"] = (
        df["Daily_Commute_km"]
        - 5 * df["Charging_Stations_Near_Home"]
        - 5 * df["Charging_Stations_Near_Work"]
        - 150 * home
    )
    x["chargers_total"] = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    x["income_x_subsidy"] = df["Annual_Income_USD"] / 1e5 * subsidy
    x["concern_x_subsidy"] = df["Environmental_Concern_Level"] * subsidy
    return x


def recipe_score(df: pd.DataFrame) -> np.ndarray:
    return (
        1.2 * df["Annual_Income_USD"] / 1e5
        + 0.6 * df["Environmental_Concern_Level"]
        + 2.0 * (df["Subsidy_Available"] == "Yes")
        - 1.0 * (df["Range_Anxiety_Level"] == "Medium")
        - 3.0 * (df["Range_Anxiety_Level"] == "High")
    ).to_numpy()


def recipe_logit(df: pd.DataFrame) -> np.ndarray:
    probability = np.clip(norm.cdf(recipe_score(df) - 5.5), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def encode_categories(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_x.copy()
    test_x = test_x.copy()

    for column in train_x.select_dtypes("object").columns:
        categories = pd.Categorical(pd.concat([train_x[column], test_x[column]])).categories
        train_x[column] = pd.Categorical(train_x[column], categories=categories).codes
        test_x[column] = pd.Categorical(test_x[column], categories=categories).codes

    return train_x, test_x


def run_xgb(
    name: str,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    margin_train: np.ndarray | None = None,
    margin_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    params = {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "auc",
        "early_stopping_rounds": 100,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test_x))
    rounds = []
    start = time.time()

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        model = XGBClassifier(**params)
        if margin_train is None:
            model.fit(
                train_x.iloc[train_idx],
                y[train_idx],
                eval_set=[(train_x.iloc[valid_idx], y[valid_idx])],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
            test_pred += model.predict_proba(test_x)[:, 1] / N_SPLITS
        else:
            model.fit(
                train_x.iloc[train_idx],
                y[train_idx],
                eval_set=[(train_x.iloc[valid_idx], y[valid_idx])],
                base_margin=margin_train[train_idx],
                base_margin_eval_set=[margin_train[valid_idx]],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(
                train_x.iloc[valid_idx],
                base_margin=margin_train[valid_idx],
            )[:, 1]
            test_pred += (
                model.predict_proba(test_x, base_margin=margin_test)[:, 1] / N_SPLITS
            )

        rounds.append(int(model.best_iteration))

    fold_aucs = [roc_auc_score(y[valid_idx], oof[valid_idx]) for _, valid_idx in folds]
    print(
        f"{name}: CV AUC={roc_auc_score(y, oof):.6f}, "
        f"folds={[round(score, 6) for score in fold_aucs]}, "
        f"rounds={rounds}, seconds={time.time() - start:.0f}",
        flush=True,
    )
    return oof, test_pred


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    y = (train[TARGET] == "Yes").astype(int).to_numpy()

    print(
        f"train {len(train):,} rows, test {len(test):,} rows, buyers {y.mean():.1%}",
        flush=True,
    )

    train_x, test_x = encode_categories(
        make_features(train),
        make_features(test),
    )
    print(f"{train_x.shape[1]} features: {list(train_x.columns)}", flush=True)

    score_train = recipe_score(train)
    score_test = recipe_score(test)
    margin_train = recipe_logit(train)
    margin_test = recipe_logit(test)
    print(f"Recipe alone train AUC: {roc_auc_score(y, score_train):.6f}", flush=True)

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(train_x, y)
    )

    m1_oof, m1_test = run_xgb("Model 1 - baseline", train_x, test_x, y, folds)
    m2_oof, m2_test = run_xgb(
        "Model 2 - recipe as base margin",
        train_x,
        test_x,
        y,
        folds,
        margin_train=margin_train,
        margin_test=margin_test,
    )

    train_x3 = train_x.copy()
    test_x3 = test_x.copy()
    train_x3["recipe_score"] = score_train
    test_x3["recipe_score"] = score_test
    m3_oof, m3_test = run_xgb("Model 3 - recipe as feature", train_x3, test_x3, y, folds)

    oof_predictions = pd.DataFrame(
        {
            "m1_baseline": m1_oof,
            "m2_base_margin": m2_oof,
            "m3_recipe_feature": m3_oof,
        }
    )
    print("\nModel correlation matrix:", flush=True)
    print(oof_predictions.corr().round(6).to_string(), flush=True)

    blend_oof = oof_predictions.mean(axis=1).to_numpy()
    blend_test = (m1_test + m2_test + m3_test) / 3
    print(f"\nEqual three-model blend CV AUC: {roc_auc_score(y, blend_oof):.6f}", flush=True)

    submission[TARGET] = blend_test
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH}", flush=True)
    print(
        f"Submission valid probabilities: {submission[TARGET].between(0, 1).all()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
