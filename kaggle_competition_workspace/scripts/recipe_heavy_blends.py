from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score


TARGET = "Will_Buy_EV"
ID_COL = "id"

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OOF_CHRIS_PATH = "oof_chris_xgb_three_models.csv"
OOF_LGBM_CHRIS_PATH = "oof_lgbm_chris_features.csv"
OOF_LGBM_NATIVE_PATH = "oof_lightgbm_5fold.csv"

SUB_CHRIS3_PATH = "submission_chris_xgb_starter_reproduction.csv"
SUB_CHRIS_PLUS_LGBM_PATH = "submission_chris_plus_lgbm_blend.csv"
SUB_LGBM_NATIVE_PATH = "submission_lightgbm_5fold.csv"


def recipe_score(df: pd.DataFrame) -> np.ndarray:
    return (
        1.2 * df["Annual_Income_USD"] / 1e5
        + 0.6 * df["Environmental_Concern_Level"]
        + 2.0 * (df["Subsidy_Available"] == "Yes")
        - 1.0 * (df["Range_Anxiety_Level"] == "Medium")
        - 3.0 * (df["Range_Anxiety_Level"] == "High")
    ).to_numpy(dtype=float)


def recipe_probability(df: pd.DataFrame) -> np.ndarray:
    return np.clip(norm.cdf(recipe_score(df) - 5.5), 1e-6, 1 - 1e-6)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def rank01(values: np.ndarray) -> np.ndarray:
    return minmax(rankdata(values, method="average"))


def rank_average(columns: list[np.ndarray]) -> np.ndarray:
    ranks = np.column_stack([rank01(column) for column in columns])
    return ranks.mean(axis=1)


def auc_table(y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    print("\n=== OOF AUCs ===", flush=True)
    for name, pred in predictions.items():
        print(f"{name}: {roc_auc_score(y, pred):.6f}", flush=True)


def correlation_table(predictions: dict[str, np.ndarray]) -> None:
    print("\n=== Correlation Matrix ===", flush=True)
    corr = pd.DataFrame(predictions).corr().round(6)
    print(corr.to_string(), flush=True)


def write_submission(name: str, ids: pd.Series, pred: np.ndarray) -> None:
    path = f"submission_{name}.csv"
    submission = pd.DataFrame({ID_COL: ids, TARGET: pred})
    submission.to_csv(path, index=False)
    print(
        f"Wrote {path}: min={submission[TARGET].min():.6f}, "
        f"mean={submission[TARGET].mean():.6f}, max={submission[TARGET].max():.6f}, "
        f"valid={submission[TARGET].between(0, 1).all()}",
        flush=True,
    )


def main() -> None:
    print("Loading train/test and saved predictions...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = (train[TARGET] == "Yes").astype(int).to_numpy()

    oof_chris = pd.read_csv(OOF_CHRIS_PATH)
    oof_lgbm_chris = pd.read_csv(OOF_LGBM_CHRIS_PATH)
    oof_lgbm_native = pd.read_csv(OOF_LGBM_NATIVE_PATH)

    sub_chris3 = pd.read_csv(SUB_CHRIS3_PATH)
    sub_chris_plus_lgbm = pd.read_csv(SUB_CHRIS_PLUS_LGBM_PATH)
    sub_lgbm_native = pd.read_csv(SUB_LGBM_NATIVE_PATH)

    recipe_oof_prob = recipe_probability(train)
    recipe_test_prob = recipe_probability(test)
    recipe_oof_rank = rank01(recipe_score(train))
    recipe_test_rank = rank01(recipe_score(test))

    chris3_oof = oof_chris[
        ["chris_m1", "chris_m2_base_margin", "chris_m3_recipe_feature"]
    ].mean(axis=1).to_numpy()
    chris3_test = sub_chris3[TARGET].to_numpy()

    lgbm_chris_oof = oof_lgbm_chris["lgbm_chris_features"].to_numpy()
    lgbm_native_oof = oof_lgbm_native["lightgbm_5fold_pred"].to_numpy()
    lgbm_native_test = sub_lgbm_native[TARGET].to_numpy()

    # This is the same 5-model equal blend submitted as submission_chris_plus_lgbm_blend.csv.
    chris_plus_lgbm_oof = (
        oof_chris["chris_m1"].to_numpy()
        + oof_chris["chris_m2_base_margin"].to_numpy()
        + oof_chris["chris_m3_recipe_feature"].to_numpy()
        + lgbm_chris_oof
        + lgbm_native_oof
    ) / 5
    chris_plus_lgbm_test = sub_chris_plus_lgbm[TARGET].to_numpy()

    base_predictions = {
        "recipe_prob": recipe_oof_prob,
        "recipe_rank": recipe_oof_rank,
        "chris3_equal": chris3_oof,
        "lgbm_chris_features": lgbm_chris_oof,
        "lgbm_native_previous": lgbm_native_oof,
        "chris_plus_lgbm_equal": chris_plus_lgbm_oof,
    }
    auc_table(y, base_predictions)
    correlation_table(base_predictions)

    candidates = {
        "recipe_probe_70_chris_plus_lgbm_30_recipe_prob": (
            0.70 * chris_plus_lgbm_oof + 0.30 * recipe_oof_prob,
            0.70 * chris_plus_lgbm_test + 0.30 * recipe_test_prob,
        ),
        "recipe_probe_50_chris_plus_lgbm_50_recipe_prob": (
            0.50 * chris_plus_lgbm_oof + 0.50 * recipe_oof_prob,
            0.50 * chris_plus_lgbm_test + 0.50 * recipe_test_prob,
        ),
        "recipe_probe_80_chris3_20_recipe_prob": (
            0.80 * chris3_oof + 0.20 * recipe_oof_prob,
            0.80 * chris3_test + 0.20 * recipe_test_prob,
        ),
        "recipe_probe_rank_chris_plus_lgbm_recipe": (
            rank_average([chris_plus_lgbm_oof, recipe_oof_prob]),
            rank_average([chris_plus_lgbm_test, recipe_test_prob]),
        ),
        "recipe_probe_formula_only_prob": (
            recipe_oof_prob,
            recipe_test_prob,
        ),
        "recipe_probe_formula_only_rank": (
            recipe_oof_rank,
            recipe_test_rank,
        ),
    }

    print("\n=== Candidate OOF AUCs ===", flush=True)
    scored = []
    for name, (oof_pred, _) in candidates.items():
        score = roc_auc_score(y, oof_pred)
        scored.append((name, score))
        print(f"{name}: {score:.6f}", flush=True)

    scored = sorted(scored, key=lambda item: item[1], reverse=True)
    print("\n=== Ranked Candidates ===", flush=True)
    for name, score in scored:
        print(f"{score:.6f}  {name}", flush=True)

    print("\nWriting candidate submissions...", flush=True)
    for name, (_, test_pred) in candidates.items():
        write_submission(name, test[ID_COL], test_pred)

    best_name, best_score = scored[0]
    print("\nBest CV candidate:", flush=True)
    print(f"{best_name}: {best_score:.6f}", flush=True)
    print(
        "For leaderboard probing, submit the top CV blend first, then try the "
        "recipe-heavier candidates if the public LB rewards formula weighting.",
        flush=True,
    )


if __name__ == "__main__":
    main()
