from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from chris_plus_lgbm_blends import make_chris_features


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID_COL = "id"

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OOF_CHRIS_PATH = "oof_chris_xgb_three_models.csv"
OOF_LGBM_CHRIS_PATH = "oof_lgbm_chris_features.csv"
OOF_LGBM_NATIVE_PATH = "oof_lightgbm_5fold.csv"
SUB_CHRIS_PLUS_LGBM_PATH = "submission_chris_plus_lgbm_blend.csv"
SUB_LGBM_CHRIS_PATH = "submission_chris_plus_lgbm_blend.csv"
SUB_LGBM_NATIVE_PATH = "submission_lightgbm_5fold.csv"

OOF_PATH = "oof_catboost_chris_features.csv"
SUB_CATBOOST_PATH = "submission_catboost_chris_features.csv"
SUB_BLEND_PATH = "submission_chris_lgbm_catboost_blend.csv"


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=350,
        learning_rate=0.10,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.3,
        bagging_temperature=0.4,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=50,
    )


def rank_average(predictions: pd.DataFrame) -> np.ndarray:
    ranked = predictions.apply(lambda column: rankdata(column, method="average"))
    averaged = ranked.mean(axis=1).to_numpy(dtype=float)
    return (averaged - averaged.min()) / (averaged.max() - averaged.min())


def train_catboost(
    train_x: pd.DataFrame,
    y: np.ndarray,
    test_x: pd.DataFrame,
    cat_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof = np.zeros(len(train_x))
    test_pred = np.zeros(len(test_x))
    fold_scores = []
    rounds = []
    test_pool = Pool(test_x, cat_features=cat_features)
    start = time.time()

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, y), start=1):
        print(f"\nCatBoost Chris fold {fold}/{N_SPLITS}", flush=True)
        train_pool = Pool(
            train_x.iloc[train_idx],
            y[train_idx],
            cat_features=cat_features,
        )
        valid_pool = Pool(
            train_x.iloc[valid_idx],
            y[valid_idx],
            cat_features=cat_features,
        )

        model = build_model()
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=40,
        )

        valid_pred = model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y[valid_idx], valid_pred)
        oof[valid_idx] = valid_pred
        test_pred += model.predict_proba(test_pool)[:, 1] / N_SPLITS
        fold_scores.append(fold_auc)
        rounds.append(model.get_best_iteration())
        print(
            f"Fold {fold} ROC AUC: {fold_auc:.6f}, "
            f"best_iteration={model.get_best_iteration()}",
            flush=True,
        )

    print("\nCatBoost Chris features summary:", flush=True)
    print(f"Fold scores: {[round(score, 6) for score in fold_scores]}", flush=True)
    print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}", flush=True)
    print(f"OOF ROC AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    print(f"Best iterations: {rounds}", flush=True)
    print(f"Seconds: {time.time() - start:.0f}", flush=True)
    return oof, test_pred


def evaluate_blends(
    y: np.ndarray,
    test_ids: pd.Series,
    cat_oof: np.ndarray,
    cat_test: np.ndarray,
) -> None:
    chris_oof = pd.read_csv(OOF_CHRIS_PATH)
    lgbm_chris_oof = pd.read_csv(OOF_LGBM_CHRIS_PATH)["lgbm_chris_features"].to_numpy()
    lgbm_native_oof = pd.read_csv(OOF_LGBM_NATIVE_PATH)["lightgbm_5fold_pred"].to_numpy()
    chris_plus_lgbm_test = pd.read_csv(SUB_CHRIS_PLUS_LGBM_PATH)[TARGET].to_numpy()
    lgbm_native_test = pd.read_csv(SUB_LGBM_NATIVE_PATH)[TARGET].to_numpy()

    chris3_oof = chris_oof[
        ["chris_m1", "chris_m2_base_margin", "chris_m3_recipe_feature"]
    ].mean(axis=1).to_numpy()
    current_best_oof = (
        chris_oof["chris_m1"].to_numpy()
        + chris_oof["chris_m2_base_margin"].to_numpy()
        + chris_oof["chris_m3_recipe_feature"].to_numpy()
        + lgbm_chris_oof
        + lgbm_native_oof
    ) / 5

    oof_frame = pd.DataFrame(
        {
            "current_best": current_best_oof,
            "chris3": chris3_oof,
            "lgbm_chris": lgbm_chris_oof,
            "lgbm_native": lgbm_native_oof,
            "catboost_chris": cat_oof,
        }
    )
    print("\nSingle/blend OOF AUCs:", flush=True)
    for column in oof_frame.columns:
        print(f"{column}: {roc_auc_score(y, oof_frame[column]):.6f}", flush=True)

    print("\nCorrelation matrix:", flush=True)
    print(oof_frame.corr().round(6).to_string(), flush=True)

    candidates = {
        "equal_current_best_plus_catboost": (
            (current_best_oof + cat_oof) / 2,
            (chris_plus_lgbm_test + cat_test) / 2,
        ),
        "weighted_80_current_20_catboost": (
            0.80 * current_best_oof + 0.20 * cat_oof,
            0.80 * chris_plus_lgbm_test + 0.20 * cat_test,
        ),
        "weighted_90_current_10_catboost": (
            0.90 * current_best_oof + 0.10 * cat_oof,
            0.90 * chris_plus_lgbm_test + 0.10 * cat_test,
        ),
        "rank_current_best_plus_catboost": (
            rank_average(oof_frame[["current_best", "catboost_chris"]]),
            rank_average(
                pd.DataFrame(
                    {
                        "current_best": chris_plus_lgbm_test,
                        "catboost_chris": cat_test,
                    }
                )
            ),
        ),
    }

    print("\nCatBoost blend candidates:", flush=True)
    scores = {}
    for name, (oof_pred, _) in candidates.items():
        scores[name] = roc_auc_score(y, oof_pred)
        print(f"{name}: {scores[name]:.6f}", flush=True)

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    best_test = candidates[best_name][1]

    submission = pd.DataFrame({ID_COL: test_ids, TARGET: best_test})
    submission.to_csv(SUB_BLEND_PATH, index=False)
    print("\nBest CatBoost blend:", flush=True)
    print(f"{best_name}: {best_score:.6f}", flush=True)
    print(f"Wrote {SUB_BLEND_PATH}", flush=True)
    print(f"Valid probabilities: {submission[TARGET].between(0, 1).all()}", flush=True)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = (train[TARGET] == "Yes").astype(int).to_numpy()

    train_x = make_chris_features(train)
    test_x = make_chris_features(test)
    cat_features = list(train_x.select_dtypes("object").columns)
    print(f"CatBoost categorical features: {cat_features}", flush=True)
    print(f"Feature count: {train_x.shape[1]}", flush=True)

    cat_oof, cat_test = train_catboost(train_x, y, test_x, cat_features)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "catboost_chris_features": cat_oof,
        }
    ).to_csv(OOF_PATH, index=False)
    pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: cat_test,
        }
    ).to_csv(SUB_CATBOOST_PATH, index=False)
    print(f"\nWrote {OOF_PATH}", flush=True)
    print(f"Wrote {SUB_CATBOOST_PATH}", flush=True)

    evaluate_blends(y, test[ID_COL], cat_oof, cat_test)


if __name__ == "__main__":
    main()
