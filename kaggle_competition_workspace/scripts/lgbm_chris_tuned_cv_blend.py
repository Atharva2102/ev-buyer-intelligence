from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from chris_plus_lgbm_blends import make_chris_features, unify_lgbm_categories


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID_COL = "id"

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
BEST_CONFIG_PATH = "lgbm_chris_best_config.json"

OOF_CHRIS_PATH = "oof_chris_xgb_three_models.csv"
OOF_LGBM_CHRIS_PATH = "oof_lgbm_chris_features.csv"
OOF_LGBM_NATIVE_PATH = "oof_lightgbm_5fold.csv"
SUB_CHRIS_PLUS_LGBM_PATH = "submission_chris_plus_lgbm_blend.csv"
SUB_LGBM_NATIVE_PATH = "submission_lightgbm_5fold.csv"

OOF_PATH = "oof_lgbm_chris_tuned.csv"
SUB_TUNED_PATH = "submission_lgbm_chris_tuned.csv"
SUB_BLEND_PATH = "submission_chris_lgbm_tuned_blend.csv"


def build_model(extra_params: dict | None = None) -> LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 3000,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.2,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }
    if extra_params:
        params.update(extra_params)
    return LGBMClassifier(**params)


def rank_average(predictions: pd.DataFrame) -> np.ndarray:
    ranked = predictions.apply(lambda column: rankdata(column, method="average"))
    averaged = ranked.mean(axis=1).to_numpy(dtype=float)
    return (averaged - averaged.min()) / (averaged.max() - averaged.min())


def run_tuned_lgbm(
    train_x: pd.DataFrame,
    y: np.ndarray,
    test_x: pd.DataFrame,
    params: dict,
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
    start = time.time()

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, y), start=1):
        x_train = train_x.iloc[train_idx].copy()
        x_valid = train_x.iloc[valid_idx].copy()
        x_test = test_x.copy()
        cat_cols = unify_lgbm_categories(x_train, x_valid, x_test)

        print(f"\nTuned LightGBM fold {fold}/{N_SPLITS}", flush=True)
        model = build_model(params)
        model.fit(
            x_train,
            y[train_idx],
            eval_set=[(x_valid, y[valid_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                early_stopping(stopping_rounds=100, verbose=False),
                log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_auc = roc_auc_score(y[valid_idx], valid_pred)
        oof[valid_idx] = valid_pred
        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS
        fold_scores.append(fold_auc)
        rounds.append(int(model.best_iteration_))
        print(
            f"Fold {fold} ROC AUC: {fold_auc:.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    print("\nTuned LightGBM summary:", flush=True)
    print(f"Fold scores: {[round(score, 6) for score in fold_scores]}", flush=True)
    print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}", flush=True)
    print(f"OOF ROC AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    print(f"Best iterations: {rounds}", flush=True)
    print(f"Seconds: {time.time() - start:.0f}", flush=True)
    return oof, test_pred


def evaluate_blends(
    y: np.ndarray,
    test_ids: pd.Series,
    tuned_oof: np.ndarray,
    tuned_test: np.ndarray,
) -> None:
    chris_oof = pd.read_csv(OOF_CHRIS_PATH)
    lgbm_chris_oof = pd.read_csv(OOF_LGBM_CHRIS_PATH)["lgbm_chris_features"].to_numpy()
    lgbm_native_oof = pd.read_csv(OOF_LGBM_NATIVE_PATH)["lightgbm_5fold_pred"].to_numpy()
    chris_plus_lgbm_test = pd.read_csv(SUB_CHRIS_PLUS_LGBM_PATH)[TARGET].to_numpy()
    lgbm_native_test = pd.read_csv(SUB_LGBM_NATIVE_PATH)[TARGET].to_numpy()

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
            "lgbm_chris_original": lgbm_chris_oof,
            "lgbm_native_previous": lgbm_native_oof,
            "lgbm_chris_tuned": tuned_oof,
        }
    )
    test_frame = pd.DataFrame(
        {
            "current_best": chris_plus_lgbm_test,
            "lgbm_native_previous": lgbm_native_test,
            "lgbm_chris_tuned": tuned_test,
        }
    )

    print("\nSingle/blend OOF AUCs:", flush=True)
    for column in oof_frame.columns:
        print(f"{column}: {roc_auc_score(y, oof_frame[column]):.6f}", flush=True)

    print("\nCorrelation matrix:", flush=True)
    print(oof_frame.corr().round(6).to_string(), flush=True)

    candidates = {
        "equal_current_best_plus_tuned_lgbm": (
            (current_best_oof + tuned_oof) / 2,
            (chris_plus_lgbm_test + tuned_test) / 2,
        ),
        "weighted_80_current_20_tuned_lgbm": (
            0.80 * current_best_oof + 0.20 * tuned_oof,
            0.80 * chris_plus_lgbm_test + 0.20 * tuned_test,
        ),
        "weighted_70_current_30_tuned_lgbm": (
            0.70 * current_best_oof + 0.30 * tuned_oof,
            0.70 * chris_plus_lgbm_test + 0.30 * tuned_test,
        ),
        "rank_current_best_plus_tuned_lgbm": (
            rank_average(oof_frame[["current_best", "lgbm_chris_tuned"]]),
            rank_average(test_frame[["current_best", "lgbm_chris_tuned"]]),
        ),
    }

    print("\nTuned LightGBM blend candidates:", flush=True)
    scores = {}
    for name, (oof_pred, _) in candidates.items():
        scores[name] = roc_auc_score(y, oof_pred)
        print(f"{name}: {scores[name]:.6f}", flush=True)

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    best_test = candidates[best_name][1]

    submission = pd.DataFrame({ID_COL: test_ids, TARGET: best_test})
    submission.to_csv(SUB_BLEND_PATH, index=False)
    print("\nBest tuned LightGBM blend:", flush=True)
    print(f"{best_name}: {best_score:.6f}", flush=True)
    print(f"Wrote {SUB_BLEND_PATH}", flush=True)
    print(f"Valid probabilities: {submission[TARGET].between(0, 1).all()}", flush=True)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = (train[TARGET] == "Yes").astype(int).to_numpy()
    params = json.load(open(BEST_CONFIG_PATH, encoding="utf-8"))["params"]
    if isinstance(params, str):
        params = json.loads(params)
    print(f"Using tuned LightGBM params: {json.dumps(params, sort_keys=True)}", flush=True)

    train_x = make_chris_features(train)
    test_x = make_chris_features(test)
    tuned_oof, tuned_test = run_tuned_lgbm(train_x, y, test_x, params)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "lgbm_chris_tuned": tuned_oof,
        }
    ).to_csv(OOF_PATH, index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: tuned_test}).to_csv(
        SUB_TUNED_PATH,
        index=False,
    )
    print(f"\nWrote {OOF_PATH}", flush=True)
    print(f"Wrote {SUB_TUNED_PATH}", flush=True)

    evaluate_blends(y, test[ID_COL], tuned_oof, tuned_test)


if __name__ == "__main__":
    main()
