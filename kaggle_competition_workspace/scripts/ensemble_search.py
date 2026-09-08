from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from starter_baseline import DATA_DIR, ID_COL, RANDOM_STATE, TARGET


XGB_TUNED_OOF_PATH = DATA_DIR / "oof_xgb_tuned_5fold.csv"
XGB_BASE_OOF_PATH = DATA_DIR / "oof_xgb_5fold.csv"
LOGREG_OOF_PATH = DATA_DIR / "oof_logreg_5fold.csv"
LIGHTGBM_OOF_PATH = DATA_DIR / "oof_lightgbm_5fold.csv"

XGB_TUNED_SUB_PATH = DATA_DIR / "submission_xgb_tuned_5fold.csv"
XGB_BASE_SUB_PATH = DATA_DIR / "submission_xgb_5fold.csv"
LOGREG_SUB_PATH = DATA_DIR / "submission_logreg_5fold.csv"
LIGHTGBM_SUB_PATH = DATA_DIR / "submission_lightgbm_5fold.csv"

BLEND_RESULTS_PATH = DATA_DIR / "ensemble_results.csv"
BEST_BLEND_PATH = DATA_DIR / "submission_blend_best.csv"
STACK_PATH = DATA_DIR / "submission_stack_logreg_meta.csv"


def load_oof_predictions() -> tuple[pd.DataFrame, pd.Series]:
    xgb_tuned = pd.read_csv(XGB_TUNED_OOF_PATH)
    xgb_base = pd.read_csv(XGB_BASE_OOF_PATH)
    logreg = pd.read_csv(LOGREG_OOF_PATH)
    lightgbm = pd.read_csv(LIGHTGBM_OOF_PATH)

    oof = xgb_tuned[[ID_COL, TARGET, "xgb_tuned_5fold_pred"]].merge(
        xgb_base[[ID_COL, "xgb_5fold_pred"]], on=ID_COL, validate="one_to_one"
    )
    oof = oof.merge(
        logreg[[ID_COL, "logreg_5fold_pred"]], on=ID_COL, validate="one_to_one"
    )
    oof = oof.merge(
        lightgbm[[ID_COL, "lightgbm_5fold_pred"]], on=ID_COL, validate="one_to_one"
    )
    y = oof[TARGET]
    features = oof[
        [
            "xgb_tuned_5fold_pred",
            "xgb_5fold_pred",
            "logreg_5fold_pred",
            "lightgbm_5fold_pred",
        ]
    ]
    return features, y


def load_test_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    xgb_tuned = pd.read_csv(XGB_TUNED_SUB_PATH)
    xgb_base = pd.read_csv(XGB_BASE_SUB_PATH)
    logreg = pd.read_csv(LOGREG_SUB_PATH)
    lightgbm = pd.read_csv(LIGHTGBM_SUB_PATH)

    test = xgb_tuned[[ID_COL]].copy()
    preds = pd.DataFrame(
        {
            "xgb_tuned_5fold_pred": xgb_tuned[TARGET],
            "xgb_5fold_pred": xgb_base[TARGET],
            "logreg_5fold_pred": logreg[TARGET],
            "lightgbm_5fold_pred": lightgbm[TARGET],
        }
    )
    return test, preds


def search_weighted_blends(features: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    columns = list(features.columns)
    candidates = [
        {
            "xgb_tuned_5fold_pred": 1.00,
            "xgb_5fold_pred": 0.00,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.00,
        },
        {
            "xgb_tuned_5fold_pred": 0.00,
            "xgb_5fold_pred": 0.00,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 1.00,
        },
        {
            "xgb_tuned_5fold_pred": 0.50,
            "xgb_5fold_pred": 0.00,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.50,
        },
        {
            "xgb_tuned_5fold_pred": 0.40,
            "xgb_5fold_pred": 0.20,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.40,
        },
        {
            "xgb_tuned_5fold_pred": 0.30,
            "xgb_5fold_pred": 0.20,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.50,
        },
        {
            "xgb_tuned_5fold_pred": 0.20,
            "xgb_5fold_pred": 0.20,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.60,
        },
        {
            "xgb_tuned_5fold_pred": 0.45,
            "xgb_5fold_pred": 0.00,
            "logreg_5fold_pred": 0.05,
            "lightgbm_5fold_pred": 0.50,
        },
        {
            "xgb_tuned_5fold_pred": 0.40,
            "xgb_5fold_pred": 0.00,
            "logreg_5fold_pred": 0.10,
            "lightgbm_5fold_pred": 0.50,
        },
        {
            "xgb_tuned_5fold_pred": 0.60,
            "xgb_5fold_pred": 0.40,
            "logreg_5fold_pred": 0.00,
            "lightgbm_5fold_pred": 0.00,
        },
    ]

    for candidate in candidates:
        weights = [candidate[column] for column in columns]

        weight_array = np.array(weights, dtype=float)
        pred = features.to_numpy() @ weight_array
        auc = roc_auc_score(y, pred)

        rows.append(
            {
                "method": "weighted_average",
                "roc_auc": auc,
                **{
                    f"weight_{column}": weight
                    for column, weight in zip(columns, weight_array)
                },
            }
        )

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def make_stack_predictions(
    features: pd.DataFrame,
    y: pd.Series,
    test_features: pd.DataFrame,
) -> tuple[float, np.ndarray]:
    meta_model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    stack_oof = cross_val_predict(
        meta_model,
        features,
        y,
        cv=folds,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    stack_auc = roc_auc_score(y, stack_oof)

    meta_model.fit(features, y)
    stack_test = meta_model.predict_proba(test_features)[:, 1]
    return stack_auc, stack_test


def main() -> None:
    print("Loading OOF and test predictions...", flush=True)
    features, y = load_oof_predictions()
    test_ids, test_features = load_test_predictions()

    print("Base OOF scores:", flush=True)
    for column in features.columns:
        print(f"  {column}: {roc_auc_score(y, features[column]):.6f}", flush=True)

    print("\nSearching weighted blends...", flush=True)
    blend_results = search_weighted_blends(features, y)
    best_blend = blend_results.iloc[0]
    blend_results.to_csv(BLEND_RESULTS_PATH, index=False)

    print("Best weighted blend:", flush=True)
    print(best_blend.to_string(), flush=True)

    weight_cols = [column for column in blend_results.columns if column.startswith("weight_")]
    weights = np.array([best_blend[column] for column in weight_cols], dtype=float)
    best_blend_pred = test_features.to_numpy() @ weights
    best_blend_submission = pd.DataFrame(
        {
            ID_COL: test_ids[ID_COL],
            TARGET: best_blend_pred,
        }
    )
    best_blend_submission.to_csv(BEST_BLEND_PATH, index=False)
    print(f"\nWrote {BEST_BLEND_PATH.name}", flush=True)

    print("\nTraining logistic meta-model stack...", flush=True)
    stack_auc, stack_test = make_stack_predictions(features, y, test_features)
    stack_submission = pd.DataFrame(
        {
            ID_COL: test_ids[ID_COL],
            TARGET: stack_test,
        }
    )
    stack_submission.to_csv(STACK_PATH, index=False)
    print(f"Stacked meta-model OOF ROC AUC: {stack_auc:.6f}", flush=True)
    print(f"Wrote {STACK_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
