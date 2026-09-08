from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from starter_baseline import (
    CATEGORICAL_COLS,
    DATA_DIR,
    ID_COL,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    add_features,
)


N_SPLITS = 5
SUBMISSION_PATH = DATA_DIR / "submission_catboost_5fold.csv"
OOF_PATH = DATA_DIR / "oof_catboost_5fold.csv"


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=500,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.4,
        bagging_temperature=0.5,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=50,
    )


def main() -> None:
    print("Loading data...", flush=True)
    train = add_features(pd.read_csv(TRAIN_PATH))
    test = add_features(pd.read_csv(TEST_PATH))

    features = CATEGORICAL_COLS + NUMERIC_COLS
    x = train[features]
    y = train[TARGET].map({"No": 0, "Yes": 1})
    x_test = test[features]

    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores: list[float] = []

    test_pool = Pool(x_test, cat_features=CATEGORICAL_COLS)

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x, y), start=1):
        print(f"\nTraining fold {fold}/{N_SPLITS}...", flush=True)
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_pool = Pool(x_train, y_train, cat_features=CATEGORICAL_COLS)
        valid_pool = Pool(x_valid, y_valid, cat_features=CATEGORICAL_COLS)

        model = build_model()
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=50,
        )

        valid_pred = model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)
        oof_pred[valid_idx] = valid_pred

        test_pred += model.predict_proba(test_pool)[:, 1] / N_SPLITS
        print(f"Fold {fold} ROC AUC: {fold_auc:.6f}", flush=True)

    overall_auc = roc_auc_score(y, oof_pred)
    print("\nFold ROC AUC scores:", flush=True)
    for fold, score in enumerate(fold_scores, start=1):
        print(f"  Fold {fold}: {score:.6f}", flush=True)
    print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}", flush=True)
    print(f"Std fold ROC AUC:  {np.std(fold_scores):.6f}", flush=True)
    print(f"OOF ROC AUC:       {overall_auc:.6f}", flush=True)

    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {SUBMISSION_PATH.name} with {len(submission):,} rows.", flush=True)

    oof = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "catboost_5fold_pred": oof_pred,
        }
    )
    oof.to_csv(OOF_PATH, index=False)
    print(f"Wrote {OOF_PATH.name} for diagnostics/blending.", flush=True)


if __name__ == "__main__":
    main()
