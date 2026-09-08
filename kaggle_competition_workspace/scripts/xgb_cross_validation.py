from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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
    build_model,
)


N_SPLITS = 5
SUBMISSION_PATH = DATA_DIR / "submission_xgb_5fold.csv"
OOF_PATH = DATA_DIR / "oof_xgb_5fold.csv"


def main() -> None:
    print("Loading data...")
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

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x, y), start=1):
        print(f"\nTraining fold {fold}/{N_SPLITS}...")
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_model()
        model.fit(x_train, y_train)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)
        oof_pred[valid_idx] = valid_pred

        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS
        print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_pred)
    print("\nFold ROC AUC scores:")
    for fold, score in enumerate(fold_scores, start=1):
        print(f"  Fold {fold}: {score:.6f}")
    print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
    print(f"Std fold ROC AUC:  {np.std(fold_scores):.6f}")
    print(f"OOF ROC AUC:       {overall_auc:.6f}")

    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {SUBMISSION_PATH.name} with {len(submission):,} rows.")

    oof = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "xgb_5fold_pred": oof_pred,
        }
    )
    oof.to_csv(OOF_PATH, index=False)
    print(f"Wrote {OOF_PATH.name} for diagnostics/blending.")


if __name__ == "__main__":
    main()
