from __future__ import annotations

import json

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
BEST_CONFIG_PATH = DATA_DIR / "xgb_best_config.json"
SUBMISSION_PATH = DATA_DIR / "submission_xgb_tuned_5fold.csv"
OOF_PATH = DATA_DIR / "oof_xgb_tuned_5fold.csv"


def load_best_params() -> dict:
    config = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Using tuned config: {config['name']}", flush=True)
    print(json.dumps(config["params"], sort_keys=True), flush=True)
    return config["params"]


def main() -> None:
    params = load_best_params()

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

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x, y), start=1):
        print(f"\nTraining tuned fold {fold}/{N_SPLITS}...", flush=True)
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_model(params)
        model.fit(x_train, y_train)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)
        oof_pred[valid_idx] = valid_pred

        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS
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
            "xgb_tuned_5fold_pred": oof_pred,
        }
    )
    oof.to_csv(OOF_PATH, index=False)
    print(f"Wrote {OOF_PATH.name} for diagnostics/blending.", flush=True)


if __name__ == "__main__":
    main()
