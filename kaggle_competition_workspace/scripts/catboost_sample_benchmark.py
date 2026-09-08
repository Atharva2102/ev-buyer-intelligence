from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from starter_baseline import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH,
    add_features,
)


SAMPLE_SIZE = 120_000


def main() -> None:
    print("Loading data...", flush=True)
    train = add_features(pd.read_csv(TRAIN_PATH))
    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1})

    train_sample, _ = train_test_split(
        train,
        train_size=SAMPLE_SIZE,
        stratify=train[TARGET],
        random_state=RANDOM_STATE,
    )

    features = CATEGORICAL_COLS + NUMERIC_COLS
    x = train_sample[features]
    y = train_sample[TARGET]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_pool = Pool(x_train, y_train, cat_features=CATEGORICAL_COLS)
    valid_pool = Pool(x_valid, y_valid, cat_features=CATEGORICAL_COLS)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=250,
        learning_rate=0.10,
        depth=6,
        l2_leaf_reg=6.0,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=25,
    )

    print(f"Training CatBoost on {len(x_train):,} rows...", flush=True)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=30,
    )

    valid_pred = model.predict_proba(valid_pool)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"Sample holdout ROC AUC: {auc:.6f}", flush=True)


if __name__ == "__main__":
    main()
