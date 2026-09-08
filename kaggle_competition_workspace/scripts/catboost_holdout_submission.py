from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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


SUBMISSION_PATH = DATA_DIR / "submission_catboost_holdout.csv"


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
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


def main() -> None:
    print("Loading data...", flush=True)
    train = add_features(pd.read_csv(TRAIN_PATH))
    test = add_features(pd.read_csv(TEST_PATH))

    features = CATEGORICAL_COLS + NUMERIC_COLS
    x = train[features]
    y = train[TARGET].map({"No": 0, "Yes": 1})
    x_test = test[features]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_pool = Pool(x_train, y_train, cat_features=CATEGORICAL_COLS)
    valid_pool = Pool(x_valid, y_valid, cat_features=CATEGORICAL_COLS)
    test_pool = Pool(x_test, cat_features=CATEGORICAL_COLS)

    print("Training CatBoost holdout model...", flush=True)
    model = build_model()
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=30,
    )

    valid_pred = model.predict_proba(valid_pool)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"Validation ROC AUC: {auc:.6f}", flush=True)

    print("Refitting CatBoost on all training data...", flush=True)
    full_pool = Pool(x, y, cat_features=CATEGORICAL_COLS)
    final_model = build_model()
    final_model.fit(full_pool)

    test_pred = final_model.predict_proba(test_pool)[:, 1]
    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            TARGET: test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH.name} with {len(submission):,} rows.", flush=True)


if __name__ == "__main__":
    main()
