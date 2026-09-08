from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from starter_baseline import (
    CATEGORICAL_COLS,
    DATA_DIR,
    ID_COL,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH,
    add_features,
    build_model,
)


RESULTS_PATH = DATA_DIR / "xgb_tuning_results.csv"
BEST_CONFIG_PATH = DATA_DIR / "xgb_best_config.json"

CONFIGS = [
    {
        "name": "baseline",
        "params": {},
    },
    {
        "name": "shallower_more_trees",
        "params": {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 6,
            "reg_lambda": 3.0,
        },
    },
    {
        "name": "deeper_lower_lr",
        "params": {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "max_depth": 6,
            "min_child_weight": 8,
            "reg_lambda": 4.0,
        },
    },
    {
        "name": "regularized",
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "max_depth": 5,
            "min_child_weight": 8,
            "reg_lambda": 6.0,
            "reg_alpha": 0.2,
        },
    },
    {
        "name": "more_random",
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.045,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 2.0,
        },
    },
]


def main() -> None:
    print("Loading data...", flush=True)
    train = add_features(pd.read_csv(TRAIN_PATH))

    features = CATEGORICAL_COLS + NUMERIC_COLS
    x = train[features]
    y = train[TARGET].map({"No": 0, "Yes": 1})

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rows = []
    best_name = None
    best_params = None
    best_auc = -1.0

    for i, config in enumerate(CONFIGS, start=1):
        name = config["name"]
        params = config["params"]
        print(f"\n[{i}/{len(CONFIGS)}] Training {name}...", flush=True)
        print(json.dumps(params, sort_keys=True), flush=True)

        model = build_model(params)
        model.fit(x_train, y_train)
        valid_pred = model.predict_proba(x_valid)[:, 1]
        auc = roc_auc_score(y_valid, valid_pred)

        row = {
            "name": name,
            "roc_auc": auc,
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        print(f"{name} ROC AUC: {auc:.6f}", flush=True)

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_params = params

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(RESULTS_PATH, index=False)

    best_config = {
        "name": best_name,
        "roc_auc": best_auc,
        "params": best_params,
    }
    BEST_CONFIG_PATH.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print("\nTuning results:", flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"\nWrote {RESULTS_PATH.name}", flush=True)
    print(f"Wrote {BEST_CONFIG_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
