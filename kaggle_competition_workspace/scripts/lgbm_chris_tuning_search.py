from __future__ import annotations

import json

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from chris_plus_lgbm_blends import make_chris_features, unify_lgbm_categories


RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
TRAIN_PATH = "train.csv"
RESULTS_PATH = "lgbm_chris_tuning_results.csv"
BEST_CONFIG_PATH = "lgbm_chris_best_config.json"


CONFIGS = [
    {
        "name": "current_lgbm_chris",
        "params": {},
    },
    {
        "name": "more_leaves_regularized",
        "params": {
            "num_leaves": 63,
            "min_child_samples": 100,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
        },
    },
    {
        "name": "smaller_leaves",
        "params": {
            "num_leaves": 24,
            "min_child_samples": 50,
            "reg_lambda": 4.0,
            "reg_alpha": 0.1,
        },
    },
    {
        "name": "more_random",
        "params": {
            "num_leaves": 31,
            "min_child_samples": 70,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
    },
    {
        "name": "slower_more_trees",
        "params": {
            "n_estimators": 4500,
            "learning_rate": 0.018,
            "num_leaves": 31,
            "min_child_samples": 60,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
    },
]


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


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    y = (train[TARGET] == "Yes").astype(int)
    x = make_chris_features(train)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cat_cols = unify_lgbm_categories(x_train, x_valid)

    rows = []
    best = None
    for i, config in enumerate(CONFIGS, start=1):
        print(f"\n[{i}/{len(CONFIGS)}] {config['name']}", flush=True)
        print(json.dumps(config["params"], sort_keys=True), flush=True)

        model = build_model(config["params"])
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                early_stopping(stopping_rounds=100, verbose=False),
                log_evaluation(period=0),
            ],
        )

        pred = model.predict_proba(x_valid)[:, 1]
        auc = roc_auc_score(y_valid, pred)
        row = {
            "name": config["name"],
            "roc_auc": auc,
            "best_iteration": model.best_iteration_,
            "params": json.dumps(config["params"], sort_keys=True),
        }
        rows.append(row)
        print(f"ROC AUC: {auc:.6f}, best_iteration={model.best_iteration_}", flush=True)
        if best is None or auc > best["roc_auc"]:
            best = row

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(RESULTS_PATH, index=False)
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print("\nTuning results:", flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"\nWrote {RESULTS_PATH}", flush=True)
    print(f"Wrote {BEST_CONFIG_PATH}", flush=True)


if __name__ == "__main__":
    main()
