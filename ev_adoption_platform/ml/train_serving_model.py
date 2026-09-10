from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.build_warehouse import TARGET, add_derived_features, load_csvs, normalize_competition


MODEL_DIR = ROOT / "models"
MODEL_VERSION = "lgbm-serving-2026-09-v1"


def main() -> None:
    train_raw, test_raw, _ = load_csvs(ROOT.parent)
    train, _ = normalize_competition(train_raw, test_raw)
    train = add_derived_features(train)
    feature_columns = [column for column in train.columns if column not in {"id", TARGET}]
    categorical_columns = [
        column for column in feature_columns if train[column].dtype == object
    ]
    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = sorted(train[column].fillna("Missing").astype(str).unique().tolist())
        categories[column] = values
        train[column] = pd.Categorical(
            train[column].fillna("Missing").astype(str), categories=values
        )

    X = train[feature_columns]
    y = train[TARGET].astype(int)
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=0.12, random_state=42, stratify=y
    )
    params = {
        "objective": "binary",
        "n_estimators": 1600,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    validation_model = lgb.LGBMClassifier(**params)
    validation_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_columns,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iteration = int(validation_model.best_iteration_)
    validation_auc = roc_auc_score(
        y_valid, validation_model.predict_proba(X_valid)[:, 1]
    )

    final_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_iteration})
    final_model.fit(X, y, categorical_feature=categorical_columns)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(MODEL_DIR / "serving_lgbm.txt"))
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train),
        "validation_auc": validation_auc,
        "best_iteration": best_iteration,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "categories": categories,
    }
    (MODEL_DIR / "serving_lgbm_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
