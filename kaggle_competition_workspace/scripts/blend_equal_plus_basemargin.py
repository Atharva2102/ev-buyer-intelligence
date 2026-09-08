from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.special import logit
from scipy.stats import norm, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

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

OOF_XGB_PATH = DATA_DIR / "oof_xgb_5fold.csv"
OOF_XGB_TUNED_PATH = DATA_DIR / "oof_xgb_tuned_5fold.csv"
SUB_XGB_PATH = DATA_DIR / "submission_xgb_5fold.csv"
SUB_XGB_TUNED_PATH = DATA_DIR / "submission_xgb_tuned_5fold.csv"
SUBMISSION_PATH = DATA_DIR / "submission_equal_blend.csv"

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def clean_category_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("Missing")


def unify_categorical_levels(*frames: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLS:
        categories = sorted(
            set().union(
                *[
                    set(clean_category_series(frame[column]).unique())
                    for frame in frames
                ]
            )
        )
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[column] = clean_category_series(frame[column]).astype(dtype)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    print("Loading competition train/test...", flush=True)
    train = add_features(pd.read_csv(TRAIN_PATH))
    test = add_features(pd.read_csv(TEST_PATH))

    y = train[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    train_x = train[FEATURE_COLS].copy()
    test_x = test[FEATURE_COLS].copy()
    unify_categorical_levels(train_x, test_x)

    return train_x, y, test[[ID_COL]].join(test_x)


def build_lgbm_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=3000,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def run_native_lgbm_cv(
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame,
    folds: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n=== Recomputing native-categorical LightGBM ===", flush=True)
    oof_pred = np.zeros(len(train_x))
    test_pred = np.zeros(len(test_x))

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, y), start=1):
        x_train = train_x.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        x_valid = train_x.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()
        x_test = test_x.copy()
        unify_categorical_levels(x_train, x_valid, x_test)

        print(
            f"LightGBM fold {fold}: train rows={len(x_train):,}, "
            f"validation rows={len(x_valid):,}",
            flush=True,
        )

        model = build_lgbm_model()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[
                early_stopping(stopping_rounds=100, verbose=False),
                log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        oof_pred[valid_idx] = valid_pred
        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS
        print(
            f"LightGBM fold {fold} ROC AUC: {fold_auc:.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    print(f"Native LightGBM OOF AUC: {roc_auc_score(y, oof_pred):.6f}", flush=True)
    return oof_pred, test_pred


def recipe_score(df: pd.DataFrame) -> np.ndarray:
    subsidy_yes = (
        clean_category_series(df["Subsidy_Available"]).eq("Yes").to_numpy(dtype=float)
    )
    range_medium = (
        clean_category_series(df["Range_Anxiety_Level"])
        .eq("Medium")
        .to_numpy(dtype=float)
    )
    range_high = (
        clean_category_series(df["Range_Anxiety_Level"]).eq("High").to_numpy(dtype=float)
    )

    return (
        1.2 * df["Annual_Income_USD"].to_numpy(dtype=float) / 1e5
        + 0.6 * df["Environmental_Concern_Level"].to_numpy(dtype=float)
        + 2.0 * subsidy_yes
        - 1.0 * range_medium
        - 3.0 * range_high
    )


def recipe_base_margin(df: pd.DataFrame) -> np.ndarray:
    probability = np.clip(norm.cdf(recipe_score(df) - 5.5), 1e-6, 1 - 1e-6)
    return logit(probability)


def build_xgb_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", "passthrough", NUMERIC_COLS),
        ]
    )


def run_basemargin_xgb_cv(
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame,
    folds: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n=== Training recipe base-margin XGBoost ===", flush=True)
    oof_pred = np.zeros(len(train_x))
    test_pred = np.zeros(len(test_x))
    margin_test = recipe_base_margin(test_x)

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, y), start=1):
        x_train = train_x.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        x_valid = train_x.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        margin_train = recipe_base_margin(x_train)
        margin_valid = recipe_base_margin(x_valid)

        preprocessor = build_xgb_preprocessor()
        x_train_encoded = preprocessor.fit_transform(x_train)
        x_valid_encoded = preprocessor.transform(x_valid)
        x_test_encoded = preprocessor.transform(test_x)

        print(
            f"Base-margin XGB fold {fold}: train rows={len(x_train):,}, "
            f"validation rows={len(x_valid):,}",
            flush=True,
        )

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(
            x_train_encoded,
            y_train,
            base_margin=margin_train,
            eval_set=[(x_valid_encoded, y_valid)],
            base_margin_eval_set=[margin_valid],
            verbose=False,
        )

        valid_pred = model.predict_proba(
            x_valid_encoded,
            base_margin=margin_valid,
        )[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        oof_pred[valid_idx] = valid_pred
        test_pred += (
            model.predict_proba(x_test_encoded, base_margin=margin_test)[:, 1]
            / N_SPLITS
        )
        print(f"Base-margin XGB fold {fold} ROC AUC: {fold_auc:.6f}", flush=True)

    print(f"Base-margin XGB OOF AUC: {roc_auc_score(y, oof_pred):.6f}", flush=True)
    return oof_pred, test_pred


def load_saved_predictions(
    test_ids: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\nLoading saved XGBoost OOF/test predictions...", flush=True)
    oof_xgb = pd.read_csv(OOF_XGB_PATH)
    oof_xgb_tuned = pd.read_csv(OOF_XGB_TUNED_PATH)
    sub_xgb = pd.read_csv(SUB_XGB_PATH)
    sub_xgb_tuned = pd.read_csv(SUB_XGB_TUNED_PATH)

    if not test_ids[ID_COL].equals(sub_xgb[ID_COL]) or not test_ids[ID_COL].equals(
        sub_xgb_tuned[ID_COL]
    ):
        raise ValueError("Saved XGBoost submission IDs do not match test IDs.")

    oof_xgb = oof_xgb.sort_values(ID_COL)
    oof_xgb_tuned = oof_xgb_tuned.sort_values(ID_COL)

    return (
        oof_xgb["xgb_5fold_pred"].to_numpy(),
        sub_xgb[TARGET].to_numpy(),
        oof_xgb_tuned["xgb_tuned_5fold_pred"].to_numpy(),
        sub_xgb_tuned[TARGET].to_numpy(),
    )


def minmax_scale(values: np.ndarray) -> np.ndarray:
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def rank_average(predictions: pd.DataFrame) -> np.ndarray:
    ranked = predictions.apply(lambda column: rankdata(column, method="average"))
    averaged = ranked.mean(axis=1).to_numpy(dtype=float)
    return minmax_scale(averaged)


def evaluate_and_write_best(
    y: pd.Series,
    test_ids: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    print("\n=== Single Model OOF AUCs ===", flush=True)
    for column in oof_predictions.columns:
        print(f"{column}: {roc_auc_score(y, oof_predictions[column]):.6f}", flush=True)

    print("\n=== Model Correlation Matrix ===", flush=True)
    print(oof_predictions.corr().round(6).to_string(), flush=True)

    blend_a_oof = oof_predictions[["lgb_native", "xgb", "xgb_tuned"]].mean(axis=1)
    blend_a_test = test_predictions[["lgb_native", "xgb", "xgb_tuned"]].mean(axis=1)

    blend_b_oof = oof_predictions[
        ["lgb_native", "xgb", "xgb_tuned", "xgb_base_margin"]
    ].mean(axis=1)
    blend_b_test = test_predictions[
        ["lgb_native", "xgb", "xgb_tuned", "xgb_base_margin"]
    ].mean(axis=1)

    blend_c_oof = rank_average(
        oof_predictions[["lgb_native", "xgb", "xgb_tuned", "xgb_base_margin"]]
    )
    blend_c_test = rank_average(
        test_predictions[["lgb_native", "xgb", "xgb_tuned", "xgb_base_margin"]]
    )

    blends = {
        "equal_3_lgb_xgb_tuned": (blend_a_oof.to_numpy(), blend_a_test.to_numpy()),
        "equal_4_plus_base_margin": (
            blend_b_oof.to_numpy(),
            blend_b_test.to_numpy(),
        ),
        "rank_average_equal_4": (blend_c_oof, blend_c_test),
    }

    print("\n=== Equal Blend OOF AUCs ===", flush=True)
    scores = {}
    for name, (oof_pred, _) in blends.items():
        scores[name] = roc_auc_score(y, oof_pred)
        print(f"{name}: {scores[name]:.6f}", flush=True)

    best_name = max(scores, key=scores.get)
    best_auc = scores[best_name]
    best_test_pred = blends[best_name][1]

    submission = pd.DataFrame(
        {
            ID_COL: test_ids[ID_COL],
            TARGET: best_test_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\n=== Best Blend ===", flush=True)
    print(f"Winner: {best_name}", flush=True)
    print(f"OOF AUC: {best_auc:.6f}", flush=True)
    print(f"Wrote {SUBMISSION_PATH.name}", flush=True)
    print(f"Submission rows: {len(submission):,}", flush=True)
    print(f"Submission columns: {list(submission.columns)}", flush=True)
    print(
        f"Probabilities in [0, 1]: {submission[TARGET].between(0, 1).all()}",
        flush=True,
    )
    print(submission[TARGET].describe().to_string(), flush=True)


def main() -> None:
    train_x, y, test = load_data()
    test_ids = test[[ID_COL]].copy()
    test_x = test[FEATURE_COLS].copy()

    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    xgb_oof, xgb_test, xgb_tuned_oof, xgb_tuned_test = load_saved_predictions(test_ids)
    lgb_oof, lgb_test = run_native_lgbm_cv(train_x, y, test_x, folds)
    bm_oof, bm_test = run_basemargin_xgb_cv(train_x, y, test_x, folds)

    oof_predictions = pd.DataFrame(
        {
            "lgb_native": lgb_oof,
            "xgb": xgb_oof,
            "xgb_tuned": xgb_tuned_oof,
            "xgb_base_margin": bm_oof,
        }
    )
    test_predictions = pd.DataFrame(
        {
            "lgb_native": lgb_test,
            "xgb": xgb_test,
            "xgb_tuned": xgb_tuned_test,
            "xgb_base_margin": bm_test,
        }
    )

    evaluate_and_write_best(y, test_ids, oof_predictions, test_predictions)


if __name__ == "__main__":
    main()
