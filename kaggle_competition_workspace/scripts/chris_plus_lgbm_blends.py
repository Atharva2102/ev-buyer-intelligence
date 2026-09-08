from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID_COL = "id"

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"

SUBMISSION_PATH = "submission_chris_plus_lgbm_blend.csv"
CHRIS_OOF_PATH = "oof_chris_xgb_three_models.csv"
LGBM_CHRIS_OOF_PATH = "oof_lgbm_chris_features.csv"


def make_chris_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns]).copy()
    home = (df["Home_Charging_Possible"] == "Yes").astype(int)
    subsidy = (df["Subsidy_Available"] == "Yes").astype(int)

    x["worry_score"] = (
        df["Daily_Commute_km"]
        - 5 * df["Charging_Stations_Near_Home"]
        - 5 * df["Charging_Stations_Near_Work"]
        - 150 * home
    )
    x["chargers_total"] = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    x["income_x_subsidy"] = df["Annual_Income_USD"] / 1e5 * subsidy
    x["concern_x_subsidy"] = df["Environmental_Concern_Level"] * subsidy
    x["recipe_score"] = recipe_score(df)
    return x


def recipe_score(df: pd.DataFrame) -> np.ndarray:
    return (
        1.2 * df["Annual_Income_USD"] / 1e5
        + 0.6 * df["Environmental_Concern_Level"]
        + 2.0 * (df["Subsidy_Available"] == "Yes")
        - 1.0 * (df["Range_Anxiety_Level"] == "Medium")
        - 3.0 * (df["Range_Anxiety_Level"] == "High")
    ).to_numpy()


def recipe_logit(df: pd.DataFrame) -> np.ndarray:
    probability = np.clip(norm.cdf(recipe_score(df) - 5.5), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def encode_xgb_categories(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_x.copy()
    test_x = test_x.copy()
    for column in train_x.select_dtypes("object").columns:
        categories = pd.Categorical(pd.concat([train_x[column], test_x[column]])).categories
        train_x[column] = pd.Categorical(train_x[column], categories=categories).codes
        test_x[column] = pd.Categorical(test_x[column], categories=categories).codes
    return train_x, test_x


def clean_category_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("Missing")


def unify_lgbm_categories(*frames: pd.DataFrame) -> list[str]:
    cat_cols = list(frames[0].select_dtypes("object").columns)
    for column in cat_cols:
        categories = sorted(
            set().union(
                *[set(clean_category_series(frame[column]).unique()) for frame in frames]
            )
        )
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[column] = clean_category_series(frame[column]).astype(dtype)
    return cat_cols


def get_folds(x: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(
        StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(x, y)
    )


def run_xgb(
    name: str,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    margin_train: np.ndarray | None = None,
    margin_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    params = {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "auc",
        "early_stopping_rounds": 100,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test_x))
    rounds = []
    start = time.time()

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        model = XGBClassifier(**params)
        if margin_train is None:
            model.fit(
                train_x.iloc[train_idx],
                y[train_idx],
                eval_set=[(train_x.iloc[valid_idx], y[valid_idx])],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
            test_pred += model.predict_proba(test_x)[:, 1] / N_SPLITS
        else:
            model.fit(
                train_x.iloc[train_idx],
                y[train_idx],
                eval_set=[(train_x.iloc[valid_idx], y[valid_idx])],
                base_margin=margin_train[train_idx],
                base_margin_eval_set=[margin_train[valid_idx]],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(
                train_x.iloc[valid_idx],
                base_margin=margin_train[valid_idx],
            )[:, 1]
            test_pred += (
                model.predict_proba(test_x, base_margin=margin_test)[:, 1] / N_SPLITS
            )
        rounds.append(int(model.best_iteration))

    fold_aucs = [roc_auc_score(y[valid_idx], oof[valid_idx]) for _, valid_idx in folds]
    print(
        f"{name}: OOF AUC={roc_auc_score(y, oof):.6f}, "
        f"folds={[round(score, 6) for score in fold_aucs]}, "
        f"rounds={rounds}, seconds={time.time() - start:.0f}",
        flush=True,
    )
    return oof, test_pred


def run_lgbm_chris_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
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

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test_x))
    rounds = []
    start = time.time()

    print("\n=== Native LightGBM with Chris features ===", flush=True)
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        x_train = train_x.iloc[train_idx].copy()
        x_valid = train_x.iloc[valid_idx].copy()
        x_test = test_x.copy()
        cat_cols = unify_lgbm_categories(x_train, x_valid, x_test)

        model = LGBMClassifier(**params)
        model.fit(
            x_train,
            y[train_idx],
            eval_set=[(x_valid, y[valid_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                early_stopping(stopping_rounds=100, verbose=False),
                log_evaluation(period=0),
            ],
        )

        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1] / N_SPLITS
        rounds.append(int(model.best_iteration_))
        print(
            f"LGBM Chris fold {fold}: AUC={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    fold_aucs = [roc_auc_score(y[valid_idx], oof[valid_idx]) for _, valid_idx in folds]
    print(
        f"LGBM Chris features: OOF AUC={roc_auc_score(y, oof):.6f}, "
        f"folds={[round(score, 6) for score in fold_aucs]}, "
        f"rounds={rounds}, seconds={time.time() - start:.0f}",
        flush=True,
    )
    return oof, test_pred


def rank_average(predictions: pd.DataFrame) -> np.ndarray:
    ranked = predictions.apply(lambda column: rankdata(column, method="average"))
    averaged = ranked.mean(axis=1).to_numpy(dtype=float)
    return (averaged - averaged.min()) / (averaged.max() - averaged.min())


def evaluate_blends(
    y: np.ndarray,
    test_ids: pd.Series,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
) -> None:
    print("\n=== Single Model AUCs ===", flush=True)
    for column in oof.columns:
        print(f"{column}: {roc_auc_score(y, oof[column]):.6f}", flush=True)

    print("\n=== Correlation Matrix ===", flush=True)
    print(oof.corr().round(6).to_string(), flush=True)

    blend_specs = {
        "equal_chris_3": ["chris_m1", "chris_m2_base_margin", "chris_m3_recipe_feature"],
        "equal_chris_3_lgbm": [
            "chris_m1",
            "chris_m2_base_margin",
            "chris_m3_recipe_feature",
            "lgbm_chris_features",
        ],
        "equal_chris_3_lgbm_native": [
            "chris_m1",
            "chris_m2_base_margin",
            "chris_m3_recipe_feature",
            "lgbm_chris_features",
            "lgbm_native_previous",
        ],
        "rank_chris_3_lgbm": [
            "chris_m1",
            "chris_m2_base_margin",
            "chris_m3_recipe_feature",
            "lgbm_chris_features",
        ],
    }

    candidates = {}
    print("\n=== Blend AUCs ===", flush=True)
    for name, columns in blend_specs.items():
        if name.startswith("rank_"):
            oof_blend = rank_average(oof[columns])
            test_blend = rank_average(test_pred[columns])
        else:
            oof_blend = oof[columns].mean(axis=1).to_numpy()
            test_blend = test_pred[columns].mean(axis=1).to_numpy()
        score = roc_auc_score(y, oof_blend)
        candidates[name] = (score, test_blend)
        print(f"{name}: {score:.6f}", flush=True)

    best_name, (best_score, best_test) = max(candidates.items(), key=lambda item: item[1][0])
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: best_test})
    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\n=== Winner ===", flush=True)
    print(f"Best blend: {best_name}", flush=True)
    print(f"Best OOF AUC: {best_score:.6f}", flush=True)
    print(f"Wrote {SUBMISSION_PATH}", flush=True)
    print(f"Valid probabilities: {submission[TARGET].between(0, 1).all()}", flush=True)
    print(submission[TARGET].describe().to_string(), flush=True)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = (train[TARGET] == "Yes").astype(int).to_numpy()
    test_ids = pd.read_csv(SAMPLE_SUBMISSION_PATH)[ID_COL]

    chris_train_raw = make_chris_features(train)
    chris_test_raw = make_chris_features(test)
    xgb_train, xgb_test = encode_xgb_categories(chris_train_raw, chris_test_raw)
    folds = get_folds(xgb_train, y)

    score_train = recipe_score(train)
    score_test = recipe_score(test)
    margin_train = recipe_logit(train)
    margin_test = recipe_logit(test)

    print(f"Recipe alone train AUC: {roc_auc_score(y, score_train):.6f}", flush=True)

    chris_m1_oof, chris_m1_test = run_xgb(
        "Chris M1 baseline", xgb_train.drop(columns=["recipe_score"]), xgb_test.drop(columns=["recipe_score"]), y, folds
    )
    chris_m2_oof, chris_m2_test = run_xgb(
        "Chris M2 base margin",
        xgb_train.drop(columns=["recipe_score"]),
        xgb_test.drop(columns=["recipe_score"]),
        y,
        folds,
        margin_train=margin_train,
        margin_test=margin_test,
    )
    chris_m3_oof, chris_m3_test = run_xgb(
        "Chris M3 recipe feature", xgb_train, xgb_test, y, folds
    )

    lgbm_chris_oof, lgbm_chris_test = run_lgbm_chris_features(
        chris_train_raw, chris_test_raw, y, folds
    )

    native_lgbm_oof = pd.read_csv("submission_lightgbm_5fold.csv")
    previous_lgbm_oof = pd.read_csv("oof_lightgbm_5fold.csv")

    oof = pd.DataFrame(
        {
            "chris_m1": chris_m1_oof,
            "chris_m2_base_margin": chris_m2_oof,
            "chris_m3_recipe_feature": chris_m3_oof,
            "lgbm_chris_features": lgbm_chris_oof,
            "lgbm_native_previous": previous_lgbm_oof["lightgbm_5fold_pred"].to_numpy(),
        }
    )
    test_pred = pd.DataFrame(
        {
            "chris_m1": chris_m1_test,
            "chris_m2_base_margin": chris_m2_test,
            "chris_m3_recipe_feature": chris_m3_test,
            "lgbm_chris_features": lgbm_chris_test,
            "lgbm_native_previous": native_lgbm_oof[TARGET].to_numpy(),
        }
    )

    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, **oof.to_dict("series")}).to_csv(
        CHRIS_OOF_PATH,
        index=False,
    )
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "lgbm_chris_features": lgbm_chris_oof,
        }
    ).to_csv(LGBM_CHRIS_OOF_PATH, index=False)
    print(f"\nWrote {CHRIS_OOF_PATH}", flush=True)
    print(f"Wrote {LGBM_CHRIS_OOF_PATH}", flush=True)

    evaluate_blends(y, test_ids, oof, test_pred)


if __name__ == "__main__":
    main()
