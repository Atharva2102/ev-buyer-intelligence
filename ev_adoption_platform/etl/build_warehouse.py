from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TARGET = "Will_Buy_EV"
ID = "id"

FEATURE_COLUMNS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def platform_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_required_file(root: Path, relative_candidates: list[str], filename: str) -> Path:
    for relative in relative_candidates:
        path = root / relative
        if path.exists():
            return path
    matches = sorted(
        path for path in root.rglob(filename) if "__pycache__" not in path.parts
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find required file: {filename}")


def load_csvs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = find_required_file(root, ["train.csv", "kaggle_competition_workspace/data/train.csv"], "train.csv")
    test_path = find_required_file(root, ["test.csv", "kaggle_competition_workspace/data/test.csv"], "test.csv")
    original_path = find_required_file(
        root,
        [
            "OG Dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv",
            "kaggle_competition_workspace/data/OG Dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv",
        ],
        "EV_Adoption_and_Range_Anxiety_Dataset.csv",
    )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    original = pd.read_csv(original_path)

    print(f"Loaded original dataset: {original_path}")
    print("Original columns:", list(original.columns))
    return train, test, original


def normalize_original(original: pd.DataFrame) -> pd.DataFrame:
    original = original.copy()
    if "Buyer_ID" in original.columns:
        original = original.rename(columns={"Buyer_ID": ID})

    original[TARGET] = original[TARGET].map({"No": 0, "Yes": 1}).astype("Int64")
    missing = [column for column in FEATURE_COLUMNS if column not in original.columns]
    print("Missing competition feature columns in original:", missing)
    for column in missing:
        original[column] = np.nan

    return original[[ID, *FEATURE_COLUMNS, TARGET]]


def normalize_competition(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    if train[TARGET].dtype == object:
        train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    return train[[ID, *FEATURE_COLUMNS, TARGET]], test[[ID, *FEATURE_COLUMNS]]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["charging_total"] = out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    out["charging_gap_home_minus_work"] = (
        out["Charging_Stations_Near_Home"] - out["Charging_Stations_Near_Work"]
    )
    out["charging_per_commute_km"] = out["charging_total"] / (out["Daily_Commute_km"].fillna(0) + 1)
    out["income_per_car"] = out["Annual_Income_USD"] / (out["Number_of_Cars_Owned"].fillna(0) + 1)
    out["has_home_charging"] = (out["Home_Charging_Possible"] == "Yes").astype(int)
    out["has_subsidy"] = (out["Subsidy_Available"] == "Yes").astype(int)
    out["high_range_anxiety"] = (out["Range_Anxiety_Level"] == "High").astype(int)
    out["city_home_charging_segment"] = (
        out["City_Type"].fillna("Missing").astype(str)
        + " | Home Charging: "
        + out["Home_Charging_Possible"].fillna("Missing").astype(str)
    )
    out["subsidy_anxiety_segment"] = (
        "Subsidy: "
        + out["Subsidy_Available"].fillna("Missing").astype(str)
        + " | Anxiety: "
        + out["Range_Anxiety_Level"].fillna("Missing").astype(str)
    )
    out["buyer_segment"] = (
        out["City_Type"].fillna("Missing").astype(str)
        + " | "
        + out["Current_Car_Type"].fillna("Missing").astype(str)
        + " | "
        + out["Range_Anxiety_Level"].fillna("Missing").astype(str)
    )
    return out


def choose_score_file(root: Path) -> Path:
    candidates = [
        "submission_tabm_rank_average.csv",
        "submission_tabm_colab.csv",
        "submission_chris_lgbm_tuned_blend.csv",
        "submission_chris_plus_lgbm_blend.csv",
        "submission_chris_xgb_starter_reproduction.csv",
        "submission_equal_blend.csv",
    ]
    for filename in candidates:
        paths = [
            root / filename,
            root / "kaggle_competition_workspace" / "submissions" / "top_10" / filename,
            root / "kaggle_competition_workspace" / "submissions" / filename,
        ]
        for path in paths:
            if path.exists():
                print(f"Using model score file: {path.name}")
                return path
        matches = sorted(
            path for path in root.rglob(filename) if "__pycache__" not in path.parts
        )
        if matches:
            print(f"Using model score file: {matches[0].name}")
            return matches[0]
    raise FileNotFoundError("No supported submission score file found in repository root.")


def add_score_bands(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["adoption_band"] = pd.cut(
        out["ev_purchase_probability"],
        bins=[-0.001, 0.15, 0.35, 0.65, 1.001],
        labels=["Low", "Watch", "High", "Priority"],
    ).astype(str)
    out["is_high_intent"] = (out["ev_purchase_probability"] >= 0.35).astype(int)
    return out


def build_segment_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    overall = scored["ev_purchase_probability"].mean()
    frames = []
    for key in [
        "buyer_segment",
        "city_home_charging_segment",
        "subsidy_anxiety_segment",
        "City_Type",
        "Current_Car_Type",
        "Range_Anxiety_Level",
    ]:
        grouped = (
            scored.groupby(key, dropna=False)
            .agg(
                population=(ID, "count"),
                avg_probability=("ev_purchase_probability", "mean"),
                high_intent_buyers=("is_high_intent", "sum"),
                avg_income=("Annual_Income_USD", "mean"),
                avg_commute_km=("Daily_Commute_km", "mean"),
            )
            .reset_index()
            .rename(columns={key: "segment"})
        )
        grouped["segment_type"] = key
        grouped["lift_vs_average"] = grouped["avg_probability"] - overall
        frames.append(grouped)

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.sort_values(["segment_type", "avg_probability"], ascending=[True, False])
    metrics["rank_in_type"] = metrics.groupby("segment_type")["avg_probability"].rank(
        method="first", ascending=False
    )
    return metrics


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expit(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def build_policy_simulation(scored: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Subsidy for customers without current subsidy": (
            (scored["Subsidy_Available"] != "Yes"),
            0.34,
        ),
        "Home charging access for customers without it": (
            (scored["Home_Charging_Possible"] != "Yes"),
            0.24,
        ),
        "Range anxiety education for medium/high anxiety": (
            scored["Range_Anxiety_Level"].isin(["Medium", "High"]),
            0.18,
        ),
        "Charging network expansion for low charger access": (
            (scored["charging_total"] <= scored["charging_total"].median()),
            0.14,
        ),
    }

    rows = []
    base_total = scored["ev_purchase_probability"].sum()
    for scenario, (mask, log_odds_lift) in scenarios.items():
        adjusted = scored["ev_purchase_probability"].copy()
        adjusted.loc[mask] = expit(logit(adjusted.loc[mask]) + log_odds_lift)
        rows.append(
            {
                "scenario": scenario,
                "affected_customers": int(mask.sum()),
                "baseline_expected_buyers": float(base_total),
                "scenario_expected_buyers": float(adjusted.sum()),
                "incremental_expected_buyers": float(adjusted.sum() - base_total),
                "avg_probability_lift": float(adjusted.mean() - scored["ev_purchase_probability"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("incremental_expected_buyers", ascending=False)


def build_data_quality(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, df in datasets.items():
        for column in df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column_name": column,
                    "row_count": len(df),
                    "missing_count": int(df[column].isna().sum()),
                    "missing_rate": float(df[column].isna().mean()),
                    "unique_count": int(df[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def build_drift_metrics(original: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_COLUMNS:
        rows.append(
            {
                "column_name": column,
                "metric_type": "numeric_mean_delta",
                "original_value": float(original[column].mean()),
                "competition_value": float(train[column].mean()),
                "absolute_delta": float(abs(original[column].mean() - train[column].mean())),
            }
        )

    for column in CATEGORICAL_COLUMNS:
        original_dist = original[column].fillna("Missing").value_counts(normalize=True)
        train_dist = train[column].fillna("Missing").value_counts(normalize=True)
        categories = sorted(set(original_dist.index) | set(train_dist.index))
        l1 = sum(abs(float(original_dist.get(category, 0)) - float(train_dist.get(category, 0))) for category in categories)
        rows.append(
            {
                "column_name": column,
                "metric_type": "categorical_distribution_l1",
                "original_value": np.nan,
                "competition_value": np.nan,
                "absolute_delta": float(l1),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_delta", ascending=False)


def write_warehouse(tables: dict[str, pd.DataFrame], db_path: Path, marts_dir: Path) -> None:
    marts_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with duckdb.connect(str(db_path)) as con:
        for name, table in tables.items():
            con.register(name, table)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}")
            table.to_csv(marts_dir / f"{name}.csv", index=False)
            print(f"Wrote {name}: {len(table):,} rows")


def main() -> None:
    root = repo_root()
    out_root = platform_root() / "warehouse"
    db_path = out_root / "ev_adoption.duckdb"
    marts_dir = out_root / "marts"

    train_raw, test_raw, original_raw = load_csvs(root)
    train = add_derived_features(normalize_competition(train_raw, test_raw)[0])
    test = add_derived_features(normalize_competition(train_raw, test_raw)[1])
    original = add_derived_features(normalize_original(original_raw))

    score_file = choose_score_file(root)
    scores = pd.read_csv(score_file)[[ID, TARGET]].rename(columns={TARGET: "ev_purchase_probability"})
    scored_current = test.merge(scores, on=ID, how="left")
    if scored_current["ev_purchase_probability"].isna().any():
        raise ValueError("Score file does not cover every test id.")
    scored_current = add_score_bands(scored_current)
    scored_current["score_source"] = score_file.name

    historical_adoption = original.copy()
    historical_adoption["source_system"] = "original_ev_adoption_survey"

    training_reference = train.copy()
    training_reference["source_system"] = "competition_train"

    segment_metrics = build_segment_metrics(scored_current)
    policy_simulation = build_policy_simulation(scored_current)
    data_quality = build_data_quality(
        {
            "competition_train": train,
            "competition_test": test,
            "original_reference": original,
            "scored_current": scored_current,
        }
    )
    drift_metrics = build_drift_metrics(original, train)

    overview_metrics = pd.DataFrame(
        [
            {
                "metric_name": "customers_scored",
                "metric_value": float(len(scored_current)),
                "metric_label": f"{len(scored_current):,}",
            },
            {
                "metric_name": "avg_ev_purchase_probability",
                "metric_value": float(scored_current["ev_purchase_probability"].mean()),
                "metric_label": f"{scored_current['ev_purchase_probability'].mean():.2%}",
            },
            {
                "metric_name": "high_intent_buyers",
                "metric_value": float(scored_current["is_high_intent"].sum()),
                "metric_label": f"{scored_current['is_high_intent'].sum():,}",
            },
            {
                "metric_name": "historical_adoption_rate",
                "metric_value": float(historical_adoption[TARGET].mean()),
                "metric_label": f"{historical_adoption[TARGET].mean():.2%}",
            },
        ]
    )

    write_warehouse(
        {
            "historical_adoption": historical_adoption,
            "training_reference": training_reference,
            "scored_current_customers": scored_current,
            "mart_segment_metrics": segment_metrics,
            "mart_policy_simulation": policy_simulation,
            "mart_data_quality": data_quality,
            "mart_drift_metrics": drift_metrics,
            "mart_overview_metrics": overview_metrics,
        },
        db_path,
        marts_dir,
    )
    print(f"Warehouse ready: {db_path}")


if __name__ == "__main__":
    main()
