from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import duckdb
import pandas as pd

from etl.build_warehouse import (
    ID,
    TARGET,
    add_derived_features,
    add_score_bands,
    build_data_quality,
    build_drift_metrics,
    build_policy_simulation,
    build_segment_metrics,
    normalize_competition,
    normalize_original,
)


REQUIRED_OBJECTS = {
    "train": "train.csv",
    "test": "test.csv",
    "original": "original_ev_adoption.csv",
}

SNOWFLAKE_TABLES = {
    "historical_adoption": ("STAGING", "HISTORICAL_ADOPTION"),
    "training_reference": ("STAGING", "TRAINING_REFERENCE"),
    "scored_current_customers": ("SERVING", "SCORED_CURRENT_CUSTOMERS"),
    "mart_segment_metrics": ("MART", "SEGMENT_METRICS"),
    "mart_policy_simulation": ("MART", "POLICY_SIMULATION"),
    "mart_data_quality": ("MART", "DATA_QUALITY"),
    "mart_drift_metrics": ("MART", "DRIFT_METRICS"),
    "mart_overview_metrics": ("MART", "OVERVIEW_METRICS"),
}


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


def object_key(prefix: str, filename: str) -> str:
    return f"{prefix.strip('/')}/{filename}"


def download_inputs(
    s3: Any,
    bucket: str,
    raw_prefix: str,
    scores_key: str,
    workdir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    objects = {
        **{name: object_key(raw_prefix, filename) for name, filename in REQUIRED_OBJECTS.items()},
        "scores": scores_key,
    }
    for name, key in objects.items():
        destination = workdir / Path(key).name
        print(f"Downloading s3://{bucket}/{key}")
        s3.download_file(bucket, key, str(destination))
        paths[name] = destination
    return paths


def build_tables(paths: dict[str, Path], score_source: str) -> dict[str, pd.DataFrame]:
    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    original_raw = pd.read_csv(paths["original"])
    scores = pd.read_csv(paths["scores"])

    train_normalized, test_normalized = normalize_competition(train_raw, test_raw)
    train = add_derived_features(train_normalized)
    test = add_derived_features(test_normalized)
    original = add_derived_features(normalize_original(original_raw))

    required_score_columns = {ID, TARGET}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file is missing columns: {sorted(missing_score_columns)}")
    if scores[ID].duplicated().any():
        raise ValueError("Score file contains duplicate ids.")

    scores = scores[[ID, TARGET]].rename(columns={TARGET: "ev_purchase_probability"})
    probabilities = pd.to_numeric(scores["ev_purchase_probability"], errors="coerce")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Score probabilities must be numeric values between 0 and 1.")
    scores["ev_purchase_probability"] = probabilities

    scored_current = test.merge(scores, on=ID, how="left", validate="one_to_one")
    if scored_current["ev_purchase_probability"].isna().any():
        missing = int(scored_current["ev_purchase_probability"].isna().sum())
        raise ValueError(f"Score file does not cover {missing:,} test ids.")
    scored_current = add_score_bands(scored_current)
    scored_current["score_source"] = score_source

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

    return {
        "historical_adoption": historical_adoption,
        "training_reference": training_reference,
        "scored_current_customers": scored_current,
        "mart_segment_metrics": segment_metrics,
        "mart_policy_simulation": policy_simulation,
        "mart_data_quality": data_quality,
        "mart_drift_metrics": drift_metrics,
        "mart_overview_metrics": overview_metrics,
    }


def write_artifacts(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    db_path = output_dir / "ev_adoption.duckdb"

    with duckdb.connect(str(db_path)) as connection:
        for table_name, frame in tables.items():
            parquet_path = output_dir / f"{table_name}.parquet"
            frame.to_parquet(parquet_path, index=False)
            connection.register("source_frame", frame)
            connection.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM source_frame')
            connection.unregister("source_frame")
            artifacts[table_name] = parquet_path
            print(f"Built {table_name}: {len(frame):,} rows")

    artifacts["warehouse"] = db_path
    return artifacts


def upload_artifacts(
    s3: Any,
    bucket: str,
    curated_prefix: str,
    run_id: str,
    artifacts: dict[str, Path],
) -> dict[str, dict[str, str]]:
    uploaded: dict[str, dict[str, str]] = {}
    for name, path in artifacts.items():
        latest_key = object_key(f"{curated_prefix}/latest", path.name)
        run_key = object_key(f"{curated_prefix}/runs/{run_id}", path.name)
        extra = {"ServerSideEncryption": "AES256"}
        s3.upload_file(str(path), bucket, run_key, ExtraArgs=extra)
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": run_key},
            Key=latest_key,
            ServerSideEncryption="AES256",
        )
        uploaded[name] = {"run": f"s3://{bucket}/{run_key}", "latest": f"s3://{bucket}/{latest_key}"}
    return uploaded


def load_secret(secret_arn: str) -> dict[str, str]:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def load_snowflake(tables: dict[str, pd.DataFrame], secret_arn: str) -> list[str]:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    secret = load_secret(secret_arn)
    required = {"account", "user", "password", "warehouse", "database"}
    missing = required - set(secret)
    if missing:
        raise ValueError(f"Snowflake secret is missing keys: {sorted(missing)}")

    connection = snowflake.connector.connect(
        account=secret["account"],
        user=secret["user"],
        password=secret["password"],
        warehouse=secret["warehouse"],
        database=secret["database"],
        role=secret.get("role"),
    )
    loaded: list[str] = []
    try:
        for schema in {schema for schema, _ in SNOWFLAKE_TABLES.values()}:
            connection.cursor().execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        for frame_name, (schema, table_name) in SNOWFLAKE_TABLES.items():
            frame = tables[frame_name].copy()
            frame.columns = [column.upper() for column in frame.columns]
            success, _, row_count, _ = write_pandas(
                connection,
                frame,
                table_name=table_name,
                database=secret["database"],
                schema=schema,
                auto_create_table=True,
                overwrite=True,
                quote_identifiers=False,
            )
            if not success or row_count != len(frame):
                raise RuntimeError(f"Snowflake load failed for {schema}.{table_name}")
            loaded.append(f"{schema}.{table_name}")
            print(f"Loaded Snowflake table {schema}.{table_name}: {row_count:,} rows")
    finally:
        connection.close()
    return loaded


def upload_manifest(s3: Any, bucket: str, key: str, manifest: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def run() -> dict[str, Any]:
    bucket = required_env("DATA_BUCKET")
    raw_prefix = os.getenv("RAW_PREFIX", "raw")
    curated_prefix = os.getenv("CURATED_PREFIX", "curated")
    scores_key = os.getenv("SCORES_KEY", object_key(raw_prefix, "scores.csv"))
    enable_snowflake = os.getenv("ENABLE_SNOWFLAKE", "false").lower() == "true"
    secret_arn = os.getenv("SNOWFLAKE_SECRET_ARN", "").strip()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started_at = datetime.now(timezone.utc)
    s3 = boto3.client("s3")
    manifest_key = object_key(f"{curated_prefix}/runs/{run_id}", "manifest.json")
    workdir = Path(tempfile.mkdtemp(prefix="ev-pipeline-"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "pipeline_version": os.getenv("PIPELINE_VERSION", "local"),
        "status": "running",
        "started_at": started_at.isoformat(),
        "inputs": {},
        "outputs": {},
        "snowflake_tables": [],
    }

    try:
        paths = download_inputs(s3, bucket, raw_prefix, scores_key, workdir)
        manifest["inputs"] = {
            name: f"s3://{bucket}/{scores_key if name == 'scores' else object_key(raw_prefix, REQUIRED_OBJECTS[name])}"
            for name in paths
        }
        tables = build_tables(paths, Path(scores_key).name)
        artifacts = write_artifacts(tables, workdir / "artifacts")
        manifest["outputs"] = upload_artifacts(
            s3, bucket, curated_prefix, run_id, artifacts
        )
        manifest["row_counts"] = {name: len(frame) for name, frame in tables.items()}

        if enable_snowflake:
            if not secret_arn:
                raise ValueError("SNOWFLAKE_SECRET_ARN is required when ENABLE_SNOWFLAKE=true")
            manifest["snowflake_tables"] = load_snowflake(tables, secret_arn)

        manifest["status"] = "succeeded"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = (
            datetime.now(timezone.utc) - started_at
        ).total_seconds()
        upload_manifest(s3, bucket, manifest_key, manifest)
        upload_manifest(
            s3,
            bucket,
            object_key(f"{curated_prefix}/latest", "manifest.json"),
            manifest,
        )
        print(json.dumps(manifest, indent=2, default=str))
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        try:
            upload_manifest(s3, bucket, manifest_key, manifest)
        except Exception as manifest_error:
            print(f"Could not upload failure manifest: {manifest_error}")
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    run()
