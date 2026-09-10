from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


def sync_warehouse() -> None:
    bucket = os.getenv("DATA_BUCKET", "").strip()
    if not bucket:
        print("DATA_BUCKET is not set; using the local warehouse artifact.")
        return

    import boto3

    key = os.getenv("WAREHOUSE_S3_KEY", "curated/latest/ev_adoption.duckdb").strip()
    target = Path(os.getenv("WAREHOUSE_PATH", APP_ROOT / "warehouse" / "ev_adoption.duckdb"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.download")

    client = boto3.client("s3")
    metadata = client.head_object(Bucket=bucket, Key=key)
    client.download_file(bucket, key, str(temporary))
    temporary.replace(target)

    source = {
        "bucket": bucket,
        "key": key,
        "etag": str(metadata.get("ETag", "")).strip('"'),
        "size_bytes": int(metadata.get("ContentLength", target.stat().st_size)),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    target.with_suffix(f"{target.suffix}.source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    print(f"Downloaded s3://{bucket}/{key} to {target}.")


if __name__ == "__main__":
    sync_warehouse()
