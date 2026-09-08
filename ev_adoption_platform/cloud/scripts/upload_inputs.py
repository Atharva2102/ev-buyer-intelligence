from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import boto3
import pandas as pd


INPUTS = {
    "train": "train.csv",
    "test": "test.csv",
    "original": "original_ev_adoption.csv",
    "scores": "scores.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(paths: dict[str, Path]) -> None:
    train_columns = set(pd.read_csv(paths["train"], nrows=5).columns)
    test_columns = set(pd.read_csv(paths["test"], nrows=5).columns)
    original_columns = set(pd.read_csv(paths["original"], nrows=5).columns)
    score_columns = set(pd.read_csv(paths["scores"], nrows=5).columns)

    if not {"id", "Will_Buy_EV"}.issubset(train_columns):
        raise ValueError("Training data must include id and Will_Buy_EV.")
    if "id" not in test_columns or "Will_Buy_EV" in test_columns:
        raise ValueError("Test data must include id and exclude Will_Buy_EV.")
    if not {"Buyer_ID", "Will_Buy_EV"}.issubset(original_columns):
        raise ValueError("Original data must include Buyer_ID and Will_Buy_EV.")
    if not {"id", "Will_Buy_EV"}.issubset(score_columns):
        raise ValueError("Scores must include id and Will_Buy_EV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and upload EV pipeline inputs to S3.")
    parser.add_argument("--bucket", required=True, help="CDK DataBucketName output")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--prefix", default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "train": args.train.resolve(),
        "test": args.test.resolve(),
        "original": args.original.resolve(),
        "scores": args.scores.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} input not found: {path}")

    validate_inputs(paths)
    s3 = boto3.client("s3")
    for name, filename in INPUTS.items():
        path = paths[name]
        key = f"{args.prefix.strip('/')}/{filename}"
        checksum = sha256(path)
        s3.upload_file(
            str(path),
            args.bucket,
            key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {"sha256": checksum},
            },
        )
        print(f"Uploaded {path} -> s3://{args.bucket}/{key} sha256={checksum}")


if __name__ == "__main__":
    main()
