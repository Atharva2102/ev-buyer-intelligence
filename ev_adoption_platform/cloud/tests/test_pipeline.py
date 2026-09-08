from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from cloud.pipeline.main import build_tables, object_key, write_artifacts
from etl.build_warehouse import FEATURE_COLUMNS


class PipelineTests(unittest.TestCase):
    def test_object_key_normalizes_slashes(self) -> None:
        self.assertEqual(object_key("/curated/latest/", "table.parquet"), "curated/latest/table.parquet")

    def test_build_tables_validates_and_builds_all_marts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                [31, 90000, 20, 1, 4, 5, 8, "Female", "Urban", "Sedan", "Yes", "Yes", "Low"],
                [52, 55000, 48, 2, 1, 2, 4, "Male", "Rural", "Truck", "No", "No", "High"],
                [42, 76000, 30, 1, 3, 4, 7, "Female", "Suburban", "SUV", "Yes", "No", "Medium"],
                [28, 68000, 15, 0, 5, 6, 9, "Male", "Urban", "Hatchback", "Yes", "Yes", "Low"],
            ]
            train = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
            train.insert(0, "id", [1, 2, 3, 4])
            train["Will_Buy_EV"] = ["Yes", "No", "No", "Yes"]

            test = pd.DataFrame(rows[:2], columns=FEATURE_COLUMNS)
            test.insert(0, "id", [10, 11])

            original = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
            original.insert(0, "Buyer_ID", [101, 102, 103, 104])
            original["Will_Buy_EV"] = ["Yes", "No", "No", "Yes"]
            scores = pd.DataFrame({"id": [10, 11], "Will_Buy_EV": [0.72, 0.18]})

            paths = {}
            for name, frame in {
                "train": train,
                "test": test,
                "original": original,
                "scores": scores,
            }.items():
                path = root / f"{name}.csv"
                frame.to_csv(path, index=False)
                paths[name] = path

            tables = build_tables(paths, "scores.csv")

            self.assertEqual(len(tables), 8)
            self.assertEqual(len(tables["scored_current_customers"]), 2)
            self.assertEqual(
                set(tables["scored_current_customers"]["adoption_band"]),
                {"Priority", "Watch"},
            )
            self.assertEqual(len(tables["mart_overview_metrics"]), 4)

            artifacts = write_artifacts(tables, root / "artifacts")
            self.assertEqual(len(artifacts), 9)
            self.assertTrue(artifacts["warehouse"].is_file())
            self.assertTrue(artifacts["mart_segment_metrics"].is_file())
            with duckdb.connect(str(artifacts["warehouse"]), read_only=True) as connection:
                scored_count = connection.execute(
                    "SELECT COUNT(*) FROM scored_current_customers"
                ).fetchone()[0]
            self.assertEqual(scored_count, 2)


if __name__ == "__main__":
    unittest.main()
