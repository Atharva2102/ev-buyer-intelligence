from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol


class EventStore(Protocol):
    backend_name: str

    def put(self, event: dict[str, Any]) -> None: ...

    def recent(self, since: datetime, limit: int = 1000) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...


class SqliteEventStore:
    backend_name = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scoring_events (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    probability REAL,
                    adoption_band TEXT,
                    latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    city_type TEXT,
                    current_car_type TEXT,
                    range_anxiety_level TEXT,
                    error_message TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def put(self, event: dict[str, Any]) -> None:
        columns = [
            "event_id", "occurred_at", "source", "status", "probability",
            "adoption_band", "latency_ms", "model_version", "city_type",
            "current_car_type", "range_anxiety_level", "error_message",
        ]
        values = [event.get(column) for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO scoring_events ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def recent(self, since: datetime, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM scoring_events
                WHERE occurred_at >= ?
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                [since.isoformat(), limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ok", "backend": self.backend_name, "path": str(self.path)}


class DynamoEventStore:
    backend_name = "dynamodb"

    def __init__(self, table_name: str) -> None:
        import boto3

        self.table = boto3.resource("dynamodb").Table(table_name)

    def put(self, event: dict[str, Any]) -> None:
        occurred_at = str(event["occurred_at"])
        item = {
            "stream_key": "SCORING",
            "occurred_at_event": f"{occurred_at}#{event['event_id']}",
            "expires_at": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            **{key: value for key, value in event.items() if value is not None},
        }
        item = {
            key: Decimal(str(value)) if isinstance(value, float) else value
            for key, value in item.items()
        }
        self.table.put_item(Item=item)

    def recent(self, since: datetime, limit: int = 1000) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=(
                Key("stream_key").eq("SCORING")
                & Key("occurred_at_event").gte(since.isoformat())
            ),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [
            {
                key: float(value) if isinstance(value, Decimal) else value
                for key, value in item.items()
                if key not in {"stream_key", "occurred_at_event", "expires_at"}
            }
            for item in response.get("Items", [])
        ]

    def health(self) -> dict[str, Any]:
        response = self.table.meta.client.describe_table(TableName=self.table.name)
        return {
            "status": "ok",
            "backend": self.backend_name,
            "table": self.table.name,
            "table_status": response["Table"]["TableStatus"],
        }


def create_event_store(app_root: Path) -> EventStore:
    table_name = os.getenv("SCORING_EVENTS_TABLE", "").strip()
    if table_name:
        return DynamoEventStore(table_name)
    path = Path(os.getenv("SCORING_EVENTS_DB", app_root / "warehouse" / "scoring_events.sqlite3"))
    return SqliteEventStore(path)
