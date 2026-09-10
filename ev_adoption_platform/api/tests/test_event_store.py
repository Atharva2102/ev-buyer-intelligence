from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from api.event_store import SqliteEventStore


class SqliteEventStoreTests(unittest.TestCase):
    def test_recent_filters_and_orders_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteEventStore(Path(directory) / "events.sqlite3")
            now = datetime.now(timezone.utc)
            for index, minutes_ago in enumerate([90, 5, 1]):
                store.put(
                    {
                        "event_id": f"event-{index}",
                        "occurred_at": (now - timedelta(minutes=minutes_ago)).isoformat(),
                        "source": "simulator",
                        "status": "succeeded",
                        "probability": 0.2 + index / 10,
                        "adoption_band": "Watch",
                        "latency_ms": 12.5,
                        "model_version": "test-model",
                        "city_type": "Urban",
                        "current_car_type": "Sedan",
                        "range_anxiety_level": "Low",
                        "error_message": None,
                    }
                )

            events = store.recent(now - timedelta(minutes=60))

            self.assertEqual([event["event_id"] for event in events], ["event-2", "event-1"])
            self.assertEqual(store.health()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
