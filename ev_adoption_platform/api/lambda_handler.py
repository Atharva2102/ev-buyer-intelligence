from __future__ import annotations

from mangum import Mangum

from api.bootstrap import sync_warehouse


sync_warehouse()

from api.main import app  # noqa: E402


handler = Mangum(app, lifespan="off")
