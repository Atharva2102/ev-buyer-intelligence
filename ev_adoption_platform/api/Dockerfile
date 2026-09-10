FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    WAREHOUSE_PATH=/app/warehouse/ev_adoption.duckdb \
    MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY api /app/api
COPY etl /app/etl
COPY models /app/models

RUN useradd --create-home --uid 10001 api \
    && mkdir -p /app/warehouse \
    && chown -R api:api /app
USER api

EXPOSE 8000

CMD ["sh", "-c", "python -m api.bootstrap && exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
