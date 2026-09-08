FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY cloud/pipeline/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY etl /app/etl
COPY cloud/pipeline/main.py /app/main.py

RUN useradd --create-home --uid 10001 pipeline && chown -R pipeline:pipeline /app
USER pipeline

ENTRYPOINT ["python", "/app/main.py"]
