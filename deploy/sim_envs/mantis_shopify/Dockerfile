FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080 \
    SEED=42 \
    FAKE_NOW=2026-06-09T09:00:00Z \
    PYTHONUNBUFFERED=1

EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=2s --start-period=10s --retries=6 \
    CMD curl -fs http://127.0.0.1:8080/__env__/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
