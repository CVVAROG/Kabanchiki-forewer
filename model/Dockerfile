FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.112" \
    "uvicorn[standard]>=0.30" \
    "onnxruntime>=1.18" \
    "numpy>=1.26" \
    "pydantic>=2.8" \
    "transformers>=4.44"

# Копируем всё содержимое директории model (модель + токенайзер + сервер)
COPY . /app/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=5 --start-period=10s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn model:app --host 0.0.0.0 --port 8000 --workers 1"]