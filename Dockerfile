# syntax=docker/dockerfile:1

# --- Frontend build ---
FROM node:22-alpine AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

# --- API runtime ---
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    RAG_BACKEND=remote

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY inference/ inference/
COPY retrieval/ retrieval/
COPY ingestion/ ingestion/
COPY evaluation/ evaluation/
COPY --from=web-build /web/dist web/dist

# Corpus artifacts are mounted at runtime (gitignored):
#   - enterprise_data/chroma_db_v3
#   - enterprise_data/bm25_index.pkl
RUN mkdir -p enterprise_data

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
