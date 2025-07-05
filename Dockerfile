# syntax=docker/dockerfile:1.6
# -------------------------------------------------------------
# Multi‑stage Dockerfile for AI‑memory‑ (image < 150 MB)
# 1. Build stage   – python:3.11‑bookworm, compile wheels & install deps
# 2. Runtime stage – python:3.11‑slim‑bookworm, copy wheels only
# -------------------------------------------------------------

############################ 1️⃣ build‑stage ############################
FROM python:3.11-bookworm AS build

# Install system build deps only in builder image
RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc \
    libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first (leverages Docker layer‑cache)
COPY requirements*.txt ./
# Enable BuildKit pip cache layer (~ /root/.cache/pip) to speed up rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels -r requirements.txt

# Copy project source after deps (changes here bust cache only when src changes)
COPY . /build/src

# Install project into a temp location
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install /build/src

############################ 2️⃣ runtime‑stage ###########################
FROM python:3.11-slim-bookworm AS runtime
LABEL maintainer="Ingver1 <github.com/Ingver1>" \
      org.opencontainers.image.source="https://github.com/Ingver1/AI-memory-" \
      org.opencontainers.image.description="Self‑hosted AI memory service (FastAPI + FAISS)"

# Workdir inside container
WORKDIR /app

# Copy Python env from build stage (only site‑packages & entrypoints)
COPY --from=build /install /usr/local

# Copy runtime assets (static files, logging config, etc.)
COPY --chown=root:root logging.yaml ./
COPY --chown=root:root memory_system ./memory_system

# Non‑root user for security (optional)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    AI_MEMORY_SETTINGS=/app/settings.toml

EXPOSE $PORT

# Default command (can be overridden by docker‑compose / k8s)
ENTRYPOINT ["uvicorn", "memory_system.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
