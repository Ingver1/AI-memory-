# Unified Memory System v0.8-alpha
# Multi-stage build for optimized production image

# ────────────────────────────────────────────────────────────────────────
# Build stage
# ────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

# Build arguments
ARG PIP_EXTRA_INDEX_URL=""
ARG BUILD_ENV="production"

# Metadata
LABEL org.opencontainers.image.title="Unified Memory System"
LABEL org.opencontainers.image.version="0.8-alpha"
LABEL org.opencontainers.image.description="Long-term memory backend with FastAPI, FAISS and Prometheus metrics"
LABEL org.opencontainers.image.source="https://github.com/your-org/unified-memory-system"
LABEL org.opencontainers.image.documentation="https://unified-memory-system.readthedocs.io/"

# Install system dependencies for building
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libopenblas-dev \
        libgomp1 \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set up virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt requirements_dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --extra-index-url "$PIP_EXTRA_INDEX_URL" -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# ────────────────────────────────────────────────────────────────────────
# Runtime stage
# ────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim as runtime

# Runtime dependencies only
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Create non-root user
RUN groupadd -r ums && useradd -r -g ums -d /app -s /bin/bash ums

# Create app directory
WORKDIR /app

# Copy application files
COPY --from=builder --chown=ums:ums /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R ums:ums /app

# Switch to non-root user
USER ums

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "memory_system.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]

# ────────────────────────────────────────────────────────────────────────
# Development stage
# ────────────────────────────────────────────────────────────────────────
FROM runtime as development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements_dev.txt

# Install development tools
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        htop \
        tree && \
    rm -rf /var/lib/apt/lists/*

# Switch back to ums user
USER ums

# Development command with auto-reload
CMD ["uvicorn", "memory_system.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--factory"]

# ────────────────────────────────────────────────────────────────────────
# Testing stage
# ────────────────────────────────────────────────────────────────────────
FROM development as testing

# Switch to root for test setup
USER root

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov httpx

# Set up test environment
ENV ENVIRONMENT=testing
ENV UMS_LOG_LEVEL=DEBUG

# Switch back to ums user
USER ums

# Test command
CMD ["pytest", "-v", "--cov=memory_system", "--cov-report=term-missing"]
