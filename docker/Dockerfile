# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency definitions
COPY pyproject.toml ./
# COPY uv.lock ./   # optional if you generate one

ARG INSTALL_EXTRAS=prod

# Install dependencies directly into system
RUN uv pip install --system --no-cache ".[${INSTALL_EXTRAS}]"

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy source code
COPY ./app ./app
COPY ./alembic.ini ./
COPY ./alembic ./alembic

# Default command (prod)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
