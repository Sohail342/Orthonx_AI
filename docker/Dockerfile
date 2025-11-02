# Stage 1: Builder
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install uv

# Copy dependency definitions
COPY pyproject.toml ./
# COPY uv.lock ./   # Optional if you generate one

# Allow specifying which extras to install
ARG INSTALL_EXTRAS=prod

# Install dependencies directly into the system (for runtime stage)
RUN uv pip install --system --no-cache ".[${INSTALL_EXTRAS}]"


# Stage 2: Runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment variables for cleaner Python logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application source code
COPY ./app ./app
COPY ./alembic.ini ./
COPY ./alembic ./alembic

# Copy and set permissions for entrypoint script
COPY docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh


RUN useradd -m appuser


RUN mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (production)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
