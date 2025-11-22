# Stage 1: Builder
FROM python:3.12-slim AS builder

# Set timezone to Pakistan (PKT)
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Install build + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast installer)
RUN pip install --no-cache-dir uv

# Copy pyproject.toml
COPY pyproject.toml ./

# Install PyTorch CPU + extras (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-cache \
    "torch==2.3.0+cpu" "torchvision==0.18.0+cpu" \
    -f https://download.pytorch.org/whl/torch_stable.html \
    ".[prod]"

# Stage 2: Runtime
FROM python:3.12-slim

# Pakistan Timezone
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser

# Copy installed packages
COPY --from=builder /usr/local /usr/local

# Copy app
COPY ./app ./app
COPY ./alembic.ini ./
COPY ./alembic ./alembic
COPY docker/docker-entrypoint.sh /usr/local/bin/

# Models (pre-downloaded)
COPY ./app/ml_models ./ml_models
COPY ./app/pytorch_models ./pytorch_models

# Setup
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
