# Stage 1: Builder
FROM python:3.12-slim AS builder

ENV HF_HOME=/tmp/hfcache
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Install system dependencies needed for building/installing
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

# Install Torch and Torchvision together from the CPU index
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Install remaining project dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-cache ".[prod]"

# Stage 2: Runtime
FROM python:3.12-slim

ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

ENV HF_HOME=/app/hfcache
ENV PYTHONUNBUFFERED=1

# Install RUNTIME dependencies (libgomp1 is vital for Torch CPU)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

COPY ./app ./app
COPY ./alembic.ini ./
COPY ./alembic ./alembic
COPY docker/docker-entrypoint.sh /usr/local/bin/

ARG HF_API_KEY
ENV HF_API_KEY=${HF_API_KEY}

#  Download models from huggingface
RUN mkdir -p /app/ml_models/bone_fracture_detection \
    && mkdir -p /app/ml_models/bone_fracture_model_onnx \
    && python -c "import app.utils.load_models"

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
