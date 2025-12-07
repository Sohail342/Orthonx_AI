# Stage 1: Builder
FROM python:3.12-slim AS builder

# Set environment variables for Hugging Face cache
ENV HF_HOME=/tmp/hfcache

# Set timezone to Pakistan (PKT)
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Install build + runtime deps
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    # Clean up apt caches to save space in this layer
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast installer)
RUN pip install --no-cache-dir uv

# Copy pyproject.toml
COPY pyproject.toml ./

# Install CPU-only PyTorch and other requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    # Clean up pip cache which might be left over in the temp dir
    && rm -rf /root/.cache/pip

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-cache \
    ".[prod]"

# Stage 2: Runtime
FROM python:3.12-slim

# Pakistan Timezone
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Set environment variables
ENV HF_HOME=/app/hfcache
ENV PYTHONUNBUFFERED=1

# Copy the minimum necessary system dependencies from the builder stage
COPY --from=builder /usr/lib/x86_64-linux-gnu/libglib* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgthread* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgobject* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmount* /usr/lib/x86_64-linux-gnu/


# Runtime deps only
RUN useradd -m appuser

# Copy installed packages
COPY --from=builder /usr/local /usr/local

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy app
COPY ./app ./app
COPY ./alembic.ini ./
COPY ./alembic ./alembic
COPY docker/docker-entrypoint.sh /usr/local/bin/

ARG HF_API_KEY
ENV HF_API_KEY=${HF_API_KEY}

# Download ML models from Hugging Face account
RUN mkdir -p /app/ml_models/bone_fracture_detection \
    && mkdir -p /app/ml_models/bone_fracture_model_onnx \
    && python -c "import app.utils.load_models"

# Setup
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
