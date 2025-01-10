FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-dev \
    git \
    git-lfs \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Create sandbox user with home directory
RUN useradd -m -s /bin/bash sandbox

# Create all required subdirectories with proper permissions
RUN mkdir -p /sandbox/models \
            /sandbox/models/cache/huggingface \
            /sandbox/data \
            /sandbox/.cache/librosa \
            /tmp/numba_cache \
            /tmp/.config/matplotlib && \
    chown -R sandbox:sandbox /sandbox /tmp/numba_cache /tmp/.config && \
    chmod -R 777 /sandbox /tmp/numba_cache /tmp/.config

# Create directories with proper permissions
RUN mkdir -p /sandbox/.cache/huggingface && \
    chown -R sandbox:sandbox /sandbox/.cache && \
    chmod -R 755 /sandbox/.cache

# Setup application
WORKDIR /app

# Install dependencies
COPY --chown=sandbox:sandbox neurons /app/neurons/
COPY --chown=sandbox:sandbox evaluation /app/evaluation/
# COPY --chown=sandbox:sandbox tests /app/tests/
COPY --chown=sandbox:sandbox setup.py pyproject.toml /app/
RUN pip install -e .
RUN pip install "fastapi>=0.68.0" "uvicorn>=0.15.0"

# Copy startup script
COPY --chown=sandbox:sandbox docker/startup.sh /startup.sh
RUN chmod +x /startup.sh

# Create model directories
RUN mkdir -p /sandbox/model && \
    chown sandbox:sandbox /sandbox/model && \
    chmod 755 /sandbox/model

# Add healthcheck with more verbose output
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -v http://localhost:8000/api/v1/health > /proc/1/fd/1 2>/proc/1/fd/2 || exit 1

# Set the startup script as entrypoint
ENTRYPOINT ["/startup.sh"]