#!/bin/bash
set -e

# Configure environment
export PYTHONPATH="/app:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Setup directories with proper permissions
mkdir -p /sandbox/data /tmp/numba_cache /sandbox/.cache/huggingface /sandbox/models
chown -R sandbox:sandbox /sandbox /tmp/numba_cache
chmod -R 755 /sandbox /tmp/numba_cache

# Start API server with proper host binding
echo "Starting validator API server..."
exec python -m neurons.validator.api.server --host 0.0.0.0 --port 8000