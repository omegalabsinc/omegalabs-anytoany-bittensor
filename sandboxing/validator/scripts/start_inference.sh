#!/bin/bash
set -e

# Verify running as sandbox user
if [ "$(id -u)" != "$(id -u sandbox)" ]; then
    echo "Must run as sandbox user"
    exit 1
fi

# Set resource limits
ulimit -n 1024  # File descriptors
ulimit -u 100   # Max user processes
ulimit -m 2097152  # Max memory size (2GB)

# Start inference process
exec python -m neurons.validator.server "$@"