#!/bin/bash
set -e  # Exit on error

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

# Create sandbox user if doesn't exist
if ! id "sandbox" &>/dev/null; then
    useradd --system \
            --shell=/bin/false \
            --create-home \
            --home-dir /home/sandbox sandbox
    echo "Created sandbox user"
else
    echo "Sandbox user already exists"
fi

# Setup sandbox directories
mkdir -p /sandbox/data
chown -R sandbox:sandbox /sandbox
chmod -R 755 /sandbox

# Set resource limits for sandbox user
cat > /etc/security/limits.d/sandbox.conf << EOF
sandbox soft nproc 100
sandbox hard nproc 200
sandbox soft nofile 1024
sandbox hard nofile 4096
sandbox soft as 2097152
sandbox hard as 2097152
EOF

echo "Sandbox environment setup complete"

# Build and start the container
docker compose -f docker/min_compute.yml up --build -d

echo "Validator container started"