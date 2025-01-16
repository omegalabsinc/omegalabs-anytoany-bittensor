#!/bin/bash

# Configure bash error handling
set -euo pipefail

# Configuration
API_HOST="localhost"
API_PORT="8000"
API_VERSION="v1"
BASE_URL="http://${API_HOST}:${API_PORT}/api/${API_VERSION}"

# Function to check if required commands are available
check_dependencies() {
    local deps=("curl" "python3" "base64")
    for cmd in "${deps[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: Required command '$cmd' not found"
            exit 1
        fi
    done
}

# Function to generate test audio data
generate_test_audio() {
    python3 - <<EOF
import numpy as np
import base64
import io

# Generate 1 second of audio at 24kHz
duration = 1.0
sample_rate = 24000
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # Hz
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Ensure proper shape [C, T] for mono audio
audio = audio.reshape(1, -1).astype(np.float32)

# Convert to base64
buffer = io.BytesIO()
np.save(buffer, audio)
print(base64.b64encode(buffer.getvalue()).decode(), end="")
EOF
}

# Function to test health endpoint
test_health() {
    echo "Testing health endpoint..."
    curl -s "${BASE_URL}/health" || {
        echo "Health check failed"
        exit 1
    }
}

# Function to test inference endpoint
test_inference() {
    echo
    echo "Testing inference endpoint..."
    local audio_data=$(generate_test_audio)
    
    curl -X POST "${BASE_URL}/inference" \
        -H "Content-Type: application/json" \
        -d "{
            \"audio_data\": \"${audio_data}\",
            \"sample_rate\": 16000
        }" || {
            echo "Inference request failed"
            exit 1
        }
}

main() {
    # check_dependencies
    # test_health
    test_inference
}

main "$@"