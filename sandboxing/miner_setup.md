# Miner API Specification

## Overview
This document outlines the required API endpoints and data formats that all miners must implement in their Docker containers.

## Required Endpoints

### Health Check
```
GET /api/v1/health
```
Returns health status of the model server.

**Response**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "gpu_available": true
}
```

### Inference
```
POST /api/v1/inference
```
Processes audio and returns transformed audio.

**Request Body**
```json
{
    "audio_data": "base64_encoded_audio_array",
    "sample_rate": 16000
}
```

**Response**
```json
{
    "audio_data": "base64_encoded_audio_array", 
    "text": "optional_transcription"
}
```

## HuggingFace Repository Structure
```
your-hf-repo/
├── model.pt             # Your model weights (any format)
├── requirements.txt     # Python dependencies if using Python
├── setup.sh            # Optional setup script
├── config.yaml         # Model configuration
└── README.md           # Documentation
```

## Docker Container Requirements

1. Must expose port 8000 for API access
2. Must implement all required endpoints
3. Must handle base64 encoded numpy arrays for audio data
4. Must include appropriate error handling
5. Must initialize within 60 seconds
6. Must process inference requests within reasonable timeout

## Resource Limits
- Memory: 16GB
- CPU: 4 cores 
- GPU: 1 device (if required)
- Network: Limited to validator communication only

## Security Requirements
1. No privileged access
2. No host filesystem access outside mounted volumes
3. No network access except to validator
4. Must run as non-root user

## Error Handling
All endpoints must return appropriate HTTP status codes:
- 200: Success
- 400: Invalid input
- 500: Server error
- 503: Model not ready

## Testing
Miners should test their container with the following:
1. Basic health check functionality
2. Audio processing with various lengths
3. Error handling for invalid inputs
4. Resource usage within limits
5. Proper cleanup on shutdown