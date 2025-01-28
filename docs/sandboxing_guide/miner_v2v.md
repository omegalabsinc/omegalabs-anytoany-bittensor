#  Bittensor Voice-to-Voice - Miner's Guide

## Overview
Evaluation system runs the models in isolated Docker containers and scores them based on their performance on voice-to-voice task.

## How to Submit Your Model

### 1. Required Repository Structure
Your HuggingFace repository must contain:
```
your_repo/
├── Dockerfile            # Docker configuration for your model
├── server.py            # FastAPI server implementation
├── inference.py         # Core inference logic
├── requirements.txt     # Python dependencies
├── hotkey.txt          # Your hotkey 
└── your_model_files/   # Your model files
```

**Example repository:** [tezuesh/moshi_general](https://huggingface.co/tezuesh/moshi_general/)

### 2. Required Endpoints

#### Health Check Endpoint
```python
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy" if model_initialized else "initializing",
        "initialization_status": {
            "model_loaded": bool,
            "error": str or None
        }
    }
```

#### Inference Endpoint
```python
@app.post("/api/v1/inference")
async def inference(request: AudioRequest) -> AudioResponse:
    class AudioRequest(BaseModel):
        audio_data: str  # Base64 encoded audio array
        sample_rate: int

    class AudioResponse(BaseModel):
        audio_data: str  # Base64 encoded output audio
        text: str = ""   # Optional generated text
```

## How The Evaluation Works

### 1. Dataset Loading
```python
# System pulls latest voice samples from dataset
dataset = pull_latest_diarization_dataset()
# Dataset contains pairs of audio segments from different speakers
```

### 2. Container Management
```python
# For each model evaluation:
docker_manager = DockerManager(base_cache_dir=local_dir)

# Start container with GPU support
container_url = docker_manager.start_container(
    uid=f"{model_id}_{timestamp}", 
    repo_id=hf_repo_id,
    gpu_id=0  # If GPU available
)
```

### 3. Evaluation Process
```python
# For each audio sample:
1. Extract audio segments
2. Send to model container:
   result = docker_manager.inference_v2v(
       url=container_url,
       audio_array=input_audio,
       sample_rate=sample_rate
   )

3. Compute metrics:
   - MIMI score
   - WER score
   - Length penalty
   - PESQ score
   - Anti-spoofing score
   - Combined score
```

### 4. Security Features
- Isolated Docker network
- Read-only model volumes
- Resource limits
- GPU isolation
- Health monitoring

## Key Features of the Evaluation System

1. **Container Isolation**
```python
'networks': {
    'isolated_network': {
        'driver': 'bridge',
        'internal': True  # Prevents internet access
    }
}
```

2. **Resource Management**
```python
'deploy': {
    'resources': {
        'reservations': {
            'devices': [{
                'driver': 'nvidia',
                'count': 'all',
                'capabilities': ['gpu', 'utility', 'compute']
            }]
        }
    }
}
```

3. **Health Monitoring**
```python
'healthcheck': {
    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/api/v1/health'],
    'interval': '30s',
    'timeout': '10s',
    'retries': 3
}
```

## Best Practices for Miners

1. **Efficient Resource Usage**
```python
# Implement proper cleanup
@app.on_event("shutdown")
async def shutdown_event():
    cleanup_resources()
```

2. **Error Handling**
```python
@app.post("/api/v1/inference")
async def inference(request: AudioRequest):
    try:
        # Process audio
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Audio Processing**
```python
def _load_audio(self, audio_array: np.ndarray, sample_rate: int):
    # Implement proper audio preprocessing
    # Handle resampling if needed
    # Handle normalization
    pass
```

## Scoring Metrics

Your model will be evaluated on:
1. Voice conversion quality (MIMI score)
2. Speech recognition accuracy (WER score)
3. Output length appropriateness (Length penalty)
4. Audio quality (PESQ score)
5. Anti-spoofing robustness
6. Overall combined score

The system calculates a final score:
```python
mean_score = np.mean(metrics['combined_score'])
```

The key differences from previous approaches are:
1. Full model flexibility
2. Secure container-based evaluation
3. Automated resource management
4. Standard API interface regardless of model architecture

This system allows miners to innovate on model architecture while maintaining a standardized evaluation process through Docker containers and well-defined APIs.