#  Bittensor Voice-to-Voice & Voice-to-Text - Miner's Guide for new competition-id = v2

## Overview
Evaluation system runs the models in isolated Docker containers and scores them based on their performance on voice-to-voice (V2V) or voice-to-text (V2T) tasks. The primary evaluation uses VoiceBench for comprehensive assessment across 11 datasets with LLM-based quality scoring.

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

**Example repository:** [siddhantoon/miner_v2t](https://huggingface.co/siddhantoon/miner_v2t/)

### 2. Required Endpoints in server.py

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

#### Voice-to-Voice Inference Endpoint
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

#### Voice-to-Text Inference Endpoint
```python
@app.post("/api/v1/v2t")
async def voice_to_text(request: AudioRequest) -> TextResponse:
    class AudioRequest(BaseModel):
        audio_data: str  # Base64 encoded audio array
        sample_rate: int

    class TextResponse(BaseModel):
        text: str  # Generated text response
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

#### VoiceBench Evaluation (Primary)
```python
# For voice-to-text evaluation using VoiceBench:
1. Load VoiceBench datasets (11 comprehensive benchmarks)
2. Send audio samples to model:
   result = docker_manager.inference_v2t(
       url=container_url,
       audio_array=input_audio,
       sample_rate=sample_rate
   )
3. Evaluate with LLM judge for quality scoring
```

#### Legacy V2V Evaluation
```python
# For voice-to-voice models:
1. Extract audio segments
2. Send to model container:
   result = docker_manager.inference_v2v(
       url=container_url,
       audio_array=input_audio,
       sample_rate=sample_rate
   )
3. Compute traditional metrics
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

2. **Complete V2T Endpoint Example**
```python
@app.post("/api/v1/v2t")
async def voice_to_text(request: AudioRequest):
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.load(io.BytesIO(audio_bytes))
        
        # Process with your model
        # 1. Speech recognition (e.g., Whisper)
        text = whisper_model.transcribe(audio_array, sample_rate=request.sample_rate)
        
        # 2. Generate response (e.g., LLM)
        response = llm_model.generate(text)
        
        return TextResponse(text=response)
    except Exception as e:
        logger.error(f"V2T inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Error Handling**
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

4. **Audio Processing**
```python
def _load_audio(self, audio_array: np.ndarray, sample_rate: int):
    # Implement proper audio preprocessing
    # Handle resampling if needed
    # Handle normalization
    pass
```

## Scoring Metrics

### VoiceBench Evaluation (Recommended for V2T)
Your model will be evaluated across 11 comprehensive datasets:

**Instruction Following & General Knowledge:**
1. **CommonEval** - Common knowledge questions
2. **WildVoice** - Conversational scenarios
4. **AdvBench** - Multi-turn conversations
5. **IFEval** - Instruction following accuracy

**Scoring with LLM Judge:**
- Each response is evaluated by an LLM judge (Qwen/GPT-4)
- Quality-based scoring (1-5 scale for open questions, Yes/No for QA)
- Weighted average across all datasets
- Real accuracy assessment, not just API success rate



The system calculates a final score:
```python
# VoiceBench scoring (primary):
voicebench_score = weighted_average(llm_scores_per_dataset)

```

The key differences from previous approaches are:
1. Full model flexibility
2. Secure container-based evaluation
3. Automated resource management
4. Standard API interface regardless of model architecture

This system allows miners to innovate on model architecture while maintaining a standardized evaluation process through Docker containers and well-defined APIs.