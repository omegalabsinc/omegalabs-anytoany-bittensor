# Bittensor Video-to-Text - Miner's Guide

## Overview
The evaluation system runs models in isolated Docker containers and scores them based on their performance on video-to-text captioning tasks. Models receive video embeddings as input and must generate appropriate textual captions.

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

**Example repository:** [tezuesh/IBLlama_v1](https://huggingface.co/tezuesh/IBLlama_v1)

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
async def inference(request: EmbeddingRequest) -> TextResponse:
    class EmbeddingRequest(BaseModel):
        embedding: List[float]  # [batch_size * 1024] dimensional vector

    class TextResponse(BaseModel):
        texts: List[str]  # List of generated captions
```

## How The Evaluation Works

### 1. Dataset Loading
```python
# System pulls latest video samples and embeddings from dataset
dataset = pull_latest_dataset()
# Dataset contains video embeddings and ground truth captions
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
# For each batch:
1. Process embeddings
2. Send to model container:
   result = docker_manager.inference_ibllama(
       url=container_url,
       video_embed=video_embed_list  # [batch_size * 1024] dimensional
   )

3. Compute metrics:
   - Text embedding similarity
   - Caption quality metrics
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
async def inference(request: EmbeddingRequest):
    try:
        # Process embeddings
        return generate_captions(request.embedding)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Embedding Processing**
```python
def process_embeddings(self, embedding_list: List[float]) -> torch.Tensor:
    """Process raw embedding list into proper tensor format"""
    embedding = torch.tensor(embedding_list)
    # Reshape to [batch_size, 1024]
    embedding = embedding.reshape(-1, 1024)
    return embedding
```

## Scoring Metrics

Your model will be evaluated on:
1. Caption quality (using embedding similarity)
2. Response time and efficiency
3. Resource utilization
4. System reliability

The system calculates a final score:
```python
# Compute similarity between generated and actual captions
text_similarity = torch.nn.functional.cosine_similarity(
    generated_embeddings,
    actual_embeddings,
    dim=-1
)

mean_similarity = torch.tensor(similarities).mean().item()
```

The key differences from previous approaches are:
1. Full model flexibility
2. Secure container-based evaluation
3. Automated resource management
4. Standardized API interface

## Implementation Requirements

### Input Format
- Embeddings come as flat lists of length batch_size * 1024
- Must reshape to [batch_size, 1024] before processing
- Batch size is variable and must be inferred from input

### Output Format
- Must return list of captions as strings
- Length of output list must match batch size
- Each caption should be coherent and descriptive

### Performance Considerations
- Implement efficient batch processing
- Optimize memory usage
- Proper GPU resource management
- Clean resource handling

This system allows miners to use any model architecture while maintaining standardized interfaces through Docker containers and well-defined APIs.