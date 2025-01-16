from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from pydantic import BaseModel
import os
import logging
import base64
import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import your evaluation metrics
from Evaluation.S2S.distance import S2SMetrics
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    sample_rate: int

class AudioResponse(BaseModel):
    audio_data: str  # base64 encoded audio
    text: str = ""

class InferenceModel:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get environment variables
        self.repo_id = os.getenv("REPO_ID")
        self.model_id = os.getenv("MODEL_ID")
        if not self.repo_id:
            raise ValueError("REPO_ID environment variable not set")
        self.load_model()
        
        # Initialize metrics for anti-spoofing
        self.metrics = S2SMetrics(cache_dir="/model_cache")

    def load_model(self):
        """Load model from HuggingFace"""
        try:
            # Download model files
            model_path = huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                filename="model.pt",
                cache_dir="/model_cache"
            )
            
            # Load your model implementation here
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {self.repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_audio(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Run inference on audio input"""
        try:
            with torch.no_grad():
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_array).float().to(self.device)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Process audio through model
                output = self.model(audio_tensor)
                
                # Convert output to numpy array
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                
                return {
                    "audio": output,
                    "text": ""
                }
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

# Global model instance
model = InferenceModel()

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/api/v1/inference")
async def inference(request: AudioRequest) -> AudioResponse:
    """Run inference on audio input"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.load(io.BytesIO(audio_bytes))
        
        # Run inference
        result = model.process_audio(audio_array, request.sample_rate)
        
        # Encode output audio
        buffer = io.BytesIO()
        np.save(buffer, result["audio"])
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return AudioResponse(
            audio_data=audio_b64,
            text=result.get("text", "")
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)