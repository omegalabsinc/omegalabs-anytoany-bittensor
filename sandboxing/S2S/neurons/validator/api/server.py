import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import argparse
from pathlib import Path
import torch
import time

from neurons.validator.config import ValidatorConfig, ModelConfig, MetricsConfig, SandboxConfig, MetricParameters
from neurons.validator.server import InferenceServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global server instance
inference_server = None
inference_server_dict = {"inference_server": inference_server}

def setup_inference_server():
    """Initialize inference server with configuration"""
    global inference_server
    
    config = ValidatorConfig(
        model=ModelConfig(
            model_id="tezuesh/moshi_general",
            revision="main",
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        metrics=MetricsConfig(
            metrics_cache_dir="/sandbox/.cache/huggingface",
            metrics={
                "mimi_score": MetricParameters(enabled=True, weight=0.3),
                "pesq_score": MetricParameters(enabled=True, weight=0.3),
                "anti_spoofing_score": MetricParameters(enabled=True, weight=0.2),
                "length_penalty": MetricParameters(enabled=True, weight=0.2)
            }
        ),
        sandbox=SandboxConfig(
            sandbox_user="sandbox",
            sandbox_dir=Path("/sandbox"),
            socket_path=Path("/sandbox/inference.sock"),
            max_memory_mb=2048,
            max_file_size_mb=1000,
            socket_timeout=30
        )
    )
    
    inference_server = InferenceServer(config)
    inference_server_dict["inference_server"] = inference_server
    inference_server.start()
    
    # Run initial scoring
    try:
        scores = inference_server.score_model(num_samples=16)
        pth = Path("/sandbox/results.json")
        inference_server.save_results(scores, pth)
        
        logger.info(f"Initial model scores: {scores}")
    except Exception as e:
        logger.error(f"Initial scoring failed: {e}")

def create_app():
    """Create FastAPI application"""
    app = FastAPI(
        title="Validator API",
        description="API for model validation",
        version="1.0.0"
    )
    
    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        try:
            logger.info("Initializing inference server...")
            setup_inference_server()
            logger.info("Inference server initialized successfully inside startup_event")
            logger.info(f"inside startup_event log 1, Inference server state: {inference_server}")
        except Exception as e:
            logger.error(f"Failed to initialize inference server: {str(e)}", exc_info=True)
            raise
    
    @app.get("/")
    async def root():
        """Root endpoint for basic connectivity check"""
        return {
            "status": "online",
            "version": "1.0.0"
        }
    
    # Create and mount router
    from .endpoints import create_router
    logger.info(f"inside create_app, Inference server state: {inference_server}")
    router = create_router(inference_server_dict)
    app.include_router(router, prefix="/api/v1")
    
    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Create and configure uvicorn server
    config = uvicorn.Config(
        "neurons.validator.api.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
        workers=1,
        loop="asyncio",
        timeout_keep_alive=1800,  # 30 minutes
        access_log=True
    )
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()