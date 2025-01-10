from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
import torch
from typing import Dict, Any, Optional
import time
import numpy as np
import json
from neurons.validator.server import InferenceServer

logger = logging.getLogger(__name__)

def create_router(inference_server_dict: Dict[str, Any]) -> APIRouter:
    """Create and configure API router with access to inference server"""
    router = APIRouter()

    logger.info(f"inside endpoints.py, Inference server state: {inference_server_dict}")
    inference_server = inference_server_dict["inference_server"]
    
    @router.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "Validator API",
            "version": "1.0.0",
            "status": "running" if inference_server and inference_server._is_running else "stopped",
            "endpoints": {
                "GET /": "API documentation",
                "GET /health": "Health check",
                "GET /score": "Get model scores",
                "GET /metrics": "Get detailed metrics",
                "GET /debug": "Debug information"
            }
        }

    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            return {
                "status": "healthy" if inference_server and inference_server._is_running else "initializing",
                "server_running": inference_server._is_running if inference_server else False,
                "gpu_available": torch.cuda.is_available(),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    @router.get("/score")
    async def get_scores():
        """Get model scores"""
        
        try:
            with open("/sandbox/results.json", "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to get scores: {e}")
            return {"status": "error", "error": str(e)}

    @router.get("/metrics")
    async def get_metrics():
        """Get model metrics"""
        if not inference_server or not inference_server._is_running:
            return {
                "status": "initializing",
                "message": "Inference server is starting up"
            }
            
        try:
            metrics = inference_server.get_latest_scores()
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"status": "error", "error": str(e)}

    @router.get("/debug")
    async def get_debug():
        """Get debug information"""
        if not inference_server:
            return {"status": "Server not initialized"}
            
        try:
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "memory_cached": torch.cuda.memory_reserved() / (1024**3)
                }
            
            return {
                "server_running": inference_server._is_running,
                "has_model": hasattr(inference_server.sandbox, '_model'),
                "has_data": inference_server.data_processor.dataset is not None,
                "last_scores": inference_server.get_latest_scores(),
                "gpu_info": gpu_info,
                "sandbox_status": inference_server.sandbox.get_status() if inference_server._is_running else None
            }
            
        except Exception as e:
            logger.error(f"Error getting debug info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router