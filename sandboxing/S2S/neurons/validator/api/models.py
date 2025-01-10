from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class AudioValidateRequest(BaseModel):
    """Request model for audio validation endpoint"""
    audio: bytes = Field(..., description="Raw audio data bytes")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    metadata: Optional[Dict] = Field(default=None, description="Optional metadata")

class ValidationResponse(BaseModel):
    """Response model for audio validation results"""
    audio_output: bytes = Field(..., description="Generated audio output")
    metrics: Dict[str, float] = Field(..., description="Computed metric scores")
    compute_score: float = Field(..., description="Compute resource efficiency score")
    duration_ms: int = Field(..., description="Processing duration in milliseconds")

class MetricResponse(BaseModel):
    """Response model for metrics endpoint"""
    wer_score: float = Field(..., description="Word Error Rate score")
    pesq_score: float = Field(..., description="PESQ score")  
    length_penalty: float = Field(..., description="Length difference penalty")
    anti_spoofing_score: float = Field(..., description="Anti-spoofing detection score")
    mimi_score: float = Field(..., description="MIMI quantization score")
    combined_score: float = Field(..., description="Overall combined score")

class ResourceUsage(BaseModel):
    """System resource utilization metrics"""
    gpu_memory_gb: float = Field(..., description="GPU memory used in GB")
    ram_percent: float = Field(..., description="RAM usage percentage") 
    cpu_percent: float = Field(..., description="CPU usage percentage")

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service health status")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    resource_usage: ResourceUsage = Field(..., description="Current resource usage")
    version: str = Field(..., description="Validator version")

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(default=None, description="Additional error details")