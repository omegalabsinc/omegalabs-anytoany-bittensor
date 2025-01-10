from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class ModelInterface(ABC):
    @abstractmethod
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize with model path and device"""
        pass
        
    @abstractmethod
    def inference(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Run inference on audio input
        Returns:
            Dict with 'audio' and 'text' keys
        """
        pass