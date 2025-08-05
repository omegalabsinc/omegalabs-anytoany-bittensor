
"""
Miner Model Assistant for VoiceBench Integration

This module provides a VoiceAssistant implementation that interfaces directly
with miner model APIs instead of using Docker containers.
"""

import json
import io
import base64
import requests
import numpy as np
import bittensor as bt
from typing import Dict, Any


class VoiceAssistant:
    """Base class for voice assistants (matching VoiceBench interface)"""
    
    def generate_audio(self, audio, max_new_tokens=2048):
        """Generate response from audio input"""
        raise NotImplementedError
    
    def generate_text(self, text):
        """Generate response from text input"""
        raise NotImplementedError


class MinerModelAssistant(VoiceAssistant):
    """
    Voice assistant that interfaces with miner model APIs.
    
    This class adapts miner models to work with VoiceBench evaluation
    by calling their HTTP APIs directly.
    """
    
    def __init__(self, api_url="http://localhost:8010/api/v1/v2t", timeout=600):
        """
        Initialize the miner model assistant.
        
        Args:
            api_url: URL of the miner model API endpoint
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        bt.logging.info(f"Initialized MinerModelAssistant with API: {api_url}")
    
    def generate_audio(self, audio, max_new_tokens=2048):
        """
        Generate text response from audio input.
        
        Args:
            audio: dict with 'array' (numpy array) and 'sampling_rate' keys
            max_new_tokens: Maximum tokens to generate (unused but kept for compatibility)
            
        Returns:
            str: Text response from the model
        """
        try:
            # Extract audio data - VoiceBench provides it as dict
            audio_array = audio['array'].astype(np.float32)
            
            # Ensure it's mono and reshape to (1, T) as the API expects
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.reshape(1, -1)
            
            # Validate audio data
            if audio_array.size == 0:
                bt.logging.warning("Empty audio array provided")
                return ""
            
            # Serialize to base64 (matching API format)
            # Use tobytes() instead of np.save to avoid pickle issues
            buf = io.BytesIO()
            np.save(buf, audio_array, allow_pickle=False)
            payload = {
                "audio_data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                "sample_rate": audio.get('sampling_rate', 16000)  # Use original sample rate or default to 16kHz
            }
            
            bt.logging.debug(f"Sending audio request to {self.api_url} with sample_rate: {payload['sample_rate']}")
            
            # API call
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            data = response.json()
            text_response = data.get("text", "")
            
            if not text_response:
                bt.logging.warning(f"Empty response from API: {data}")
                return ""
            
            bt.logging.debug(f"Received response: {text_response}...")
            return text_response.strip()
            
        except requests.exceptions.Timeout:
            bt.logging.error(f"Request timeout after {self.timeout} seconds")
            return ""
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Request error calling miner model API: {e}")
            return ""
        except Exception as e:
            bt.logging.error(f"Error calling miner model API: {e}")
            return ""
    
    def generate_text(self, text):
        """
        Generate response from text input.
        
        Since this is a voice-to-text model, it doesn't support text input.
        
        Args:
            text: Text input (not supported)
            
        Returns:
            str: Error message indicating audio-only support
        """
        bt.logging.warning("Text input requested for audio-only model")
        return "Error: This model only supports audio input"
    
    def generate_ttft(self, audio):
        """
        Generate response for time-to-first-token measurement.
        
        Args:
            audio: Audio input dict
            
        Returns:
            str: Text response
        """
        return self.generate_audio(audio)


class DynamicMinerModelAssistant(MinerModelAssistant):
    """
    Dynamic version that can adapt to different miner model API endpoints.
    
    This version can be configured for different miner models with different
    API endpoints and formats.
    """
    
    def __init__(self, base_url="http://localhost", port=8010, endpoint="/api/v1/v2t", timeout=600):
        """
        Initialize with flexible URL construction.
        
        Args:
            base_url: Base URL of the miner model service
            port: Port number
            endpoint: API endpoint path
            timeout: Request timeout
        """
        api_url = f"{base_url}:{port}{endpoint}"
        super().__init__(api_url, timeout)
        self.base_url = base_url
        self.port = port
        self.endpoint = endpoint
    
    def update_endpoint(self, new_endpoint):
        """Update the API endpoint dynamically"""
        self.endpoint = new_endpoint
        self.api_url = f"{self.base_url}:{self.port}{new_endpoint}"
        bt.logging.info(f"Updated API endpoint to: {self.api_url}")


def create_miner_assistant(model_config: Dict[str, Any]) -> MinerModelAssistant:
    """
    Factory function to create miner model assistants based on configuration.
    
    Args:
        model_config: Configuration dict with API details
        
    Returns:
        MinerModelAssistant: Configured assistant instance
    """
    if 'api_url' in model_config:
        # Direct URL configuration
        return MinerModelAssistant(
            api_url=model_config['api_url'],
            timeout=model_config.get('timeout', 600)
        )
    else:
        # Component-based URL construction
        return DynamicMinerModelAssistant(
            base_url=model_config.get('base_url', 'http://localhost'),
            port=model_config.get('port', 8010),
            endpoint=model_config.get('endpoint', '/api/v1/v2t'),
            timeout=model_config.get('timeout', 600)
        )


# Example usage configurations
EXAMPLE_CONFIGS = {
    'landcruiser_model': {
        'api_url': 'http://localhost:8010/api/v1/v2t',
        'timeout': 600
    },
    'generic_miner': {
        'base_url': 'http://localhost',
        'port': 8010,
        'endpoint': '/api/v1/inference',
        'timeout': 300
    }
}