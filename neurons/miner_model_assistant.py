
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
    
    def __init__(self, api_url="http://localhost:8000/api/v1/v2t", timeout=600):
        """
        Initialize the miner model assistant.
        
        Args:
            api_url: URL of the miner model API endpoint
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        bt.logging.info(f"Initialized MinerModelAssistant with API: {api_url}")
    
    def inference_v2t(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Send inference request to server (matching DockerManager interface).
        
        Args:
            audio_array: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing inference results
        """
        # Convert audio array to base64 (matching Docker flow)
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Send request
        response = requests.post(
            self.api_url,
            json={
                "audio_data": audio_b64,
                "sample_rate": sample_rate
            },
            timeout=self.timeout
        )
        
        response.raise_for_status()
        
        return response.json()
    
    def inference_v2v(self, url: str, audio_array: np.ndarray, sample_rate: int, timeout: int = 60) -> Dict[str, Any]:
        """
        Send voice-to-voice inference request to container.

        Args:
            url: Container API URL
            audio_array: Input audio array
            sample_rate: Audio sample rate
            timeout: Request timeout in seconds

        Returns:
            Dict containing inference results with audio and text
        """
        try:
            # Input audio preprocessing 
            processed_audio = audio_array
            processed_sample_rate = sample_rate

            # Ensure mono
            if processed_audio.ndim == 2:
                processed_audio = processed_audio.mean(axis=1)

            # Ensure float32
            processed_audio = processed_audio.astype(np.float32)

            # Resample to 16kHz if needed
            TARGET_SR = 16000
            if processed_sample_rate != TARGET_SR:
                import librosa
                processed_audio = librosa.resample(processed_audio, orig_sr=processed_sample_rate, target_sr=TARGET_SR)
                processed_sample_rate = TARGET_SR

            # Add channel dimension (1, T) - matching streamlit app
            processed_audio = processed_audio.reshape(1, -1)

            # Convert audio array to base64
            buffer = io.BytesIO()
            np.save(buffer, processed_audio)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Send request to v2v endpoint (not v2t!)
            response = requests.post(
                f"{url}/api/v1/v2v",
                json={
                    "audio_data": audio_b64,
                    "sample_rate": processed_sample_rate
                },
                timeout=timeout
            )

            response.raise_for_status()

            # Parse response
            result = response.json()
            response_data = {}

            # Get audio if available (for v2v models) - enhanced processing
            if "audio_data" in result:
                audio_bytes = base64.b64decode(result["audio_data"])
                audio = np.load(io.BytesIO(audio_bytes), allow_pickle=False)

                # Handle dimension (1, T) -> (T,) matching streamlit app
                if audio.ndim == 2:
                    audio = audio.squeeze(0)

                # Store numpy array for compatibility
                response_data["audio"] = audio

                # Convert to WAV bytes for file saving - matching streamlit app
                import soundfile as sf
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio, processed_sample_rate, format='WAV')
                wav_buf.seek(0)
                response_data["audio_wav_bytes"] = wav_buf.getvalue()
                response_data["audio_sample_rate"] = processed_sample_rate

            return response_data

        except requests.exceptions.RequestException as e:
            bt.logging.error(f"V2V inference request failed: {str(e)}")
            raise
        except Exception as e:
            bt.logging.error(f"inside docker_manager: Failed to process v2v inference result: {str(e)}")
            raise
    
    def generate_audio(self, audio, max_new_tokens=2048):
        """
        Generate text response from audio input (legacy interface).
        
        Args:
            audio: dict with 'array' (numpy array) and 'sampling_rate' keys
            max_new_tokens: Maximum tokens to generate (unused)
            
        Returns:
            str: Text response from the model
        """
        audio_array = np.array(audio['array'])
        sample_rate = audio.get('sampling_rate', 16000)
        
        result = self.inference_v2t(audio_array, sample_rate)
        
        text_response = result.get("text", "")
        if not text_response:
            bt.logging.warning(f"Empty response from API: {result}")
            return ""
        
        return text_response.strip()
    
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
    
    def __init__(self, base_url="http://localhost", port=8000, endpoint="/api/v1/v2t", timeout=600):
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
            port=model_config.get('port', 8000),
            endpoint=model_config.get('endpoint', '/api/v1/v2t'),
            timeout=model_config.get('timeout', 600)
        )


# Example usage configurations
EXAMPLE_CONFIGS = {
    'landcruiser_model': {
        'api_url': 'http://localhost:8000/api/v1/v2t',
        'timeout': 600
    },
    'generic_miner': {
        'base_url': 'http://localhost',
        'port': 8000,
        'endpoint': '/api/v1/inference',
        'timeout': 300
    }
}