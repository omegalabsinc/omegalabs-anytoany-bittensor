
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

from constants import V2T_TIMEOUT, V2V_TIMEOUT


class MinerModelAssistant():
    """
    Voice assistant that interfaces with miner model APIs.
    
    This class adapts miner models to work with VoiceBench evaluation
    by calling their HTTP APIs directly.
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        """
        Initialize the miner model assistant.
        
        Args:
            base_url: base URL of the miner model API endpoint
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        bt.logging.info(f"Initialized MinerModelAssistant with API: {base_url}")
    
    
    def inference_v2v(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
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

        # Ensure mono
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # Ensure float32
        audio_array = audio_array.astype(np.float32)

        # Resample to 16kHz if needed
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR

        # Add channel dimension (1, T) - matching streamlit app
        audio_array = audio_array.reshape(1, -1)

        # Convert audio array to base64
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Send request to v2v endpoint (not v2t!)
        response = requests.post(
            f"{self.base_url}/api/v1/v2v",
            json={
                "audio_data": audio_b64,
                "sample_rate": sample_rate
            },
            timeout=V2V_TIMEOUT
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
            sf.write(wav_buf, audio, sample_rate, format='WAV')
            wav_buf.seek(0)
            response_data["audio_wav_bytes"] = wav_buf.getvalue()
            response_data["audio_sample_rate"] = sample_rate
            if response_data["audio_wav_bytes"] is None:
                raise ValueError("Audio WAV bytes conversion failed")
            
            return response_data
        else:
            raise ValueError("No audio data returned from v2v API")


    def inference_v2t(self, audio_array: np.ndarray, sample_rate: int):
        """
        Generate text response from audio input
        
        Args:
            audio: dict with 'array' (numpy array) and 'sampling_rate' keys
            max_new_tokens: Maximum tokens to generate (unused)
            
        Returns:
            str: Text response from the model
        """
        
        # Convert audio array to base64 (matching Docker flow)
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Send request
        response = requests.post(
            f"{self.base_url}/api/v1/v2t",
            json={
                "audio_data": audio_b64,
                "sample_rate": sample_rate
            },
            timeout=V2T_TIMEOUT
        )
        
        response.raise_for_status()
        
        result = response.json()
        
        text_response = result.get("text", "")
        if not text_response:
            bt.logging.warning(f"Empty response from API: {result}")
            return ""
        
        return text_response.strip()
