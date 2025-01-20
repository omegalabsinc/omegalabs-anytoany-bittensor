import os
import time
import logging
import numpy as np
import torch
import json
import requests
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager

from scoring.docker_manager import DockerManager
from evaluation.S2S.distance import S2SMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_REPO_ID = "tezuesh/moshi_general"
TEST_AUDIO_LENGTH = 48000  # 3 seconds at 16kHz
SAMPLE_RATE = 24000
MAX_STARTUP_RETRIES = 5
HEALTH_CHECK_TIMEOUT = 180
INFERENCE_TIMEOUT = 60

class DockerInferenceTester:
    """
    Comprehensive testing framework for Docker-based model inference.
    Handles container lifecycle, resource management, and validation.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("./cache")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.docker_manager = DockerManager(base_cache_dir=str(self.base_dir))
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Ensure complete resource cleanup"""
        try:
            self.docker_manager.cleanup_docker_resources()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    @staticmethod
    def create_test_audio() -> np.ndarray:
        """
        Generate test audio signal with proper format for inference testing.
        Returns:
            np.ndarray: Audio signal shaped as [channels=1, samples]
        """
        duration = TEST_AUDIO_LENGTH / SAMPLE_RATE
        t = np.linspace(0, duration, TEST_AUDIO_LENGTH)
        frequency = 440  # Hz A4 note
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio.reshape(1, -1).astype(np.float32)
        
    @staticmethod
    def validate_container_health(container_url: str) -> bool:
        """
        Validate container health status with retries.
        Args:
            container_url: Base URL for container health checks
        Returns:
            bool: True if container is healthy
        """
        health_url = f"{container_url}/api/v1/health"
        start_time = time.time()
        
        while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
            try:
                response = requests.get(health_url, timeout=5)
                health_data = response.json()
                
                if health_data.get("status") == "healthy":
                    logger.info("Container health check passed")
                    logger.debug(f"Health data: {json.dumps(health_data, indent=2)}")
                    return True
                    
                if "initialization_status" in health_data:
                    init_status = health_data["initialization_status"]
                    if init_status.get("error"):
                        logger.error(f"Container initialization failed: {init_status['error']}")
                        return False
                        
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check attempt failed: {e}")
                
            time.sleep(5)
            
        logger.error("Container health check timed out")
        return False
        
    def validate_inference_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate inference result structure and content.
        Args:
            result: Inference output dictionary
        Returns:
            bool: True if result is valid
        """
        try:
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict result, got {type(result)}")
                
            if 'audio' not in result:
                raise ValueError("Missing 'audio' key in result")
                
            audio = result['audio']
            if not isinstance(audio, np.ndarray):
                raise ValueError(f"Expected numpy array for audio, got {type(audio)}")
                
            if len(audio.shape) != 2:
                raise ValueError(f"Expected 2D audio array [C,T], got shape {audio.shape}")
                
            logger.info(f"Inference result validation passed: shape={audio.shape}, "
                       f"dtype={audio.dtype}, range=[{audio.min():.3f}, {audio.max():.3f}]")
            return True
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            return False
            
    def test_container_setup(self) -> Tuple[bool, Optional[str]]:
        """
        Test Docker container setup and initialization.
        Returns:
            Tuple[bool, Optional[str]]: Success status and container URL if successful
        """
        logger.info("Testing container setup...")
        
        try:
            # Configure GPU if available
            gpu_id = 0 if torch.cuda.is_available() else None
            gpu_info = f"GPU#{gpu_id}" if gpu_id is not None else "CPU"
            logger.info(f"Running on {gpu_info}")
            
            # Start container with retries
            for attempt in range(MAX_STARTUP_RETRIES):
                try:
                    container_url = self.docker_manager.start_container(
                        uid="test_model",
                        repo_id=TEST_REPO_ID,
                        gpu_id=gpu_id
                    )
                    
                    if not container_url:
                        raise RuntimeError("Container URL not returned")
                        
                    if self.validate_container_health(container_url):
                        logger.info(f"Container started successfully: {container_url}")
                        return True, container_url
                        
                except Exception as e:
                    logger.warning(f"Startup attempt {attempt + 1} failed: {e}")
                    self.cleanup()
                    time.sleep(5)
                    
            logger.error("Container setup failed after maximum retries")
            return False, None
            
        except Exception as e:
            logger.error(f"Container setup failed: {e}")
            return False, None
            
    def test_inference(self, container_url: str) -> bool:
        """
        Test model inference with validation.
        Args:
            container_url: Container endpoint URL
        Returns:
            bool: True if inference test passes
        """
        logger.info("Testing inference...")
        
        try:
            test_audio = self.create_test_audio()
            logger.info(f"Generated test audio: shape={test_audio.shape}, "
                       f"sr={SAMPLE_RATE}, duration={len(test_audio[0])/SAMPLE_RATE:.2f}s")
            breakpoint()
            
            result = self.docker_manager.inference(
                url=container_url,
                audio_array=test_audio,
                sample_rate=SAMPLE_RATE,
                timeout=INFERENCE_TIMEOUT
            )
            breakpoint()
            
            return self.validate_inference_result(result)
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return False
            
    def run_all_tests(self) -> bool:
        logger.info("Starting test suite...")
        
        try:
            # Test container setup
            setup_success, container_url = self.test_container_setup()
            if not setup_success or not container_url:
                return False
                
            # Validate CUDA first
            # self.validate_cuda()    # Add this line
                
            # Test inference
            if not self.test_inference(container_url):
                return False
                
            logger.info("All tests passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False
            
        finally:
            self.cleanup()

def main():
    """Main entry point for test execution"""
    try:
        with DockerInferenceTester() as tester:
            success = tester.run_all_tests()
            exit_code = 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        exit_code = 1
        
    exit(exit_code)

if __name__ == "__main__":
    main()