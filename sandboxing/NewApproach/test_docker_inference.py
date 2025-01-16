import os
import time
import logging
import numpy as np
import torch
import json
import requests
from pathlib import Path

from scoring.docker_manager import DockerManager
from evaluation.S2S.distance import S2SMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Enable HF transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Test configuration
TEST_REPO_ID = "tezuesh/moshi_general"  # Updated to use the correct model repo
TEST_AUDIO_LENGTH = 48000  # 3 seconds at 16kHz
SAMPLE_RATE = 24000  # Updated to match model's expected sample rate

def create_test_audio():
    """Create a simple test audio signal"""
    duration = TEST_AUDIO_LENGTH / SAMPLE_RATE
    t = np.linspace(0, duration, TEST_AUDIO_LENGTH)
    # Generate a simple sine wave
    frequency = 440  # Hz
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def setup_test_directories():
    """Setup necessary directories for testing"""
    base_dir = Path("./cache")
    base_dir.mkdir(exist_ok=True, parents=True)
    return base_dir

def test_docker_container():
    """Test Docker container setup and API endpoints"""
    logger.info("Testing Docker container setup...")
    
    # Setup directories
    base_dir = setup_test_directories()
    
    # Initialize Docker manager
    docker_manager = DockerManager(base_cache_dir=str(base_dir))
    
    try:
        # Start container with the test model
        container_url = docker_manager.start_container(
            uid="test_model",
            repo_id=TEST_REPO_ID,
            gpu_id=0  # Specify GPU if available
        )
        
        if not container_url:
            logger.error("inside test_docker_container: Failed to start container")
            return False
            
        logger.info(f"inside test_docker_container: Container started successfully at {container_url}")
        # breakpoint()
        
        # Test health endpoint
        health_url = f"{container_url}/api/v1/health"
        retries = 0
        max_retries = 10  # Increased retries for model loading
        
        while retries < max_retries:
            try:
                response = requests.get(health_url)
                health_data = response.json()
                
                if health_data["status"] == "healthy":
                    logger.info("Health check passed")
                    logger.info(f"Health status: {health_data}")
                    break
                
                logger.info("Model still initializing, waiting...")
                time.sleep(20)  # Increased wait time
                retries += 1
                
            except Exception as e:
                logger.warning(f"Health check attempt {retries + 1} failed: {e}")
                time.sleep(10)
                retries += 1
                
        if retries >= max_retries:
            logger.error("Health check failed after maximum retries")
            return False
            
        # Test inference
        test_audio = create_test_audio()
        try:
            result = docker_manager.inference(
                container_url,
                test_audio,
                SAMPLE_RATE
            )
            
            # Validate result
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "audio" in result, "Result should contain 'audio' key"
            assert isinstance(result["audio"], np.ndarray), "Audio should be numpy array"
            
            # Log output audio properties
            logger.info(f"Output audio shape: {result['audio'].shape}")
            logger.info(f"Output audio min/max: {result['audio'].min():.3f}/{result['audio'].max():.3f}")
            
            logger.info("Inference test passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Container test failed: {e}")
        raise
        
    finally:
        logger.info("Cleaning up container...")
        docker_manager.stop_container("test_model")

def test_scoring_pipeline():
    """Test the complete scoring pipeline"""
    logger.info("Testing scoring pipeline...")
    
    base_dir = setup_test_directories()
    docker_manager = DockerManager(base_cache_dir=str(base_dir))
    metrics = S2SMetrics(cache_dir=str(base_dir))
    
    try:
        # Start container
        container_url = docker_manager.start_container(
            uid="test_model",
            repo_id=TEST_REPO_ID,
            gpu_id=0
        )
        
        if not container_url:
            logger.error("inside test_scoring_pipeline: Failed to start container")
            return False
        
        # Create test input
        test_audio = create_test_audio()
        
        # Run inference
        result = docker_manager.inference(
            container_url,
            test_audio,
            SAMPLE_RATE
        )
        
        # Test scoring
        gt_audio_arrs = [(test_audio, SAMPLE_RATE)]
        generated_audio_arrs = [(result['audio'], SAMPLE_RATE)]
        
        scores = metrics.compute_distance(gt_audio_arrs, generated_audio_arrs)
        
        logger.info("Scoring results:")
        for metric, value in scores.items():
            logger.info(f"{metric}: {value}")
            
        return True
        
    except Exception as e:
        logger.error(f"Scoring test failed: {e}")
        raise
        
    finally:
        docker_manager.stop_container("test_model")

def main():
    """Run all tests"""
    logger.info("Starting tests...")
    
    try:
        # Test Docker container and API
        container_success = test_docker_container()
        
        if container_success:
            # Test scoring pipeline
            scoring_success = test_scoring_pipeline()
            
            if scoring_success:
                logger.info("All tests passed successfully!")
            else:
                logger.error("Scoring pipeline test failed")
        else:
            logger.error("Docker container test failed")
            
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Tests failed with error: {e}")
        raise

if __name__ == "__main__":
    main()