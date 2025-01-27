import docker
import requests 
import time
import numpy as np
import base64
import io
import os
import logging
from pathlib import Path
import huggingface_hub
import psutil
import shutil
from typing import Optional, Dict, Any, List
from docker.models.containers import Container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Enable HF transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class DockerManager:
    """Docker container manager for model inference with improved error handling and resource management."""
    
    def __init__(self, base_cache_dir: str = "./cache", cleanup_on_init: bool = True):
        """
        Initialize Docker manager.
        
        Args:
            base_cache_dir: Base directory for caching models and docker files
            cleanup_on_init: Whether to cleanup Docker resources on initialization
        """
        self.client = docker.from_env()
        self.base_cache_dir = Path(os.path.abspath(base_cache_dir))
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        self.active_containers: Dict[str, Container] = {}
        
        logger.info(f"Initializing DockerManager with cache directory: {self.base_cache_dir}")
        
        if cleanup_on_init:
            self.cleanup_docker_resources()
    
    def _download_miner_files(self, repo_id: str, uid: str) -> Path:
        """
        Downloads required files from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            uid: Unique identifier for the download
            
        Returns:
            Path to downloaded files
        """
        try:
            repo_name = repo_id.replace("/", "_")
            miner_dir = self.base_cache_dir / repo_name
            miner_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading files from {repo_id} to {miner_dir}")
            
            try:
                downloaded_path = huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    local_dir=miner_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded files to {downloaded_path}")
            except Exception as e:
                logger.error(f"Failed to download repo: {str(e)}")
                raise
                
            return miner_dir
                
        except Exception as e:
            logger.error(f"Failed to download miner files: {str(e)}")
            raise

    def get_memory_limit(self):
        total_memory = psutil.virtual_memory().total
        memory_limit = int(total_memory * 0.9)
        return memory_limit

    def _check_disk_space(self, required_gb: int = 10) -> None:
        """Check available disk space and clean if necessary."""
        total, used, free = shutil.disk_usage(str(self.base_cache_dir))
        free_gb = free / (1024 * 1024 * 1024)
        
        if free_gb < required_gb:
            logger.warning(f"Low disk space ({free_gb:.2f}GB free), cleaning cache directory")
            shutil.rmtree(str(self.base_cache_dir))
            self.base_cache_dir.mkdir(parents=True, exist_ok=True)
  
    def start_container(self, uid: str, repo_id: str, gpu_id: Optional[int] = None) -> str:
        """Start a container with proper GPU configuration."""
        container_name = f"miner_{uid}"
        
        try:
            # Check disk space
            self._check_disk_space()
            
            # Create necessary cache subdirectories
            cache_dirs = ['models', 'hub', 'downloads']
            for dir_name in cache_dirs:
                (self.base_cache_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
            # Remove existing container if any
            try:
                old_container = self.client.containers.get(container_name)
                logger.info(f"Removing existing container: {container_name}")
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Download miner files
            miner_dir = self._download_miner_files(repo_id, uid)
            
            # Create isolated network if it doesn't exist
            try:
                self.client.networks.get('isolated_network')
            except docker.errors.NotFound:
                self.client.networks.create('isolated_network', driver='bridge', internal=True)
            
            # Build and start container with GPU support
            logger.info(f"Building container image: {container_name}")
            start = time.time()
            dockerfile_path = str(miner_dir / "Dockerfile")
            self._validate_dockerfile(dockerfile_path)
            image, build_logs = self.client.images.build(
                path=str(miner_dir),
                dockerfile=dockerfile_path,
                tag=f"{container_name}:latest",
                rm=True
            )
            logger.info(f"Finished building image for {container_name} in {time.time() - start} seconds")
            
            logger.info(f"Starting container: {container_name}")
            container = self.client.containers.run(
                image.id,
                name=container_name,
                detach=True,
                user='nobody',
                security_opt=[
                    'no-new-privileges:true',
                    'seccomp=default-docker-profile.json'
                ],
                cap_drop=['ALL'],
                mem_limit=self.get_memory_limit(),
                ports={'8000/tcp': ('0.0.0.0', 8000)},
                environment={
                    'CUDA_VISIBLE_DEVICES': str(gpu_id) if gpu_id is not None else "all",
                    'MODEL_PATH': '/app/cache',
                    'HF_HOME': '/app/cache',
                    'PYTHONUNBUFFERED': '1',
                    'MODEL_ID': repo_id,
                    'REPO_ID': repo_id,
                    'NVIDIA_VISIBLE_DEVICES': 'all',
                    'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility,graphics'
                },
                volumes={
                    str(miner_dir): {
                        'bind': '/app/src',
                        'mode': 'ro',
                        'propagation': 'private'
                    },
                    str(self.base_cache_dir): {
                        'bind': '/app/cache',
                        'mode': 'rw',
                        'propagation': 'private'
                    }
                },
                runtime='nvidia',
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[['gpu', 'utility', 'compute']]
                    )
                ]
            )
            
            # Wait for container to be healthy
            self._wait_for_container(container)
            
            # Store container reference
            self.active_containers[uid] = container
            
            return "http://localhost:8000"

        except Exception as e:
            logger.error(f"Failed to start container: {str(e)}")
            self.stop_container(uid)
            raise

    def _validate_dockerfile(self, dockerfile_path: Path):
        with open(dockerfile_path) as f:
            content = f.read()
            if any(dangerous_cmd in content.lower() for dangerous_cmd in [
                'privileged',
                'host',
                'sudo',
                'chmod 777',
                '/var/run/docker.sock'
            ]):
                raise Exception("Dockerfile contains dangerous commands")

    def stop_container(self, uid: str) -> None:
        """Stop and remove container."""
        try:
            if uid in self.active_containers:
                container = self.active_containers[uid]
                logger.info(f"Stopping container: {container.name}")
                container.stop(timeout=10)
                container.remove(force=True)
                del self.active_containers[uid]
        except Exception as e:
            logger.error(f"inside docker_manager: Failed to stop container: {str(e)}")
            raise

    def cleanup_docker_resources(self) -> None:
        """Clean up all Docker resources."""
        try:
            # Stop all active containers
            for uid in list(self.active_containers.keys()):
                self.stop_container(uid)
            
            # Remove all stopped containers
            containers = self.client.containers.list(all=True)
            for container in containers:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"inside docker_manager: Failed to remove container {container.name}: {str(e)}")

            # Prune resources
            self.client.images.prune()
            self.client.containers.prune()
            self.client.volumes.prune()
            
            logger.info("inside docker_manager: Docker resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"inside docker_manager: Failed to cleanup Docker resources: {str(e)}")
            raise

    def inference_v2v(self, url: str, audio_array: np.ndarray, sample_rate: int, timeout: int = 30) -> Dict[str, Any]:
        """
        Send inference request to container.
        
        Args:
            url: Container API URL
            audio_array: Input audio array
            sample_rate: Audio sample rate
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing inference results
        """
        try:
            # Convert audio array to base64
            buffer = io.BytesIO()
            np.save(buffer, audio_array)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Send request
            response = requests.post(
                f"{url}/api/v1/inference",
                json={
                    "audio_data": audio_b64,
                    "sample_rate": sample_rate
                },
                timeout=timeout
            )
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            audio_bytes = base64.b64decode(result["audio_data"])
            audio = np.load(io.BytesIO(audio_bytes))
            
            return {
                "audio": audio,
                "text": result.get("text", "")
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"inside docker_manager: Failed to process inference result: {str(e)}")
            raise
    

    def inference_ibllama(self, url: str, video_embed: List[float], timeout: int = 30) -> Dict[str, Any]:
        """
        Send inference request to container for IBLlama model.
        
        Args:
            url: Container API URL
            video_embed: Input video embedding as list of floats
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing inference results with generated captions
        """
        try:
            # Send request
            response = requests.post(
                f"{url}/api/v1/inference",
                json={
                    "embedding": video_embed
                },
                timeout=timeout
            )
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return {
                "captions": result.get("texts", [])
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"inside docker_manager: Failed to process inference result: {str(e)}")
            raise

    def _wait_for_container(self, container: Container, timeout: int = 180) -> None:
        """
        Implements a robust health check mechanism for container initialization.
        
        Args:
            container: Docker container instance
            timeout: Maximum wait time in seconds
            
        Raises:
            TimeoutError: If container fails to become healthy within timeout
            RuntimeError: If container health check fails
        """
        start_time = time.time()
        health_check_url = "http://localhost:8000/api/v1/health"
        
        while time.time() - start_time < timeout:
            try:
                # Check container state
                container.reload()
                container_state = container.attrs.get('State', {})
                status = container_state.get('Status', '')
                
                if status == 'exited':
                    logs = container.logs().decode('utf-8')
                    raise RuntimeError(f"Container exited unexpectedly. Logs:\n{logs}")
                
                # Try health check endpoint
                response = requests.get(health_check_url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        logging.info("Container is healthy and ready")
                        return
                    elif "initialization_status" in health_data:
                        init_status = health_data["initialization_status"]
                        if init_status.get("error"):
                            logging.warning(f"Initialization error: {init_status['error']}")
                
            except requests.exceptions.RequestException as e:
                logging.debug(f"Health check not ready: {str(e)}")
            except Exception as e:
                logging.warning(f"Health check error: {str(e)}")
            
            time.sleep(2)
            
        # If we get here, we've timed out
        logs = container.logs().decode('utf-8')
        raise TimeoutError(
            f"Container failed to become healthy within {timeout} seconds.\n"
            f"Container logs:\n{logs}"
        )