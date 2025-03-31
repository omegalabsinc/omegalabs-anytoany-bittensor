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
import socket
from typing import Optional, Dict, Any, List, Tuple
from docker.models.containers import Container
import bittensor as bt

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
        
        bt.logging.info(f"Initializing DockerManager with cache directory: {self.base_cache_dir}")
        
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
            
            bt.logging.info(f"Downloading files from {repo_id} to {miner_dir}")
            
            try:
                downloaded_path = huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    local_dir=miner_dir,
                    local_dir_use_symlinks=False
                )
                bt.logging.info(f"Downloaded files to {downloaded_path}")
            except Exception as e:
                bt.logging.error(f"Failed to download repo: {str(e)}")
                raise
                
            return miner_dir
                
        except Exception as e:
            bt.logging.error(f"Failed to download miner files: {str(e)}")
            raise

    def get_memory_limit(self):
        total_memory = psutil.virtual_memory().total
        memory_limit = int(total_memory * 0.9)
        return memory_limit

    def _check_disk_space(self, required_gb: int = 20) -> None:
        """Check available disk space and clean if necessary."""
        total, used, free = shutil.disk_usage(str(self.base_cache_dir))
        free_gb = free / (1024 * 1024 * 1024)
        
        bt.logging.info(f"Available disk space: {free_gb:.2f}GB")
        
        if free_gb < required_gb:
            bt.logging.warning(f"Low disk space ({free_gb:.2f}GB free), cleaning cache directory")
            # Clean Docker resources first
            self.cleanup_docker_resources(aggressive=True)
            
            # Clear HuggingFace cache if exists
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(hf_cache):
                bt.logging.info(f"Cleaning HuggingFace cache: {hf_cache}")
                try:
                    shutil.rmtree(hf_cache)
                except Exception as e:
                    bt.logging.error(f"Failed to clean HuggingFace cache: {str(e)}")
            
            # Clear our cache directory
            for item in os.listdir(str(self.base_cache_dir)):
                item_path = os.path.join(str(self.base_cache_dir), item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        bt.logging.info(f"Removed directory: {item_path}")
                    except Exception as e:
                        bt.logging.error(f"Failed to remove directory {item_path}: {str(e)}")
                else:
                    try:
                        os.remove(item_path)
                        bt.logging.info(f"Removed file: {item_path}")
                    except Exception as e:
                        bt.logging.error(f"Failed to remove file {item_path}: {str(e)}")
            
            # Recreate necessary directories
            self.base_cache_dir.mkdir(parents=True, exist_ok=True)
            for dir_name in ['models', 'hub', 'downloads']:
                (self.base_cache_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
            # Check if we have enough space now
            _, _, free = shutil.disk_usage(str(self.base_cache_dir))
            free_gb = free / (1024 * 1024 * 1024)
            bt.logging.info(f"Available disk space after cleanup: {free_gb:.2f}GB")
            
            if free_gb < required_gb:
                bt.logging.error(f"Still insufficient disk space ({free_gb:.2f}GB) after cleanup")
                raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB available, {required_gb}GB required")

    def _find_available_port(self, start_port: int = 8000, max_port: int = 9000) -> int:
        """
        Find an available port in the given range.
        
        Args:
            start_port: Starting port number to check
            max_port: Maximum port number to check
            
        Returns:
            Available port number
        """
        for port in range(start_port, max_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

    def start_container(self, uid: str, repo_id: str, gpu_id: Optional[int] = None) -> str:
        """Start a container with proper GPU configuration."""
        container_name = f"miner_{uid}"
        
        try:
            # Check disk space with higher requirement (40GB)
            self._check_disk_space(required_gb=40)
            
            # Create necessary cache subdirectories
            cache_dirs = ['models', 'hub', 'downloads']
            for dir_name in cache_dirs:
                (self.base_cache_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
            # Force pruning before building
            self.cleanup_docker_resources(aggressive=True)
            
            # Remove existing container if any
            try:
                old_container = self.client.containers.get(container_name)
                bt.logging.info(f"Removing existing container: {container_name}")
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
            bt.logging.info(f"Building container image: {container_name}")
            start = time.time()
            dockerfile_path = str(miner_dir / "Dockerfile")
            self._validate_dockerfile(dockerfile_path)
            
            # Check available disk space before building
            total, used, free = shutil.disk_usage(str(self.base_cache_dir))
            free_gb = free / (1024 * 1024 * 1024)
            bt.logging.info(f"Available disk space before build: {free_gb:.2f}GB")
            
            image, build_logs = self.client.images.build(
                path=str(miner_dir),
                dockerfile=dockerfile_path,
                tag=f"{container_name}:latest",
                rm=True,
                nocache=True,  # Force no cache to avoid using corrupt layers
                forcerm=True   # Force remove intermediate containers
            )
            bt.logging.info(f"Finished building image for {container_name} in {time.time() - start} seconds")
            
            # Find an available port
            host_port = self._find_available_port()
            print(f"Using available port: {host_port}")
            bt.logging.info(f"Using available port: {host_port}")
            
            bt.logging.info(f"Starting container: {container_name}")
            container = self.client.containers.run(
                image.id,
                name=container_name,
                detach=True,
                user='nobody',
                security_opt=[
                    'no-new-privileges:true',
                ],
                cap_drop=['ALL'],
                mem_limit=self.get_memory_limit(),
                ports={'8000/tcp': ('0.0.0.0', host_port)},
                environment={
                    'CUDA_VISIBLE_DEVICES': str(gpu_id) if gpu_id is not None else "all",
                    'PYTHONUNBUFFERED': '1',
                    'MODEL_ID': repo_id,
                    'REPO_ID': repo_id,
                    'NVIDIA_VISIBLE_DEVICES': 'all',
                    'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility,graphics'
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
            
            return f"http://localhost:{host_port}"

        except Exception as e:
            bt.logging.error(f"Failed to start container: {str(e)}")
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
                bt.logging.info(f"Stopping container: {container.name}")
                container.stop(timeout=10)
                container.remove(force=True)
                del self.active_containers[uid]
        except Exception as e:
            bt.logging.error(f"inside docker_manager: Failed to stop container: {str(e)}")
            raise

    def cleanup_docker_resources(self, aggressive: bool = False) -> None:
        """Clean up all Docker resources."""
        try:
            images_to_clean = []

            # Stop all active containers
            for uid in list(self.active_containers.keys()):
                images_to_clean.append(self.active_containers[uid].image.id)
                self.stop_container(uid)

            # Remove all stopped containers
            containers = self.client.containers.list(all=True)
            for container in containers:
                try:
                    container.remove(force=True)
                except Exception as e:
                    bt.logging.warning(f"inside docker_manager: Failed to remove container {container.name}: {str(e)}")

            for image_id in images_to_clean:
                try:
                    self.client.images.remove(image_id, force=True)
                    bt.logging.debug(f"inside docker_manager: Removed image {image_id}")
                except Exception as e:
                    bt.logging.warning(f"inside docker_manager: Failed to remove image {image_id}: {str(e)}")

            # Prune resources
            self.client.images.prune(filters={'dangling': True})
            self.client.containers.prune()
            self.client.volumes.prune()
            
            # If aggressive is set, remove all unused images as well
            if aggressive:
                bt.logging.info("Performing aggressive Docker cleanup")
                try:
                    unused_images = self.client.images.list()
                    for image in unused_images:
                        if not image.tags:  # Remove untagged images
                            try:
                                self.client.images.remove(image.id, force=True)
                                bt.logging.debug(f"Removed untagged image {image.id}")
                            except Exception as e:
                                bt.logging.warning(f"Failed to remove image {image.id}: {str(e)}")
                    
                    # Force system prune to clean everything
                    import subprocess
                    subprocess.run(["docker", "system", "prune", "-af", "--volumes"], check=False)
                    bt.logging.info("Performed docker system prune")
                except Exception as e:
                    bt.logging.warning(f"Error during aggressive cleanup: {str(e)}")

            bt.logging.info("inside docker_manager: Docker resources cleaned up successfully")

        except Exception as e:
            bt.logging.error(f"inside docker_manager: Failed to cleanup Docker resources: {str(e)}")
            raise

    def inference_v2v(self, url: str, audio_array: np.ndarray, sample_rate: int, timeout: int = 60) -> Dict[str, Any]:
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
            bt.logging.error(f"Inference request failed: {str(e)}")
            raise
        except Exception as e:
            bt.logging.error(f"inside docker_manager: Failed to process inference result: {str(e)}")
            raise
    

    def inference_ibllama(self, url: str, video_embed: List[float], timeout: int = 60) -> Dict[str, Any]:
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
            bt.logging.error(f"Inference request failed: {str(e)}")
            raise
        except Exception as e:
            bt.logging.error(f"inside docker_manager: Failed to process inference result: {str(e)}")
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
        
        # Get the actual port mapping from container
        container.reload()
        ports = container.attrs['NetworkSettings']['Ports']
        if not ports or '8000/tcp' not in ports:
            raise RuntimeError("Container port mapping not found")
            
        host_port = ports['8000/tcp'][0]['HostPort']
        health_check_url = f"http://localhost:{host_port}/api/v1/health"
        
        bt.logging.info(f"Checking container health at {health_check_url}")
        
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
                        bt.logging.info("Container is healthy and ready")
                        return
                    elif "initialization_status" in health_data:
                        init_status = health_data["initialization_status"]
                        if init_status.get("error"):
                            bt.logging.warning(f"Initialization error: {init_status['error']}")
                
            except requests.exceptions.RequestException as e:
                bt.logging.debug(f"Health check not ready: {str(e)}")
            except Exception as e:
                bt.logging.warning(f"Health check error: {str(e)}")
            
            time.sleep(2)
            
        # If we get here, we've timed out
        logs = container.logs().decode('utf-8')
        raise TimeoutError(
            f"Container failed to become healthy within {timeout} seconds.\n"
            f"Container logs:\n{logs}"
        )
