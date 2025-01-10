import os
import signal
import logging
import sys
from pathlib import Path
from typing import Optional
from socket import socket, AF_UNIX, SOCK_STREAM
from subprocess import Popen
import json
from huggingface_hub import HfApi
import torch

from .protocol import SandboxProtocol
from .security import verify_sandbox_context

logger = logging.getLogger(__name__)

# Constants
MAX_HF_MODEL_SIZE_GB = 20

class SandboxManager:
    def __init__(self, model_config, sandbox_config):
        """Initialize sandbox manager"""
        self.model_config = model_config
        self.sandbox_config = sandbox_config
        self._socket: Optional[socket] = None
        self._process: Optional[Popen] = None
        self._protocol: Optional[SandboxProtocol] = None
        self.hf_api = HfApi()

    def _download_model(self) -> None:
        """Download model from HuggingFace with size verification"""
        try:
            # Get model info and check size
            model_info = self.hf_api.model_info(
                self.model_config.model_id,
                revision=self.model_config.revision,
                files_metadata=True
            )
            
            total_size = sum(sibling.size for sibling in model_info.siblings)
            if total_size > MAX_HF_MODEL_SIZE_GB * 1024 ** 3:
                raise RuntimeError(
                    f"Model size exceeds {MAX_HF_MODEL_SIZE_GB} GB limit"
                )

            # Clear old model if exists
            model_path = self.sandbox_config.sandbox_dir / "model"
            if model_path.exists():
                import shutil
                shutil.rmtree(str(model_path))
            
            # Download model
            local_dir = self.hf_api.snapshot_download(
                repo_id=self.model_config.model_id,
                revision=self.model_config.revision,
                local_dir=model_path,
                cache_dir=self.sandbox_config.sandbox_dir / ".cache/huggingface",
                max_workers=4,  # Limit concurrent downloads
                local_files_only=False
            )
            
            # Verify model files exist
            required_files = ["inference.py"]
            missing = [f for f in required_files if not (model_path / f).exists()]
            if missing:
                raise RuntimeError(f"Missing required files: {missing}")
            
            logger.info(f"Model downloaded to {local_dir}")

        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
        
    def setup(self):
        """Initialize sandbox environment"""
        try:
            logger.info("Setting up sandbox environment...")
            
            # Create necessary directories
            os.makedirs(str(self.sandbox_config.sandbox_dir), exist_ok=True)
            os.makedirs(str(self.sandbox_config.sandbox_dir / "model"), exist_ok=True)
            
            # Download model
            logger.info("Downloading model...")
            self._download_model()
            logger.info("Model downloaded successfully")

            # Setup Unix socket
            socket_path = self.sandbox_config.socket_path
            if socket_path.exists():
                os.unlink(str(socket_path))
                
            self._socket = socket(AF_UNIX, SOCK_STREAM)
            self._socket.bind(str(socket_path))
            os.chmod(str(socket_path), 0o770)
            
            # Initialize protocol
            self._protocol = SandboxProtocol(self._socket)
            logger.info("Sandbox setup completed successfully")
            
        except Exception as e:
            logger.error(f"Sandbox setup failed: {str(e)}", exc_info=True)
            self.cleanup()
            raise RuntimeError(f"Sandbox setup failed: {e}")
            
    def start(self):
        """Start sandboxed inference process"""
        
        try:
            inference_script = Path(__file__).parent / "process.py"
            model_path = str(self.sandbox_config.sandbox_dir / "model")
            
            # Setup sandbox environment vars
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{model_path}:{env.get('PYTHONPATH', '')}"
            env["HOME"] = "/sandbox"
            
            # Start process
            self._process = Popen([
                "python3", str(inference_script),
                "--socket", str(self.sandbox_config.socket_path),
                "--model-path", model_path,
                "--device", self.model_config.device
            ], env=env, start_new_session=True)
            
            # Accept connection
            self._socket.listen(1)
            self._protocol.accept_connection()
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start inference: {e}")
 
    def predict(self, tensor: torch.Tensor, sample_rate: int = 24000) -> dict:
        """Run inference in sandbox"""
        if not self._protocol or not self._protocol.is_connected():
            raise RuntimeError("Sandbox not started")
            
        return self._protocol.run_inference(tensor, sample_rate)
    
    def cleanup(self):
        """Cleanup sandbox resources"""
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
            self._process = None

        if self._protocol:
            self._protocol.close()
            self._protocol = None
            
        if self._socket:
            try:
                self._socket.close()
                os.unlink(str(self.sandbox_config.socket_path))
            except Exception as e:
                logger.error(f"Socket cleanup error: {e}")
            self._socket = None

        # Clean model files if needed
        try:
            model_path = self.sandbox_config.sandbox_dir / "model"
            if model_path.exists():
                import shutil
                shutil.rmtree(str(model_path))
        except Exception as e:
            logger.error(f"Model cleanup error: {e}")