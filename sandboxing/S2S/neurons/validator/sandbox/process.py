import os
import sys
import json
import struct
import argparse
import logging
import signal
from pathlib import Path
import importlib.util
import pwd
from socket import socket, AF_UNIX, SOCK_STREAM
import torch
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def receive_tensor(sock: socket) -> np.ndarray:
    """Receive tensor data efficiently"""
    try:
        # Receive tensor metadata
        meta_size = struct.unpack("!I", sock.recv(4))[0]
        meta_bytes = sock.recv(meta_size)
        meta = json.loads(meta_bytes)
        
        # Receive tensor data
        tensor_size = struct.unpack("!Q", sock.recv(8))[0]
        tensor_bytes = b""
        
        # Read in chunks
        while len(tensor_bytes) < tensor_size:
            chunk = sock.recv(min(8192, tensor_size - len(tensor_bytes)))
            if not chunk:
                raise RuntimeError("Connection closed while receiving tensor")
            tensor_bytes += chunk
        
        # Convert to numpy array
        array = np.frombuffer(tensor_bytes, dtype=np.float32)
        
        # Verify size matches expected shape
        expected_size = np.prod(meta["shape"])
        if array.size != expected_size:
            raise ValueError(f"Tensor size mismatch: got {array.size}, expected {expected_size}")
            
        return array.reshape(-1)  # Always return 1D array
        
    except Exception as e:
        logger.error(f"Error receiving tensor: {str(e)}")
        raise

def receive_metadata(sock: socket) -> dict:
    """Receive additional metadata"""
    size = struct.unpack("!I", sock.recv(4))[0]
    data = sock.recv(size)
    return json.loads(data)

def load_inference_module(model_path: str):
    """Load and initialize the inference module"""
    try:
        # Add model path to Python path
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
            
        # Load module
        spec = importlib.util.spec_from_file_location(
            "inference",
            os.path.join(model_path, "inference.py")
        )
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)
        logger.info("Successfully loaded inference module")
        
        # Check interface implementation
        if not hasattr(inference_module, 'InferenceRecipe'):
            raise RuntimeError("No InferenceRecipe class found")
        
        # Initialize with model path
        wrapper = inference_module.InferenceRecipe(model_path=model_path)
        logger.info("Successfully initialized inference wrapper")
        
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to load inference module: {str(e)}")
        raise

def send_chunked_response(sock: socket, data: dict, chunk_size: int = 1024*1024):
    """Send large response data in chunks"""
    try:
        # Convert data to JSON bytes
        json_bytes = json.dumps(data).encode()
        
        # Split into chunks
        chunks = [json_bytes[i:i + chunk_size] 
                 for i in range(0, len(json_bytes), chunk_size)]
        
        # Send number of chunks
        sock.sendall(struct.pack("!I", len(chunks)))
        
        # Send each chunk
        for chunk in chunks:
            # Send chunk size
            sock.sendall(struct.pack("!I", len(chunk)))
            # Send chunk data
            sock.sendall(chunk)
            
    except Exception as e:
        logger.error(f"Error sending chunked response: {e}")
        raise

def verify_sandbox():
    """Verify we're running in sandbox context"""
    sandbox_uid = pwd.getpwnam("sandbox").pw_uid
    if os.getuid() != sandbox_uid:
        return False
        
    if os.access("/etc/passwd", os.W_OK):
        return False
        
    if not os.path.exists("/sandbox"):
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Verify sandbox
    if not verify_sandbox():
        logger.error("Not running in sandbox context")
        sys.exit(1)

    # Handle signals 
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}")
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        logger.info(f"Connecting to socket at {args.socket}")
        sock = socket(AF_UNIX, SOCK_STREAM)
        sock.connect(args.socket)
        logger.info("Successfully connected to socket")

        wrapper = load_inference_module(args.model_path)
        logger.info("Inference wrapper loaded, starting prediction loop")

        while True:
            try:
                # Receive tensor data 
                input_array = receive_tensor(sock)
                
                # Receive metadata
                metadata = receive_metadata(sock)
                sample_rate = metadata.get("sample_rate", 24000)
                
                # Run inference
                try:
                    prediction = wrapper.inference(input_array, sample_rate)
                    if not isinstance(prediction, dict):
                        raise ValueError("Prediction must be a dictionary")
                    if 'audio' not in prediction or 'text' not in prediction:
                        raise ValueError("Prediction must contain 'audio' and 'text' keys")
                except Exception as e:
                    logger.error(f"Inference error: {str(e)}")
                    prediction = {"error": str(e)}

                # Send chunked response
                send_chunked_response(sock, prediction)

            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                error_resp = {"error": str(e)}
                try:
                    send_chunked_response(sock, error_resp)
                except:
                    break

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()