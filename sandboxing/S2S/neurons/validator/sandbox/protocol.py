import json
import struct
import logging
import numpy as np
from socket import socket
from typing import Optional, Any
import torch

logger = logging.getLogger(__name__)

class SandboxProtocol:
    """Handles communication with sandboxed process"""
    
    def __init__(self, sock: socket):
        self._socket = sock
        self._connection: Optional[socket] = None
        
    def accept_connection(self):
        """Accept connection from sandbox process"""
        self._connection, _ = self._socket.accept()
        logger.info("Accepted new sandbox connection")
        
    def is_connected(self) -> bool:
        return bool(self._connection)

    def _send_tensor(self, tensor: torch.Tensor) -> None:
        """Send tensor data efficiently"""
        if not self._connection:
            raise RuntimeError("No connection")
            
        # Ensure tensor is contiguous, float32, and 1D
        tensor = tensor.contiguous().float()
        if tensor.dim() > 1:
            tensor = tensor.flatten()
        
        # Convert to numpy and get bytes
        tensor_bytes = tensor.cpu().numpy().tobytes()
        
        # Send tensor shape metadata
        shape_data = json.dumps({
            "shape": list(tensor.shape),
            "dtype": "float32"
        }).encode()
        
        # Send metadata size and data
        self._connection.sendall(struct.pack("!I", len(shape_data)))
        self._connection.sendall(shape_data)
        
        # Send tensor size and data
        self._connection.sendall(struct.pack("!Q", len(tensor_bytes)))
        self._connection.sendall(tensor_bytes)

    def _send_metadata(self, metadata: dict) -> None:
        """Send additional metadata"""
        if not self._connection:
            raise RuntimeError("No connection")
            
        metadata_bytes = json.dumps(metadata).encode()
        self._connection.sendall(struct.pack("!I", len(metadata_bytes)))
        self._connection.sendall(metadata_bytes)

    def _receive_data(self) -> dict:
        """Receive data from sandbox process with chunked transfer"""
        if not self._connection:
            raise RuntimeError("No connection")
            
        try:
            # Receive number of chunks
            num_chunks = struct.unpack("!I", self._connection.recv(4))[0]
            
            # Receive all chunks
            data_chunks = []
            for _ in range(num_chunks):
                # Receive chunk size
                chunk_size = struct.unpack("!I", self._connection.recv(4))[0]
                
                # Receive chunk data
                chunk_data = b""
                while len(chunk_data) < chunk_size:
                    remaining = chunk_size - len(chunk_data)
                    chunk = self._connection.recv(min(8192, remaining))
                    if not chunk:
                        raise RuntimeError("Connection closed while receiving chunk")
                    chunk_data += chunk
                    
                data_chunks.append(chunk_data)
                
            # Combine chunks and parse JSON
            complete_data = b"".join(data_chunks)
            return json.loads(complete_data)
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to receive data: {e}")
            
    def run_inference(self, tensor: torch.Tensor, sample_rate: int = 24000) -> dict:
        """Send tensor and receive prediction"""
        try:
            # Ensure tensor is contiguous and in the correct shape
            tensor = tensor.contiguous()
            
            # Send tensor data
            self._send_tensor(tensor)
            
            # Send metadata
            self._send_metadata({
                "sample_rate": sample_rate
            })
            
            # Receive response
            response = self._receive_data()
            
            if "error" in response:
                logger.error(f"Sandbox error: {response['error']}")
                raise RuntimeError(response["error"])
                
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"error": str(e)}

    def close(self):
        """Close connection"""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None