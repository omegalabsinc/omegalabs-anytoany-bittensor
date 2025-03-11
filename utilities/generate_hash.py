import hashlib
import os
import json
from pathlib import Path

# Common model file extensions
MODEL_EXTENSIONS = {
    '.safetensors',  # HuggingFace Safetensors format
    '.pt',           # PyTorch format
    '.pth',          # PyTorch alternative format
    '.bin',          # Binary model files
    '.ckpt',         # Checkpoint files
    '.model',        # Generic model files
    '.weights',      # Weight files
    '.onnx',         # ONNX format
    '.h5',           # Keras/TensorFlow format
    '.pb',           # TensorFlow ProtoBuf format
}

def compute_file_hash(filepath, chunk_size=8192):
    """Compute SHA-256 hash of a file in chunks to handle large files efficiently."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def compute_combined_hash(file_hashes):
    """Compute a single hash from multiple file hashes."""
    combined = hashlib.sha256()
    # Sort to ensure consistent order
    for file_hash in sorted(file_hashes):
        combined.update(file_hash.encode())
    return combined.hexdigest()

def is_model_file(filepath):
    """Check if a file is a model file based on its extension."""
    return filepath.suffix.lower() in MODEL_EXTENSIONS

def find_model_files(directory):
    """Recursively find all model files in a directory."""
    model_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = Path(root) / file
            if is_model_file(filepath):
                model_files.append(filepath)
    return sorted(model_files)

def get_model_hashes(repo_path):
    """Get hashes for all model files in a repository."""
    model_files = find_model_files(repo_path)
    
    if not model_files:
        return None
    
    file_hashes = []
    file_details = []
    
    for file_path in model_files:
        # Get relative path from repo root for cleaner output
        rel_path = file_path.relative_to(repo_path)
        file_hash = compute_file_hash(file_path)
        file_hashes.append(file_hash)
        file_details.append({
            "filename": str(rel_path),
            "size": f"{os.path.getsize(file_path) / (1024*1024*1024):.2f} GB",
            "hash": file_hash,
            "type": file_path.suffix.lower()
        })

    return compute_combined_hash(file_hashes)

