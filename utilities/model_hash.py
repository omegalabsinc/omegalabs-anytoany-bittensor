import hashlib
import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Set

# List of important auxiliary files to include in hash
AUXILIARY_FILES = {'Dockerfile', 'requirements.txt', 'config.json', 'tokenizer.json', 'vocab.json'}

# Directories and files to exclude from hashing
EXCLUDED_PATTERNS = {
    '__pycache__',
    '.git',
    '.gitignore',
    '.github',
    '.gitattributes',
    '.pytest_cache',
    '.coverage',
    '.DS_Store',
    '.env',
    'node_modules',
    '.idea',
    '.vscode',
    '.mypy_cache',
    'wandb',
    'logs',
    'tmp',
    'data',
    'webui',
    'figures',
    'samples',
    'hotkey.txt',
    '.cache',
    '.locks',
    'refs',
    'blobs',
    'version.txt',
    'audio_qa_out_cache.wav',
    'vision_qa_out_cache.wav'
}

def set_deterministic_mode():
    """
    Set all seeds and flags for deterministic behavior across machines
    """
    # Set fixed seeds for all random number generators
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Enable deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def normalize_path(path: str) -> str:
    """
    Normalize path to ensure consistent behavior across different OS
    """
    # Convert path separators to forward slashes
    normalized = path.replace(os.sep, '/')
    # Remove any trailing slashes
    normalized = normalized.rstrip('/')
    return normalized

def should_exclude(path: str) -> bool:
    """
    Check if a path should be excluded from hashing
    """
    normalized_path = normalize_path(path)
    parts = normalized_path.split('/')
    
    # Check if any part of the path matches excluded patterns
    if any(excluded in parts for excluded in EXCLUDED_PATTERNS):
        return True
        
    # Check if the full path ends with any excluded pattern
    return any(normalized_path.endswith(excluded) for excluded in EXCLUDED_PATTERNS)

def get_model_content_hash(model_path: str) -> str:
    """
    Generate a hash based on file sizes for files larger than 250MB.
    """
    # Set deterministic mode for all operations
    set_deterministic_mode()
    
    SIZE_THRESHOLD = 250 * 1024 * 1024  # 250MB in bytes
    combined_hash = hashlib.sha256()
    
    # Store large file information
    structure_info = []
    
    for root, dirs, files in os.walk(model_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        dirs.sort()  # Ensure consistent directory traversal order
        
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if should_exclude(file_path):
                continue
            
            file_size = os.path.getsize(file_path)
            if file_size < SIZE_THRESHOLD:
                continue
                
            # Get path relative to model_path
            rel_path = os.path.relpath(file_path, model_path)
            # Normalize to forward slashes
            normalized_path = rel_path.replace('\\', '/').strip('/')
            
            structure_info.append({
                'path': normalized_path,
                'size': file_size
            })
    
    # Convert structure info to JSON string and hash it
    structure_str = json.dumps(structure_info, sort_keys=True)
    combined_hash.update(structure_str.encode('utf-8'))
    
    return combined_hash.hexdigest()

def get_model_metadata_hash(metadata: Dict[str, Any]) -> str:
    """
    Generate a deterministic hash of model metadata that will be the same across all validators.
    """
    # Set deterministic mode
    set_deterministic_mode()
    
    # Ensure consistent string encoding
    def encode_value(v: Any) -> Any:
        if isinstance(v, str):
            return v.encode('utf-8')
        elif isinstance(v, (list, tuple)):
            return [encode_value(x) for x in v]
        elif isinstance(v, dict):
            return {k: encode_value(v) for k, v in sorted(v.items())}
        return v
    
    # Sort and encode the metadata dictionary
    encoded_metadata = encode_value(metadata)
    metadata_str = json.dumps(encoded_metadata, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(metadata_str.encode('utf-8')).hexdigest()

def get_combined_model_hash(model_path: str, metadata: Dict[str, Any]) -> str:
    """
    Generate a combined hash that represents both the model content and metadata.
    Ensures deterministic behavior across different machines.
    """
    # Set deterministic mode
    set_deterministic_mode()
    
    content_hash = get_model_content_hash(model_path)
    metadata_hash = get_model_metadata_hash(metadata)
    
    # Combine both hashes
    combined = hashlib.sha256()
    combined.update(content_hash.encode('utf-8'))
    combined.update(metadata_hash.encode('utf-8'))
    return combined.hexdigest() 

def get_tree_hash(model_path: str) -> str:
    """
    Generate a hash based on directory structure and file names for files smaller than 250MB.
    """
    # Set deterministic mode for all operations
    set_deterministic_mode()
    
    SIZE_THRESHOLD = 250 * 1024 * 1024  # 250MB in bytes
    tree_hash = hashlib.sha256()
    
    # Store directory structure information
    tree_info = []
    
    for root, dirs, files in os.walk(model_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        dirs.sort()  # Ensure consistent directory traversal order
        
        # Add file paths only if they're smaller than threshold
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if should_exclude(file_path):
                continue
            
            file_size = os.path.getsize(file_path)
            if file_size >= SIZE_THRESHOLD:
                continue
                
            # Get path relative to model_path
            rel_path = os.path.relpath(file_path, model_path)
            # Normalize to forward slashes
            normalized_path = rel_path.replace('\\', '/').strip('/')
            
            tree_info.append(normalized_path)
    
    # Convert tree info to JSON string and hash it
    tree_str = json.dumps(sorted(tree_info), sort_keys=True)
    tree_hash.update(tree_str.encode('utf-8'))
    
    return tree_hash.hexdigest()

def get_final_hash(model_path: str) -> str:
    """
    Generate a final hash that combines both tree structure and content hashes.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        A combined hash string
    """
    # Get both hashes
    tree_hash = get_tree_hash(model_path)
    content_hash = get_model_content_hash(model_path)
    
    # Combine both hashes
    final_hash = hashlib.sha256()
    final_hash.update(tree_hash.encode('utf-8'))
    final_hash.update(content_hash.encode('utf-8'))
    
    return final_hash.hexdigest()

def test_model_hash(model_path: str):
    """
    Test hash generation for a given model path.
    
    Args:
        model_path: Path to the model directory to test
    """
    try:
        tree_hash = get_tree_hash(model_path)
        content_hash = get_model_content_hash(model_path)
        final_hash = get_final_hash(model_path)
        print(f"Tree hash: {tree_hash}")
        print(f"Content hash: {content_hash}")
        print(f"Final hash: {final_hash}")
        
    except Exception as e:
        print(f"\nError testing model hash: {str(e)}")

if __name__ == "__main__":
    test_model_hash("/home/salman/tezuesh/omega-dev/model_cache/v1_model")
    test_model_hash("/home/salman/tezuesh/omega-dev/model_cache/v2_model")
