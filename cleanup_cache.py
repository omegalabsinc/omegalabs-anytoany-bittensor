#!/usr/bin/env python3
"""
Cleanup script for corrupted HuggingFace cache and model cache
"""

import shutil
from pathlib import Path
import os

def cleanup_cache():
    """Clean up all cache directories that might cause issues"""
    
    print("🧹 Cleaning up cache directories...")
    
    # 1. Clean up local model cache
    model_cache = Path("model_cache")
    if model_cache.exists():
        print(f"🗑️  Removing local model cache: {model_cache}")
        shutil.rmtree(model_cache)
    
    # 2. Clean up HuggingFace cache
    hf_cache_home = Path.home() / '.cache' / 'huggingface'
    if hf_cache_home.exists():
        print(f"🗑️  Cleaning HuggingFace cache: {hf_cache_home}")
        
        # Remove incomplete files
        incomplete_files = list(hf_cache_home.rglob('*.incomplete'))
        if incomplete_files:
            print(f"   Found {len(incomplete_files)} incomplete files")
            for file in incomplete_files:
                try:
                    file.unlink()
                    print(f"   ✅ Removed: {file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {file.name}: {e}")
        
        # Remove lock files
        lock_files = list(hf_cache_home.rglob('*.lock'))
        if lock_files:
            print(f"   Found {len(lock_files)} lock files")
            for file in lock_files:
                try:
                    file.unlink()
                    print(f"   ✅ Removed: {file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {file.name}: {e}")
    
    # 3. Clean up temp directories
    temp_dirs = []
    temp_dirs.extend(Path("/tmp").glob("tmp*"))
    if temp_dirs:
        print(f"🗑️  Cleaning up temp directories...")
        for temp_dir in temp_dirs:
            if temp_dir.is_dir() and "siddhantoon" in str(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"   ✅ Removed: {temp_dir}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {temp_dir}: {e}")
    
    # 4. Clear environment variables that might interfere
    env_vars_to_clear = ['HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'HF_HUB_CACHE']
    for var in env_vars_to_clear:
        if var in os.environ:
            print(f"🧹 Clearing environment variable: {var}")
            del os.environ[var]
    
    print("✅ Cache cleanup completed!")

if __name__ == "__main__":
    cleanup_cache()