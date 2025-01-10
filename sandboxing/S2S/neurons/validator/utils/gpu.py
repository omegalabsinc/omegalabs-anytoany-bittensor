

import torch
import logging
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)

def get_gpu_memory() -> Tuple[float, float, float]:
    """Get GPU memory stats in GB"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader']
        )
        total_gb, used_gb = map(lambda s: int(s) / 1024, output.decode('utf-8').split(','))
        return total_gb, used_gb, total_gb - used_gb
    except:
        return 0, 0, 0

def log_gpu_memory(msg: str = ''):
    """Log GPU memory usage"""
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        logger.info(f"GPU-MEM {msg} Total: {t/1e9:.2f}GB, Reserved: {r/1e9:.2f}GB, Allocated: {a/1e9:.2f}GB")
    except:
        logger.warning("Unable to get GPU memory stats")

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()