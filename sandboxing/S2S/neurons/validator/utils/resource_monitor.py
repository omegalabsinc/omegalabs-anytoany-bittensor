import threading
import time
import psutil
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ResourceThresholds:
    """Resource threshold configuration"""
    max_gpu_memory_gb: float = 40.0  # Maximum GPU memory in GB
    max_ram_percent: float = 90.0    # Maximum RAM usage percentage
    max_cpu_percent: float = 95.0    # Maximum CPU usage percentage
    check_interval: float = 0.1      # How often to check resources (seconds)

class ResourceMonitor:
    """Monitors system resources in a separate thread"""
    
    def __init__(self, thresholds: ResourceThresholds):
        self.thresholds = thresholds
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._violations: List[str] = []
        self.peak_stats: Dict[str, float] = {
            'gpu_memory_gb': 0.0,
            'ram_percent': 0.0,
            'cpu_percent': 0.0
        }

    def _get_gpu_memory_gb(self) -> Tuple[float, float]:
        """Get GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                gpu_idx = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(gpu_idx) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_idx) / 1024**3
                return allocated, reserved
        except Exception as e:
            logger.warning(f"Error getting GPU memory: {e}")
        return 0.0, 0.0

    def print_resources(self):
        """Print current resource usage"""
        try:
            # Get current resource usage
            allocated_gpu, reserved_gpu = self._get_gpu_memory_gb()
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Format output
            logger.info("\n=== Resource Usage ===")
            logger.info(f"GPU Memory:")
            logger.info(f"  - Allocated: {allocated_gpu:.2f} GB")
            logger.info(f"  - Reserved:  {reserved_gpu:.2f} GB")
            logger.info(f"  - Threshold: {self.thresholds.max_gpu_memory_gb:.2f} GB")
            logger.info(f"\nRAM Usage:")
            logger.info(f"  - Used:      {ram.percent:.1f}%")
            logger.info(f"  - Available: {ram.available/1024**3:.1f} GB")
            logger.info(f"  - Total:     {ram.total/1024**3:.1f} GB") 
            logger.info(f"  - Threshold: {self.thresholds.max_ram_percent:.1f}%")
            logger.info(f"\nCPU Usage:")
            logger.info(f"  - Current:   {cpu:.1f}%")
            logger.info(f"  - Threshold: {self.thresholds.max_cpu_percent:.1f}%")
            logger.info("===================\n")
            
        except Exception as e:
            logger.error(f"Error printing resources: {e}")

    def _monitor_resources(self):
        """Resource monitoring loop"""
        while self._monitoring:
            try:
                # Get current resource usage
                allocated_gpu, reserved_gpu = self._get_gpu_memory_gb() 
                ram_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent()

                # Update peak stats
                with self._lock:
                    self.peak_stats['gpu_memory_gb'] = max(
                        self.peak_stats['gpu_memory_gb'], 
                        reserved_gpu
                    )
                    self.peak_stats['ram_percent'] = max(
                        self.peak_stats['ram_percent'], 
                        ram_percent
                    )
                    self.peak_stats['cpu_percent'] = max(
                        self.peak_stats['cpu_percent'], 
                        cpu_percent
                    )

                    # Check for violations
                    if reserved_gpu > self.thresholds.max_gpu_memory_gb:
                        self._violations.append(
                            f"GPU memory ({reserved_gpu:.1f}GB) exceeded threshold "
                            f"({self.thresholds.max_gpu_memory_gb}GB)"
                        )
                    if ram_percent > self.thresholds.max_ram_percent:
                        self._violations.append(
                            f"RAM usage ({ram_percent:.1f}%) exceeded threshold "
                            f"({self.thresholds.max_ram_percent}%)"
                        )
                    if cpu_percent > self.thresholds.max_cpu_percent:
                        self._violations.append(
                            f"CPU usage ({cpu_percent:.1f}%) exceeded threshold "
                            f"({self.thresholds.max_cpu_percent}%)"
                        )

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")

            time.sleep(self.thresholds.check_interval)

    def start(self):
        """Start resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._violations = []
            self.peak_stats = {k: 0.0 for k in self.peak_stats}
            self._thread = threading.Thread(target=self._monitor_resources)
            self._thread.daemon = True
            self._thread.start()
            # self.print_resources()
            
    def stop(self) -> Tuple[bool, Dict[str, float], List[str]]:
        """Stop monitoring and return results"""
        if self._monitoring:
            self._monitoring = False
            if self._thread:
                self._thread.join()
                self._thread = None

            with self._lock:
                violations = self._violations.copy()
                stats = self.peak_stats.copy()

            return len(violations) == 0, stats, violations
        return True, self.peak_stats, []

    def get_compute_score(self) -> float:
        """Calculate compute score based on resource usage"""
        _, stats, violations = self.stop()
        
        if violations:
            return 0.0
            
        # Calculate scores for each resource (1.0 = no usage, 0.0 = at threshold)
        # Binary scores - 1 if under threshold, 0 if over
        gpu_score = 1.0 if stats['gpu_memory_gb'] < self.thresholds.max_gpu_memory_gb else 0.0
        ram_score = 1.0 if stats['ram_percent'] < self.thresholds.max_ram_percent else 0.0
        cpu_score = 1.0 if stats['cpu_percent'] < self.thresholds.max_cpu_percent else 0.0
        
        # Return 1.0 only if all resources are under their thresholds
        return 1.0 if (gpu_score == 1.0 and ram_score == 1.0 and cpu_score == 1.0) else 0.0