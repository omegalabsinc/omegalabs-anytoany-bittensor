from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import torch
import yaml

@dataclass
class MetricParameters:
    """Parameters for individual metrics"""
    enabled: bool = True
    weight: float = 1.0
    threshold: Optional[float] = None
    parameters: Dict = field(default_factory=dict)

@dataclass
class MetricsConfig:
    """Flexible metrics configuration"""
    # Dictionary storing metric configurations
    metrics: Dict[str, MetricParameters] = field(default_factory=lambda: {
        "wer": MetricParameters(
            enabled=True,
            weight=0.3,
            parameters={"lowercase": True, "normalize": True}
        ),
        "pesq": MetricParameters(
            enabled=True,
            weight=0.3,
            parameters={"mode": "wb"}
        ),
        "length_penalty": MetricParameters(
            enabled=True,
            weight=0.2,
            parameters={"min_ratio": 0.5, "max_ratio": 2.0}
        ),
        "anti_spoofing": MetricParameters(
            enabled=True,
            weight=0.2,
            parameters={"noise_level": 0.1}
        )
    })
    
    # Cache directory for metrics computation
    metrics_cache_dir: str = "./metrics_cache"
    
    # Minimum required score for validation
    min_total_score: float = 0.5
    
    @property
    def enabled_metrics(self) -> List[str]:
        """Get list of enabled metrics"""
        return [name for name, config in self.metrics.items() 
                if config.enabled]
    
    def get_weights(self) -> Dict[str, float]:
        """Get dictionary of metric weights"""
        return {name: config.weight 
                for name, config in self.metrics.items() 
                if config.enabled}
    
    def enable_metric(self, metric_name: str, weight: Optional[float] = None):
        """Enable a specific metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].enabled = True
            if weight is not None:
                self.metrics[metric_name].weight = weight
    
    def disable_metric(self, metric_name: str):
        """Disable a specific metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].enabled = False
    
    def update_metric_params(self, metric_name: str, **params):
        """Update parameters for a specific metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].parameters.update(params)
    
    def save_to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "metrics": {
                name: {
                    "enabled": metric.enabled,
                    "weight": metric.weight,
                    "threshold": metric.threshold,
                    "parameters": metric.parameters
                }
                for name, metric in self.metrics.items()
            },
            "metrics_cache_dir": self.metrics_cache_dir,
            "min_total_score": self.min_total_score
        }
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'MetricsConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        metrics = {}
        for name, params in config_dict["metrics"].items():
            metrics[name] = MetricParameters(
                enabled=params["enabled"],
                weight=params["weight"],
                threshold=params.get("threshold"),
                parameters=params.get("parameters", {})
            )
        
        return cls(
            metrics=metrics,
            metrics_cache_dir=config_dict["metrics_cache_dir"],
            min_total_score=config_dict["min_total_score"]
        )

@dataclass
class SandboxConfig:
    sandbox_user: str = "sandbox"
    sandbox_dir: Path = Path("/sandbox")
    socket_path: Path = Path("/sandbox/inference.sock")
    max_memory_mb: int = 2048
    max_file_size_mb: int = 1000
    socket_timeout: int = 30

@dataclass
class ModelConfig:
    model_id: str
    revision: str = "main"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    dataset_name: str = "omegalabsinc/omega-voice"
    data_prefix: str = "default/train/"
    min_age: int = 8 * 60 * 60  # 8 hours
    max_files: int = 8
    sampling_rate: int = 16000
    min_segment_duration: float = 0.25  # seconds

@dataclass
class ValidatorConfig:
    sandbox: SandboxConfig = SandboxConfig()
    model: ModelConfig = ModelConfig(model_id="tezuesh/moshi_general")
    data: DataConfig = DataConfig()
    metrics: MetricsConfig = MetricsConfig()
    log_level: str = "INFO"
    
    def save_config(self, file_path: str):
        """Save full configuration to YAML"""
        config_dict = {
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "device": self.model.device
            },
            "data": {
                "dataset_name": self.data.dataset_name,
                "data_prefix": self.data.data_prefix,
                "min_age": self.data.min_age,
                "max_files": self.data.max_files,
                "sampling_rate": self.data.sampling_rate,
                "min_segment_duration": self.data.min_segment_duration
            },
            "metrics": {
                name: {
                    "enabled": metric.enabled,
                    "weight": metric.weight,
                    "threshold": metric.threshold,
                    "parameters": metric.parameters
                }
                for name, metric in self.metrics.metrics.items()
            }
        }
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
    
    @classmethod
    def load_config(cls, file_path: str) -> 'ValidatorConfig':
        """Load configuration from YAML"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict["model"]),
            data=DataConfig(**config_dict["data"]),
            metrics=MetricsConfig.load_from_yaml(file_path)
        )