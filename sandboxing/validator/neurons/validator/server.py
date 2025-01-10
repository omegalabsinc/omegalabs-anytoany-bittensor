import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np

from .config import ValidatorConfig, ModelConfig
from .sandbox.manager import SandboxManager
from .utils.data_processor import DataProcessor
from .metrics.s2s import S2SMetrics
from .utils.gpu import cleanup_gpu_memory, log_gpu_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceServer:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        logger.debug(f"Initializing server with config: {config}")
        
        # Initialize sandbox manager
        self.sandbox = SandboxManager(
            model_config=config.model,
            sandbox_config=config.sandbox
        )
        
        # Initialize data processor
        self.data_processor = DataProcessor(config.data)
        
        # Initialize metrics
        self.metrics = S2SMetrics(config.metrics)
        
        self._is_running = False

    def start(self):
        """Start the inference server"""
        try:
            logger.info("Starting inference server...")
            self.sandbox.setup()
            self.sandbox.start()
            self._is_running = True
            logger.info("Inference server started successfully")
            log_gpu_memory("after server start")
        except Exception as e:
            logger.error(f"Inference server start failed: {e}")
            self.cleanup()
            raise

    def stop(self):
        """Stop the server"""
        logger.info("Stopping server...")
        self.cleanup()
        logger.info("Server stopped")

    def cleanup(self):
        """Cleanup resources"""
        self._is_running = False
        if hasattr(self, 'sandbox'):
            self.sandbox.cleanup()
        cleanup_gpu_memory()

    def predict(self, audio: torch.Tensor, sample_rate: int) -> Optional[Dict]:
        """Run prediction in sandbox"""
        if not self._is_running:
            raise RuntimeError("Server not running")
            
        try:
            # Ensure audio is a tensor and properly formatted
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio, dtype=torch.float32)
            
            # Ensure 1D tensor
            if audio.dim() > 1:
                audio = audio.flatten()
            
            # Validate tensor
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError("Input tensor contains NaN or Inf values")
            
            # Run prediction through sandbox
            prediction = self.sandbox.predict(audio, sample_rate)
            
            if not prediction:
                raise RuntimeError("No prediction returned from sandbox")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def evaluate_sample(self, sample: Dict) -> Optional[Dict[str, float]]:
        """Evaluate a single sample"""
        try:
            # Get input and reference audio
            input_audio = sample['input_audio']
            reference_audio = sample['target_audio']
            sample_rate = sample['sample_rate']

            
            # Run prediction
            result = self.predict(
                torch.tensor(input_audio), 
                sample_rate
            )

            
            if not result or 'audio' not in result:
                return None
                
            # Compute metrics
            scores = self.metrics.compute_distance(
                gt_audio_arrs=[[reference_audio, sample_rate]],
                generated_audio_arrs=[[np.array(result['audio']).squeeze(0).squeeze(0), sample_rate]],
            )
            
            return scores
            
        except Exception as e:
            logger.error(f"Sample evaluation failed: {e}")
            return None

    def score_model(self, num_samples: int = 1) -> Optional[Dict[str, float]]:
        """Score model on validation data"""
        if not self._is_running:
            raise RuntimeError("Server not running")
            
        try:
            # Get dataset
            dataset = self.data_processor.pull_latest_dataset()
            if not dataset:
                logger.error("Failed to get dataset")
                return None
                
            # Collect scores
            all_scores = []
            logger.info(f"Evaluating {min(len(dataset), num_samples)} samples")
            for i in range(2):
                sample = self.data_processor.prepare_sample(dataset[i])
                if sample:  # Only evaluate if sample preparation succeeded
                    scores = self.evaluate_sample(sample)
                    if scores:
                        all_scores.append(scores)
                    
            if not all_scores:
                logger.error("No valid scores collected")
                return None
                
            # Aggregate scores
            mean_scores = {}
            for metric in all_scores[0].keys():
                values = [s[metric] for s in all_scores if metric in s]
                if values:
                    mean_scores[metric] = float(np.mean(values))
                    
            logger.info(f"Model evaluation complete. Scores: {mean_scores}")
            return mean_scores
            
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            return None

    def save_config(self, path: str):
        """Save current configuration"""
        self.config.save_config(path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--revision", default="main", help="Model revision")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(args.log_level)
    
    # Create config
    config = ValidatorConfig(
        model=ModelConfig(
            model_id=args.model_id,
            revision=args.revision
        )
    )

    server = InferenceServer(config)
    
    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        server.start()
        
        # Run initial model scoring
        logger.info("Running initial model scoring...")
        scores = server.score_model()
        if scores:
            logger.info(f"Initial model scores: {scores}")
            
            # Save config with scores
            if args.config:
                server.save_config(args.config)
        
        logger.info("Server running. Press Ctrl+C to stop.")
        signal.pause()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        server.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        server.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()