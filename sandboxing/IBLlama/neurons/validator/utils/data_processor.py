import os
import time
import logging
from typing import Optional, Dict, Any
import random
import ulid

from datasets import load_dataset, Dataset, DownloadConfig
import huggingface_hub
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles multimodal dataset processing for ImageBind-LLaMA validation"""
    
    def __init__(self, config):
        """Initialize data processor with configuration"""
        self.config = config
        self.dataset: Optional[Dataset] = None
        
    def _get_timestamp_from_filename(self, filename: str) -> float:
        """Extract timestamp from filename using ULID"""
        try:
            return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp()
        except Exception:
            return 0

    def pull_latest_dataset(self) -> Optional[Dataset]:
        """Pull and process latest dataset files"""
        try:
            logger.info(f"Pulling dataset from {self.config.dataset_name}")
            
            # Get dataset file info
            ds_files = huggingface_hub.repo_info(
                repo_id=self.config.dataset_name, 
                repo_type="dataset"
            ).siblings

            # Filter recent files
            recent_files = [
                f.rfilename
                for f in ds_files if
                f.rfilename.startswith(self.config.data_prefix) and
                time.time() - self._get_timestamp_from_filename(f.rfilename) < self.config.min_age
            ][:self.config.max_files]

            if not recent_files:
                logger.warning("No recent files found")
                return None

            logger.info(f"Found {len(recent_files)} recent files")
            
            # Configure download
            download_config = DownloadConfig(download_desc="Downloading Multimodal Dataset")

            # Load dataset
            logger.info("Loading dataset files...")
            dataset = load_dataset(
                self.config.dataset_name,
                data_files=recent_files,
                download_config=download_config
            )["train"]

            # Get initial batch and shuffle
            logger.debug("Getting initial batch and shuffling")
            dataset = next(dataset.shuffle().iter(batch_size=64))
            
            # Process samples
            processed_data = {k: [] for k in dataset.keys()}
            logger.info("Processing samples...")

            # Process each sample
            for i in range(len(dataset["video_embed"])):
                # Get video embedding
                video_embed = np.array(dataset["video_embed"][i])
                
                # Skip invalid samples
                if video_embed.size == 0:
                    logger.debug(f"Skipping sample {i} - empty video embedding")
                    continue
                    
                description = dataset["description"][i]
                if not description or not isinstance(description, str):
                    logger.debug(f"Skipping sample {i} - invalid description")
                    continue

                # Add all fields for this sample
                for k in dataset.keys():
                    value = dataset[k][i]
                    processed_data[k].append(value)

                # Stop after collecting enough valid samples
                if len(processed_data["video_embed"]) >= 16:
                    logger.info("Collected 16 valid samples")
                    break

            if len(processed_data["video_embed"]) < 1:
                logger.warning("No valid samples found")
                return None

            logger.info(f"Successfully processed {len(processed_data['video_embed'])} samples")
            self.dataset = Dataset.from_dict(processed_data)
            return self.dataset

        except Exception as e:
            logger.error(f"Error pulling dataset: {e}")
            return None

    def prepare_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare a single sample for evaluation"""
        try:
            # Get video embedding
            video_embed = np.array(sample["video_embed"])
            
            # Basic validation
            if video_embed.size == 0:
                logger.warning("Empty video embedding")
                return None

            # Get description
            description = sample.get("description", "")
            if not description or not isinstance(description, str):
                logger.warning("Invalid description")
                return None

            # Get metadata
            metadata = {
                "youtube_id": sample.get("youtube_id", ""),
                "duration": sample.get("duration", 0),
                "fps": sample.get("fps", 30)
            }

            # Convert video embedding to tensor
            video_tensor = torch.from_numpy(video_embed).float()

            return {
                "input_video": video_tensor,
                "target_caption": description,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error preparing sample: {e}")
            return None