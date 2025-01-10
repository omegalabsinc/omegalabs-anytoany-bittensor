import os
import time
import numpy as np
from typing import Optional, Dict, Any
from datasets import load_dataset, Audio, Dataset, DownloadConfig
import huggingface_hub
import logging
import ulid
import random
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        """Initialize data processor with configuration"""
        self.config = config
        
    def _get_timestamp_from_filename(self, filename: str) -> float:
        """Extract timestamp from filename using ULID"""
        try:
            return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp
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

            # Load dataset directly without temp directory
            download_config = DownloadConfig(download_desc="Downloading Voice Dataset")
            logger.info("Loading dataset files...")
            dataset = load_dataset(
                self.config.dataset_name,
                data_files=recent_files,
                download_config=download_config
            )["train"]

            # Configure audio
            logger.debug("Configuring audio sampling rate")
            dataset = dataset.cast_column(
                "audio", 
                Audio(sampling_rate=self.config.sampling_rate)
            )

            # Get initial batch
            logger.debug("Getting initial batch of 64 samples")
            dataset = next(dataset.shuffle().iter(batch_size=64))

            # Process samples
            processed_data = {k: [] for k in dataset.keys()}
            logger.info("Processing audio samples...")

            # Process each audio sample
            for i in range(len(dataset['audio'])):
                # Extract raw audio array
                audio_array = dataset['audio'][i]
                # Get speaker timestamps and IDs
                timestamps_start = np.array(dataset['diar_timestamps_start'][i])
                speakers = np.array(dataset['diar_speakers'][i])

                # Skip samples with only 1 speaker
                if len(set(speakers)) == 1:
                    logger.debug(f"Skipping sample {i} - only 1 speaker")
                    continue

                # Add all fields for this sample
                for k in dataset.keys():
                    value = audio_array if k == 'audio' else dataset[k][i]
                    processed_data[k].append(value)

                # Stop after collecting 16 valid samples
                if len(processed_data['audio']) >= 16:
                    logger.info("Collected 16 valid samples")
                    break

            if len(processed_data['audio']) < 1:
                logger.warning("No valid samples found")
                return None

            logger.info(f"Successfully processed {len(processed_data['audio'])} samples")
            return Dataset.from_dict(processed_data)

        except Exception as e:
            logger.error(f"Error pulling dataset: {e}")
            return None

    def prepare_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare a single sample for inference"""
        try:
            # Get audio data
            audio_array = np.array(sample['audio']['array'])
            sample_rate = sample['audio']['sampling_rate']

            # Get timestamps and speakers
            timestamps_start = np.array(sample['diar_timestamps_start'])
            timestamps_end = np.array(sample['diar_timestamps_end'])
            speakers = np.array(sample['diar_speakers'])

            if len(timestamps_start) < 2:
                logger.warning("Sample has insufficient segments")
                return None

            # Select random segment
            test_idx = random.randint(0, len(timestamps_start) - 2)
            
            # Get input and target audio
            input_audio = audio_array[
                int(timestamps_start[test_idx] * sample_rate):
                int(timestamps_end[test_idx] * sample_rate)
            ]
            
            target_audio = audio_array[
                int(timestamps_start[test_idx+1] * sample_rate):
                int(timestamps_end[test_idx+1] * sample_rate)
            ]

            # Check minimum length (250ms)
            min_samples = int(0.25 * sample_rate)
            if len(input_audio) < min_samples or len(target_audio) < min_samples:
                logger.warning("Sample segments too short")
                return None

            return {
                'input_audio': input_audio,
                'target_audio': target_audio,
                'sample_rate': sample_rate,
                'speaker': speakers[test_idx],
                'youtube_id': sample.get('youtube_id')
            }

        except Exception as e:
            logger.error(f"Error preparing sample: {e}")
            return None