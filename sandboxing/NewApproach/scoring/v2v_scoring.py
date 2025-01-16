import os
import time
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import random
import logging
from datasets import Dataset, load_dataset, Audio, DownloadConfig
import huggingface_hub
from tempfile import TemporaryDirectory

from evaluation.S2S.distance import S2SMetrics
from scoring.docker_manager import DockerManager

# Constants
HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 8 * 60 * 60  # 8 hours
MAX_FILES = 8

def get_dataset() -> Optional[Dataset]:
    """Get latest dataset from HuggingFace."""
    try:
        # Get recent files
        ds_files = huggingface_hub.repo_info(
            repo_id=HF_DATASET, 
            repo_type="dataset"
        ).siblings
        
        recent_files = [
            f.rfilename
            for f in ds_files if
            f.rfilename.startswith(DATA_FILES_PREFIX) and
            time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
        ][:MAX_FILES]

        if not recent_files:
            logging.warning("No recent files found")
            return None

        with TemporaryDirectory(dir='./data_cache') as temp_dir:
            # Load dataset
            download_config = DownloadConfig()
            dataset = load_dataset(
                HF_DATASET,
                data_files=recent_files,
                cache_dir=temp_dir,
                download_config=download_config
            )["train"]
            
            # Process audio
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            dataset = next(dataset.shuffle().iter(batch_size=64))
            
            # Filter samples
            filtered = self._filter_samples(dataset)
            
            return filtered if filtered['audio'] else None

    except Exception as e:
        logging.error(f"Failed to get dataset: {e}")
        return None

def _filter_samples(dataset: Dataset) -> Dict:
    """Filter and process dataset samples."""
    filtered = {k: [] for k in dataset.keys()}
    
    for i in range(len(dataset['audio'])):
        audio = dataset['audio'][i]
        timestamps = np.array(dataset['diar_timestamps_start'][i])
        speakers = np.array(dataset['diar_speakers'][i])

        # Skip single speaker samples
        if len(set(speakers)) == 1:
            continue

        # Add all fields
        for k in dataset.keys():
            value = audio if k == 'audio' else dataset[k][i]
            filtered[k].append(value)

        # Limit samples
        if len(filtered['audio']) >= 16:
            break

    return Dataset.from_dict(filtered)

def compute_scores(
    model_id: str,
    repo_id: str,
    dataset: Dataset,
    cache_dir: str,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Compute scores for a model."""
    try:
        docker_manager = DockerManager(base_cache_dir=cache_dir)
        metrics = S2SMetrics(cache_dir=cache_dir, device=device)

        # Start container
        container_url = docker_manager.start_container(
            uid=model_id,
            repo_id=repo_id
        )
        
        if not container_url:
            logging.error(f"Failed to start container for {model_id}")
            return get_zero_scores()

        try:
            return process_samples(
                container_url=container_url,
                dataset=dataset,
                metrics=metrics,
                docker_manager=docker_manager
            )
        finally:
            docker_manager.stop_container(model_id)

    except Exception as e:
        logging.error(f"Scoring failed for {model_id}: {e}")
        return get_zero_scores()

def process_samples(
    container_url: str,
    dataset: Dataset,
    metrics: S2SMetrics,
    docker_manager: DockerManager
) -> Dict[str, float]:
    """Process dataset samples through model."""
    scores = {
        'mimi_score': [],
        'wer_score': [],
        'length_penalty': [],
        'pesq_score': [],
        'anti_spoofing_score': [],
        'combined_score': [],
        'total_samples': 0
    }

    for i in range(len(dataset['youtube_id'])):
        # Get sample pair
        sample_pair = extract_sample_pair(
            audio=dataset['audio'][i]['array'],
            timestamps_start=dataset['diar_timestamps_start'][i],
            timestamps_end=dataset['diar_timestamps_end'][i],
            sample_rate=dataset['audio'][i]['sampling_rate']
        )
        
        if not sample_pair:
            continue

        test_audio, gt_audio = sample_pair
        sample_rate = dataset['audio'][i]['sampling_rate']

        try:
            # Run inference
            result = docker_manager.inference(
                container_url,
                test_audio, 
                sample_rate
            )

            # Update metrics
            update_metrics(
                scores=scores,
                prediction=result['audio'],
                ground_truth=gt_audio,
                sample_rate=sample_rate,
                metrics=metrics
            )

        except Exception as e:
            logging.error(f"Sample {i} failed: {e}")
            update_zero_metrics(scores)

    return {
        k: float(np.mean(v)) if v else 0.0 
        for k, v in scores.items()
        if k != 'total_samples'
    }

def get_zero_scores() -> Dict[str, float]:
    """Get zero scores for all metrics."""
    return {
        'mimi_score': 0.0,
        'wer_score': 0.0,
        'length_penalty': 0.0,
        'pesq_score': 0.0,
        'anti_spoofing_score': 0.0,
        'combined_score': 0.0
    }