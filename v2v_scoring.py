import os
from datasets import load_dataset, Audio
import huggingface_hub
from tempfile import TemporaryDirectory
import time
from typing import Optional
from datasets import Dataset
import ulid
import pandas as pd
import tempfile
from tqdm import tqdm
from models.S2S import inference as s2s_inference
import numpy as np
from evaluation.S2S.distance import S2SMetrics
import random

HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 48 * 60 * 60  # 48 hours
MAX_FILES = 2000


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp


def pull_latest_diarization_dataset() -> Optional[Dataset]:    
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX)
    ][:MAX_FILES]

    print(recent_files)
    if len(recent_files) == 0:
        return None
    temp_dir = "./data_cache"
    # with TemporaryDirectory(dir='./data_cache') as temp_dir:
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir)["train"]
    omega_dataset.cast_column("audio", Audio(sampling_rate=16000))
    omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
    return omega_dataset


def inference(model_id: str, audio_array: np.array, sample_rate: int):
    return s2s_inference(model_id, audio_array, sample_rate)


def compute_s2s_metrics(dataset: Dataset):
    s2s_metrics = S2SMetrics()
    metrics = {'mimi_score': [],
                'wer_score': [],
                'length_penalty': [],
                'pesq_score': [],
                'anti_spoofing_score': [],
                'combined_score': [],
                'total_samples': 0}
    from datetime import datetime
   
    for i in tqdm(range(10)):
        youtube_id = dataset['youtube_id'][i]
        audio_array = np.array(dataset['audio'][i]['array'])
        sample_rate = dataset['audio'][i]['sampling_rate']

        diar_timestamps_start = np.array(dataset['diar_timestamps_start'][i])
        diar_timestamps_end = np.array(dataset['diar_timestamps_end'][i])
        diar_speakers = np.array(dataset['diar_speakers'][i])
        if len(diar_timestamps_start) == 1:
            continue
        

        test_idx = random.randint(0, len(diar_timestamps_start) - 2)
        diar_sample = audio_array[int(diar_timestamps_start[test_idx] * sample_rate):int(diar_timestamps_end[test_idx] * sample_rate)]
        diar_gt = audio_array[int(diar_timestamps_start[test_idx+1] * sample_rate):int(diar_timestamps_end[test_idx+1] * sample_rate)]
        speaker = diar_speakers[test_idx]

        # Add minimum length check (250ms = 0.25 seconds)
        min_samples = int(0.25 * sample_rate)
        if len(diar_sample) < min_samples or len(diar_gt) < min_samples:
            continue

        # Perform inference
        result = inference(model_id="moshi", audio_array=diar_sample, sample_rate=sample_rate)
        pred_audio = result['audio']
        
        # Check if inference produced valid audio
        if pred_audio is None or len(pred_audio) == 0 or pred_audio.shape[-1] == 0:
            for k, v in metrics.items():
                if k == 'total_samples':
                    metrics[k] += 1
                else:
                    metrics[k].append(0)
            continue

            
        metrics['total_samples'] += 1

        metrics_dict = s2s_metrics.compute_distance(gt_audio_arrs=[[diar_gt, sample_rate]], generated_audio_arrs=[[pred_audio, sample_rate]])
        for key, value in metrics_dict.items():
            metrics[key].append(value)

    for key, value in metrics.items():
        if key != 'total_samples':
            metrics[key] = np.sum(np.array(value)) / metrics['total_samples']

    return metrics



if __name__ == "__main__":
    dataset = pull_latest_diarization_dataset()
    print(compute_s2s_metrics(dataset))
