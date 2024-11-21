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
import numpy as np
import random
import subprocess
import torch
import bittensor as bt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.S2S import inference as s2s_inference
from evaluation.S2S.distance import S2SMetrics

HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 48 * 60 * 60  # 48 hours
MAX_FILES = 10


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp


def pull_latest_diarization_dataset() -> Optional[Dataset]:    
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX)
    ][:MAX_FILES]

    if len(recent_files) == 0:
        return None
    temp_dir = "./data_cache"
    # with TemporaryDirectory(dir='./data_cache') as temp_dir:
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir)["train"]
    omega_dataset.cast_column("audio", Audio(sampling_rate=16000))
    omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
    return omega_dataset



def get_gpu_memory():
    # system-level
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'])
    total_gb, used_gb = map(lambda s: int(s) / 1e3, output.decode('utf-8').split(','))
    return total_gb, used_gb, total_gb - used_gb

def log_gpu_memory(msg=''):
    # process-level
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    bt.logging.info(f"GPU-MEM {msg} Total: {t/1e9:.2f}GB, Reserved: {r/1e9:.2f}GB, Allocated: {a/1e9:.2f}GB")

def cleanup_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_ckpt_from_hf(model_id: str, hf_repo_id: str, device: str='cuda'):
    return s2s_inference(model_id, hf_repo_id, device)


def compute_s2s_metrics(model_id: str, hf_repo_id: str, dataset: Dataset):
    cleanup_gpu_memory()
    log_gpu_memory('before model load')
    model = load_ckpt_from_hf(model_id, hf_repo_id, device='cuda')
    log_gpu_memory('after model load')
    s2s_metrics = S2SMetrics()
    metrics = {'mimi_score': [],
                'wer_score': [],
                'length_penalty': [],
                'pesq_score': [],
                'anti_spoofing_score': [],
                'combined_score': [],
                'total_samples': 0}
    
   
    for i in tqdm(range(64)):
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
        result = model.inference(audio_array=diar_sample, sample_rate=sample_rate)
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

        metrics_dict = s2s_metrics.compute_distance(gt_audio_arrs=[[diar_gt, sample_rate]], generated_audio_arrs=[[pred_audio.squeeze(0).squeeze(0), sample_rate]])
        
        for key, value in metrics_dict.items():
            metrics[key].append(value)

    
    mean_score = np.mean(metrics['combined_score'])
    bt.logging.info(f"Scoring {model_id} {hf_repo_id} complete: {mean_score:0.5f}")
    cleanup_gpu_memory()
    log_gpu_memory('after model clean-up')

    return mean_score


if __name__ == "__main__":
    dataset = pull_latest_diarization_dataset()
    print(compute_s2s_metrics(model_id="moshi", hf_repo_id="kyutai/moshiko-pytorch-bf16", dataset=dataset))
