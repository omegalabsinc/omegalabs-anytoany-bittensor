import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset, Audio, DownloadConfig
import huggingface_hub
from tempfile import TemporaryDirectory
import time
from typing import Optional
from datasets import Dataset
import ulid
import numpy as np
import random
import subprocess
import torch
import bittensor as bt
from pathlib import Path
from neurons.docker_manager import DockerManager
from evaluation.S2S.distance import S2SMetrics


HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 8 * 60 * 60  # 8 hours
MAX_FILES = 8

def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

def pull_latest_diarization_dataset() -> Optional[Dataset]:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]

    download_config = DownloadConfig(download_desc="Downloading Omega Voice Dataset")

    if len(recent_files) == 0:
        return None

    with TemporaryDirectory(dir='./data_cache') as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir, download_config=download_config)["train"]
        omega_dataset.cast_column("audio", Audio(sampling_rate=16000))
        omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))

        overall_dataset = {k: [] for k in omega_dataset.keys()}

        for i in range(len(omega_dataset['audio'])):
            audio_array = omega_dataset['audio'][i]
            diar_timestamps_start = np.array(omega_dataset['diar_timestamps_start'][i])
            diar_speakers = np.array(omega_dataset['diar_speakers'][i])

            if len(set(diar_speakers)) == 1:
                continue

            for k in omega_dataset.keys():
                value = audio_array if k == 'audio' else omega_dataset[k][i]
                overall_dataset[k].append(value)

            if len(overall_dataset['audio']) >= 8:
                break

        if len(overall_dataset['audio']) < 1:
            return None

        return Dataset.from_dict(overall_dataset)

def get_gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'])
    total_gb, used_gb = map(lambda s: int(s) / 1e3, output.decode('utf-8').split(','))
    return total_gb, used_gb, total_gb - used_gb

def log_gpu_memory(msg=''):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    bt.logging.info(f"GPU-MEM {msg} Total: {t/1e9:.2f}GB, Reserved: {r/1e9:.2f}GB, Allocated: {a/1e9:.2f}GB")

def cleanup_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def compute_s2s_metrics(model_id: str, hf_repo_id: str, local_dir: str, mini_batch: Dataset, hotkey: str, block, model_tracker, device: str='cuda'):
    cleanup_gpu_memory()
    log_gpu_memory('before container start')
    
    # Initialize Docker manager
    docker_manager = DockerManager(base_cache_dir=local_dir)
    try:
        # Start Docker container for the model
        container_url = docker_manager.start_container(
            uid=f"{model_id}_{int(time.time())}", 
            repo_id=hf_repo_id,
            gpu_id=0 if device == 'cuda' else None
        )
        # Check hotkey if provided
        if hotkey is not None:
            try:
                target_file_path = huggingface_hub.hf_hub_download(
                    repo_id=hf_repo_id, 
                    filename="hotkey.txt",
                    local_dir=Path(local_dir) / hf_repo_id
                )
                with open(target_file_path, 'r') as file:
                    hotkey_contents = file.read()
                if hotkey_contents != hotkey:
                    bt.logging.warning("Hotkey mismatch. Returning score of 0.")
                    return 0
            except huggingface_hub.utils._errors.EntryNotFoundError:
                bt.logging.info("No hotkey file found in repository")
            except Exception as e:
                bt.logging.error(f"Error reading hotkey file: {str(e)}")
                return 0

        log_gpu_memory('after container start')
        
        # Initialize metrics
        metrics = {
            'mimi_score': [],
            'wer_score': [],
            'length_penalty': [],
            'pesq_score': [],
            'anti_spoofing_score': [],
            'combined_score': [],
            'total_samples': 0
        }

        # Process each sample in mini batch
        for i in range(len(mini_batch['youtube_id'])):
            youtube_id = mini_batch['youtube_id'][i]
            audio_array = np.array(mini_batch['audio'][i]['array'])
            sample_rate = mini_batch['audio'][i]['sampling_rate']

            diar_timestamps_start = np.array(mini_batch['diar_timestamps_start'][i])
            diar_timestamps_end = np.array(mini_batch['diar_timestamps_end'][i])
            diar_speakers = np.array(mini_batch['diar_speakers'][i])
            
            if len(diar_timestamps_start) == 1:
                continue

            test_idx = random.randint(0, len(diar_timestamps_start) - 2)
            diar_sample = audio_array[int(diar_timestamps_start[test_idx] * sample_rate):int(diar_timestamps_end[test_idx] * sample_rate)]
            diar_gt = audio_array[int(diar_timestamps_start[test_idx+1] * sample_rate):int(diar_timestamps_end[test_idx+1] * sample_rate)]

            # Check minimum length
            min_samples = int(0.25 * sample_rate)
            if len(diar_sample) < min_samples or len(diar_gt) < min_samples:
                continue

            # Reshape audio to [C,T] format (mono -> 1 channel)
            diar_sample = diar_sample.reshape(1, -1)

            # Perform inference using Docker container
            try:
                result = docker_manager.inference(
                    url=container_url,
                    audio_array=diar_sample,
                    sample_rate=sample_rate
                )
                pred_audio = result['audio']
                
                if pred_audio is None or len(pred_audio) == 0 or pred_audio.shape[-1] == 0:
                    for k, v in metrics.items():
                        if k == 'total_samples':
                            metrics[k] += 1
                        else:
                            metrics[k].append([0])
                    continue

                metrics['total_samples'] += 1
                
                # Compute metrics using S2SMetrics
                s2s_metrics = S2SMetrics(cache_dir="./model_cache")
                metrics_dict = s2s_metrics.compute_distance(
                    gt_audio_arrs=[[diar_gt, sample_rate]], 
                    generated_audio_arrs=[[pred_audio, sample_rate]]
                )
                
                for key, value in metrics_dict.items():
                    metrics[key].append(value)
                    
            except Exception as e:
                bt.logging.error(f"Error during inference: {str(e)}")
                continue

        mean_score = np.mean(metrics['combined_score'])
        bt.logging.info(f"Scoring {model_id} {hf_repo_id} complete: {mean_score:0.5f}")
        
        return mean_score
        
    except Exception as e:
        bt.logging.error(f"Error in compute_s2s_metrics: {str(e)}")
        return 0
        
    finally:
        # Cleanup
        try:
            docker_manager.cleanup_docker_resources()
        except Exception as e:
            bt.logging.error(f"Error cleaning up Docker resources: {str(e)}")
        cleanup_gpu_memory()
        log_gpu_memory('after cleanup')

if __name__ == "__main__":
    from utilities.temp_dir_cache import TempDirCache
    temp_dir_cache = TempDirCache(10)
    
    for epoch in range(2):
        for hf_repo_id in ["tezuesh/moshi_general", "tezuesh/moshi_general"]:
            start_time = time.time()
            diar_time = time.time()
            
            mini_batch = pull_latest_diarization_dataset()
            bt.logging.info(f"Time taken for diarization dataset: {time.time() - diar_time:.2f} seconds")
            
            local_dir = './model_cache'
            hotkey = None
            block = 1
            model_tracker = None
            
            vals = compute_s2s_metrics(
                model_id="moshi",
                hf_repo_id=hf_repo_id,
                mini_batch=mini_batch,
                local_dir=local_dir,
                hotkey=hotkey,
                block=block,
                model_tracker=model_tracker
            )
            
            end_time = time.time()
            bt.logging.info(f"Processing {hf_repo_id} complete. Time taken: {end_time - start_time:.2f} seconds")
            bt.logging.info(f"Combined score: {vals}")
            exit(0)