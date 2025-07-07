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
import bittensor as bt
from pathlib import Path
from neurons.docker_manager import DockerManager
from evaluation.S2S.distance import S2SMetrics
from utilities.gpu import log_gpu_memory, cleanup_gpu_memory
from constants import MAX_DS_FILES, MIN_AGE, penalty_score

from utilities.compare_block_and_model import compare_block_and_model

HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"

def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

def pull_latest_diarization_dataset() -> Optional[Dataset]:
    os.system("rm -rf ./data_cache/*")

    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_DS_FILES]

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


def pull_latest_diarization_dataset_fallback() -> Optional[Dataset]:
    """Pull latest dataset from HuggingFace."""
    try:
        os.system("rm -rf ./data_cache/*")
        omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
        recent_files = [
            f.rfilename
            for f in omega_ds_files if
            f.rfilename.startswith(DATA_FILES_PREFIX) 
        ][:MAX_DS_FILES]

        if len(recent_files) == 0:
            return None

        download_config = DownloadConfig(download_desc="Downloading Omega Voice Dataset")

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
        
    except Exception as e:
        bt.logging.error(f"Error pulling dataset: {str(e)}")
        return None
    
def compute_s2s_metrics(hf_repo_id: str, local_dir: str, mini_batch: Dataset, hotkey: str, block, model_tracker, device: str='cuda')->dict:
    cleanup_gpu_memory()
    log_gpu_memory('before container start')
    
    # Initialize Docker manager
    docker_manager = DockerManager(base_cache_dir=local_dir)
    try:
        # Sanitize the repository name for Docker tag
        # Replace invalid characters and ensure it follows Docker tag naming rules
        sanitized_name = hf_repo_id.split('/')[-1].lower()
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in sanitized_name)
        container_uid = f"{sanitized_name}_{int(time.time())}"
        
        # Start Docker container for the model
        container_url = docker_manager.start_container(
            uid=container_uid,
            repo_id=hf_repo_id,
            gpu_id=0 if device == 'cuda' else None
        )

        bt.logging.info(f"i am here inside compute_s2s_metrics {container_uid}, hotkey: {hotkey}")
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
                    return {'combined_score':penalty_score}
            except huggingface_hub.utils.EntryNotFoundError:
                bt.logging.info("No hotkey file found in repository")
            except Exception as e:
                bt.logging.error(f"Error reading hotkey file: {str(e)}")
                return {'combined_score':0}

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

        s2s_metrics = S2SMetrics(cache_dir=".checkpoints")

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
                result = docker_manager.inference_v2v(
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
                
                metrics_dict = s2s_metrics.compute_distance(
                    gt_audio_arrs=[[diar_gt, sample_rate]], 
                    generated_audio_arrs=[[pred_audio, sample_rate]]
                )
                
                for key, value in metrics_dict.items():
                    metrics[key].append(value)
                    
            except Exception as e:
                bt.logging.error(f"Error during inference: {str(e)}")
                continue
                
        result_dict = {
            'mimi_score': np.mean(metrics['mimi_score']),
            'wer_score': np.mean(metrics['wer_score']),
            'length_penalty': np.mean(metrics['length_penalty']),
            'pesq_score': np.mean(metrics['pesq_score']),
            'anti_spoofing_score': np.mean(metrics['anti_spoofing_score']),
            'combined_score': np.mean(metrics['combined_score'])
        }

        bt.logging.info(f"Scoring {hf_repo_id} complete: {result_dict['combined_score']:0.5f}")
        
        return result_dict

    finally:
        # Cleanup
        try:
            docker_manager.cleanup_docker_resources()
        except Exception as e:
            bt.logging.error(f"Error cleaning up Docker resources: {str(e)}")
        cleanup_gpu_memory()
        log_gpu_memory('after cleanup')

def run_v2v_scoring(hf_repo_id: str, hotkey: str, block: int, model_tracker: str, local_dir: str):
    if not compare_block_and_model(block, hf_repo_id):
        bt.logging.info(f"Block {block} is older than model {hf_repo_id}. Penalizing model.")
        return {"combined_score":penalty_score}
    
    start_time = time.time()
    diar_time = time.time()
    
    mini_batch = pull_latest_diarization_dataset()
    if mini_batch is None:
        bt.logging.info(f"Pulling fallback dataset.")
        mini_batch = pull_latest_diarization_dataset_fallback()
        if mini_batch is None:
            bt.logging.error(f"No diarization dataset found")
            return {"combined_score":0}
            
    bt.logging.info(f"Time taken for diarization dataset: {time.time() - diar_time:.2f} seconds")
    
    vals = compute_s2s_metrics(
        hf_repo_id=hf_repo_id,
        mini_batch=mini_batch,
        local_dir=local_dir,
        hotkey=hotkey,
        block=block,
        model_tracker=model_tracker
    ) #vals is a dict containing all metrics. with keys mimi_score, wer_score, length_penalty, pesq_score, anti_spoofing_score, combined_score
    
    end_time = time.time()
    bt.logging.info(f"Processing {hf_repo_id} complete. Time taken: {end_time - start_time:.2f} seconds")
    bt.logging.info(f"All scores: {vals}")
    return vals


if __name__ == "__main__":
    for epoch in range(2):
        for hf_repo_id in ["eggmoo/omega_gQdQiVq", "tezuesh/moshi_general"]:
            vals = run_v2v_scoring(hf_repo_id, hotkey=None, block=5268488, model_tracker=None, local_dir="./model_cache")
            print(vals)
            exit(0)
