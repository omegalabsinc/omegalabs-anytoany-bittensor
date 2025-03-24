import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import time
import torch
import ulid
import huggingface_hub
from datasets import load_dataset, Dataset, DownloadConfig
import bittensor as bt

from evaluation.VideoCapt.ib_wrapper import ImageBind

from neurons.docker_manager import DockerManager
from utilities.gpu import log_gpu_memory, cleanup_gpu_memory
from constants import MAX_DS_FILES, MIN_AGE
from tempfile import TemporaryDirectory
import constants

from utilities.compare_block_and_model import compare_block_and_model


# Constants
HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MODEL_FILE_PREFIX = "meta_model"


class IBModel:
    def __init__(self):
        self.embed_model = ImageBind(v2=True)

    def embed_text(self, text):
        return self.embed_model.embed_text(text)


def get_timestamp_from_filename(filename: str):
    """Extract timestamp from filename using ULID."""
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

def pull_latest_dataset() -> Optional[Dataset]:
    """Pull latest dataset from HuggingFace."""
    try:
        os.system("rm -rf ./data_cache/*")
        omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
        recent_files = [
            f.rfilename
            for f in omega_ds_files if
            f.rfilename.startswith(DATA_FILES_PREFIX) and
            time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
        ][:MAX_DS_FILES]

        if len(recent_files) == 0:
            return None

        download_config = DownloadConfig(download_desc="Downloading Omega Multimodal Dataset")

        with TemporaryDirectory(dir='./data_cache') as temp_dir:
            print("temp_dir", temp_dir)
            omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir, download_config=download_config)["train"]
            omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
            return omega_dataset
        
    except Exception as e:
        bt.logging.error(f"Error pulling dataset: {str(e)}")
        return None


def pull_latest_dataset_fallback() -> Optional[Dataset]:
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

        download_config = DownloadConfig(download_desc="Downloading Omega Multimodal Dataset")

        with TemporaryDirectory(dir='./data_cache') as temp_dir:
            print("temp_dir", temp_dir)
            omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir, download_config=download_config)["train"]
            omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
            return omega_dataset
        
    except Exception as e:
        bt.logging.error(f"Error pulling dataset: {str(e)}")
        return None

def verify_hotkey(hf_repo_id: str, local_dir: str, hotkey: str) -> bool:
    """Verify hotkey matches the one in the repository."""
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
            return False
        return True
    except huggingface_hub.utils.EntryNotFoundError:
        bt.logging.info("No hotkey file found in repository")
        return True
    except Exception as e:
        bt.logging.error(f"Error reading hotkey file: {str(e)}")
        return False

def compute_model_score(
    hf_repo_id: str,
    local_dir: str,
    mini_batch: Dataset,
    hotkey: Optional[str] = None,
    block: Optional[int] = None,
    model_tracker: Any = None,
    device: str = 'cuda'
) -> float:
    """
    Compute model score using Docker-based inference.
    
    Args:
        hf_repo_id: HuggingFace repository ID
        local_dir: Local directory for caching
        mini_batch: Dataset batch to process
        hotkey: Optional hotkey to verify
        block: Optional block number
        model_tracker: Optional model tracker
        device: Device to use for computation
    
    Returns:
        float: Computed score
    """
    cleanup_gpu_memory()
    log_gpu_memory('before container start')
    if not compare_block_and_model(block, hf_repo_id):
        bt.logging.info(f"Block {block} is older than model {hf_repo_id}. Penalizing model.")
        return constants.penalty_score

    embed_model = IBModel()

    # Initialize Docker manager
    docker_manager = DockerManager(base_cache_dir=local_dir)

    try:
        # Start Docker container
        container_url = docker_manager.start_container(
            uid=f"{int(time.time())}",
            repo_id=hf_repo_id,
            gpu_id=0 if device == 'cuda' else None
        )

        # Verify hotkey if provided
        if hotkey and not verify_hotkey(hf_repo_id, local_dir, hotkey):
            return constants.penalty_score

        log_gpu_memory('after container start')

        # Process batch and compute scores
        similarities = []
        batch_size = 4  # Process one at a time since server expects single embeddings
        print("len(mini_batch['video_embed'])", len(mini_batch["video_embed"]))

        for idx in range(0, len(mini_batch["video_embed"]), batch_size):
            print("idx", idx)
            video_embed = torch.tensor(mini_batch["video_embed"][idx:idx+batch_size])
            actual_caption = mini_batch["description"][idx:idx+batch_size]
            print("video_embed", video_embed.shape)
            
            # Perform inference using Docker container
            try:
                # Convert tensor batch to list of float lists
                video_embed_list = video_embed.flatten().tolist()
                

                # Get generated captions from Docker container
                result = docker_manager.inference_ibllama(
                    url=container_url,
                    video_embed=video_embed_list
                )
                
                generated_caption = result.get('captions', [])
                
                if not generated_caption:
                    continue
                
                if len(generated_caption) != len(actual_caption):
                    similarities.extend([constants.penalty_score] * len(actual_caption))
                    continue

                # Get text embeddings for similarity computation
                generated_embeddings = embed_model.embed_text(
                    text=generated_caption
                )
                actual_embeddings = embed_model.embed_text(
                    text=actual_caption
                )
                
                if generated_embeddings is None or actual_embeddings is None:
                    continue

                # Compute similarity between generated and actual captions
                text_similarity = torch.nn.functional.cosine_similarity(
                    generated_embeddings,  # First embedding (generated)
                    actual_embeddings,   # Second embedding (actual)
                    dim=-1
                )
                # print("text_similarity", text_similarity)
                similarities.extend(text_similarity.tolist())

            except Exception as e:
                bt.logging.error(f"Error during inference: {str(e)}")
                continue

        mean_similarity = torch.tensor(similarities).mean().item() if similarities else 0
        bt.logging.info(f"Scoring {hf_repo_id} complete: {mean_similarity:0.5f}")

        return mean_similarity

    finally:
        # Cleanup
        try:
            docker_manager.cleanup_docker_resources()
        except Exception as e:
            bt.logging.error(f"Error cleaning up Docker resources: {str(e)}")
        cleanup_gpu_memory()
        log_gpu_memory('after cleanup')

def run_o1_scoring(hf_repo_id: str, hotkey: str, block: int, model_tracker: str, local_dir: str):
    mini_batch = pull_latest_dataset()

    if mini_batch is None:
        bt.logging.error("Failed to pull latest dataset, trying fallback")
        mini_batch = pull_latest_dataset_fallback()

    start_time = time.time()
    score = compute_model_score(
        hf_repo_id=hf_repo_id,
        mini_batch=mini_batch,
        local_dir=local_dir,
        hotkey=hotkey,
        block=block,
        model_tracker=model_tracker
    )
    
    end_time = time.time()
    bt.logging.info(f"Processing {hf_repo_id} complete. Time taken: {end_time - start_time:.2f} seconds")
    bt.logging.info(f"Score: {score}")
    return score
if __name__ == "__main__":

    # Example usage
    score = run_o1_scoring(hf_repo_id="TFOCUS/mfm_8", hotkey=None, block=1, model_tracker=None, local_dir="./model_cache")
    print("score", score)