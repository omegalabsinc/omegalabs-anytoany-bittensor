import os
import io
from tempfile import TemporaryDirectory
import time
from typing import List, Optional
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import subprocess   

import torch
import ulid
from omegaconf import OmegaConf, DictConfig
import huggingface_hub
from datasets import load_dataset, Dataset
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from imagebind.models.imagebind_model import ModalityType

import bittensor as bt



import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from tune_recipes.gen import InferenceRecipe
from tune_recipes.gen import InferenceRecipe

from neurons.distance import levenshtein_distance_with_length_penalty


HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 4 * 60 * 60  # 4 hours
MAX_FILES = 1
MODEL_FILE_PREFIX = "meta_model"
CONFIG_FILE = "training_config.yml"
BPE_PATH = "./models/bpe_simple_vocab_16e6.txt.gz"

CACHE_DIR = ".checkpoints"
CACHE_EXPIRY_HOURS = 24

LENGTH_DIFF_PENALTY_STEEPNESS = 2
SIMILARITY_WEIGHT = 0.5

def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp


def pull_latest_omega_dataset() -> Optional[Dataset]:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]
    if len(recent_files) == 0:
        return None
    with TemporaryDirectory(dir='./data_cache') as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir)["train"]
        omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
    return omega_dataset


def load_ckpt_from_hf(hf_repo_id: str) -> InferenceRecipe:
    # assert False, "make sure not to cache downloaded checkpoints"
    hf_api = huggingface_hub.HfApi()
    ckpt_files = [f for f in hf_api.list_repo_files(repo_id=hf_repo_id) if f.startswith(MODEL_FILE_PREFIX)]
    if len(ckpt_files) == 0:
        raise ValueError(f"No checkpoint files found in {hf_repo_id}")
    with TemporaryDirectory() as temp_dir:
        config_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=CONFIG_FILE, local_dir=temp_dir)
        ckpt_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=ckpt_files[0], local_dir=temp_dir)
        train_cfg = OmegaConf.load(config_path)
        train_cfg.model = DictConfig({
            "_component_": "models.mmllama3_8b",
            "use_clip": False,
            "perception_tokens": train_cfg.model.perception_tokens,
        })
        train_cfg.checkpointer.checkpoint_dir = os.path.dirname(ckpt_path)
        train_cfg.checkpointer.checkpoint_files = [os.path.basename(ckpt_path)]
        train_cfg.tokenizer.path = "./models/tokenizer.model"
        train_cfg.inference.max_new_tokens = 300
        
        inference_recipe = InferenceRecipe(train_cfg)
        inference_recipe.setup(cfg=train_cfg)
        

    return inference_recipe, train_cfg

def is_file_outdated(file_path: str, expiry_hours: int) -> bool:
    """Check if the file is older than the specified expiry time."""
    if not os.path.exists(file_path):
        return True
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    return datetime.now() - file_mod_time > timedelta(hours=expiry_hours)

def load_ckpt_from_hf_cached(hf_repo_id: str, refresh_cache: bool = True) -> InferenceRecipe:
    hf_api = huggingface_hub.HfApi()
    ckpt_files = [f for f in hf_api.list_repo_files(repo_id=hf_repo_id) if f.startswith(MODEL_FILE_PREFIX)]
    if len(ckpt_files) == 0:
        raise ValueError(f"No checkpoint files found in {hf_repo_id}")

    # Create a unique subdirectory for each repository within the cache directory
    repo_cache_dir = os.path.join(CACHE_DIR, hf_repo_id.replace("/", "_"))
    os.makedirs(repo_cache_dir, exist_ok=True)

    # Define paths for the config and checkpoint files
    config_path = os.path.join(repo_cache_dir, CONFIG_FILE)
    ckpt_path = os.path.join(repo_cache_dir, ckpt_files[0])

    # Check if the files are outdated
    if is_file_outdated(config_path, CACHE_EXPIRY_HOURS) and refresh_cache:
        if os.path.exists(config_path):
            os.remove(config_path)
        config_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=CONFIG_FILE, local_dir=repo_cache_dir)

    if is_file_outdated(ckpt_path, CACHE_EXPIRY_HOURS) and refresh_cache:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        ckpt_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=ckpt_files[0], local_dir=repo_cache_dir)

    train_cfg = OmegaConf.load(config_path)
    train_cfg.model = DictConfig({
        "_component_": "models.mmllama3_8b",
        "use_clip": False,
        "perception_tokens": train_cfg.model.perception_tokens,
    })
    train_cfg.checkpointer.checkpoint_dir = os.path.dirname(ckpt_path)
    train_cfg.checkpointer.checkpoint_files = [os.path.basename(ckpt_path)]
    train_cfg.inference.max_new_tokens = 300
    train_cfg.tokenizer.path = "./models/tokenizer.model"
    
    inference_recipe = InferenceRecipe(train_cfg)
    inference_recipe.setup(cfg=train_cfg)

    return inference_recipe, train_cfg


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

def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens

def embed_text(imagebind, texts: List[str], device) -> List[torch.FloatTensor]:
    return imagebind({ModalityType.TEXT: load_and_transform_text(texts, device)})[ModalityType.TEXT]

def tokenize_text(tokenizer, texts: List[str]) -> List[int]:
    return [tokenizer.encode(text, add_eos=False, add_bos=False) for text in texts]

def get_model_score(hf_repo_id, mini_batch, local_dir, hotkey="default_hotkey", block=0, model_tracker=None):
    # cleanup_gpu_memory()
    # log_gpu_memory('before model load')
    inference_recipe, config = load_ckpt_from_hf(hf_repo_id)

    # Check if the model is unique. Calculates the model's checkpoint (.pt) file hash for storage.
    # is_model_unique, model_hash = model_tracker.is_model_unique(
    #     hotkey, 
    #     block, 
    #     config.checkpointer.checkpoint_dir + "/" + config.checkpointer.checkpoint_files[0]
    # )
    # if is_model_unique:
    #     bt.logging.info(f"Model with hash {model_hash} on block {block} is unique.")
    # else:
    #     bt.logging.warning(f"*** Model with hash {model_hash} on block {block} is not unique. Returning score of 0. ***")
    #     cleanup_gpu_memory()
    #     log_gpu_memory('after model clean-up')
    #     return 0

    # bt.logging.info(f"Scoring {hf_repo_id}...")
    # log_gpu_memory('after model load')

    weighed_metric_list = []
    
    for video_emb, actual_caption in zip(mini_batch["video_embed"], mini_batch["description"]):
        generated_caption = inference_recipe.generate_batch(cfg=config, embeddings=[{"Video":video_emb}], prompt="Caption the previous video.")
        text_embeddings = embed_text(inference_recipe._embed_model, [generated_caption, actual_caption], device=inference_recipe._device)
        text_similarity = torch.nn.functional.cosine_similarity(text_embeddings[0], text_embeddings[1], dim=-1)


        gt_caption_tokens, generated_caption_tokens = tokenize_text(
            inference_recipe._tokenizer,
            [actual_caption, generated_caption[0]]
        )
        
        levenshtein_distance = levenshtein_distance_with_length_penalty(
            generated_caption_tokens,
            gt_caption_tokens,
            device=inference_recipe._device,
            length_diff_penalty_steepness=LENGTH_DIFF_PENALTY_STEEPNESS,
            dtype=inference_recipe._dtype
        )

        weighed_metrics = (
            text_similarity.item() * SIMILARITY_WEIGHT +
            levenshtein_distance * (1 - SIMILARITY_WEIGHT)
        )
        weighed_metric_list.append(weighed_metrics)


    mean_distance = torch.tensor(weighed_metric_list).mean().item() 
    bt.logging.info(f"Scoring {hf_repo_id} complete: {mean_distance:0.5f}")
    cleanup_gpu_memory()
    log_gpu_memory('after model clean-up')
    return mean_distance


if __name__ == "__main__":
    from utilities.temp_dir_cache import TempDirCache
    temp_dir_cache = TempDirCache(10)
    for epoch in range(2):
        mini_batch = pull_latest_omega_dataset()
        for hf_repo_id in ["briggers/omega_a2a_test2", "salmanshahid/omega_a2a_test", "briggers/omega_a2a_test",]:
            local_dir = temp_dir_cache.get_temp_dir(hf_repo_id)
            local_dir = './model_cache' #temp_dir_cache.get_temp_dir(hf_repo_id)
            get_model_score(hf_repo_id, mini_batch, local_dir)
