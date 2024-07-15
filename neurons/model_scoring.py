import os
from tempfile import TemporaryDirectory
import time
from typing import List, Optional
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

from tune_recipes.gen import InferenceRecipe


HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 4 * 60 * 60  # 4 hours
MAX_FILES = 8
MODEL_FILE_PREFIX = "meta_model"
CONFIG_FILE = "training_config.yml"
BPE_PATH = "./models/bpe_simple_vocab_16e6.txt.gz"


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


def load_ckpt_from_hf(hf_repo_id: str, local_dir) -> InferenceRecipe:
    repo_dir = Path(local_dir) / hf_repo_id
    bt.logging.info(f"Loading ckpt {hf_repo_id}, repo_dir: {repo_dir}")

    hf_api = huggingface_hub.HfApi()
    ckpt_files = [f for f in hf_api.list_repo_files(repo_id=hf_repo_id) if f.startswith(MODEL_FILE_PREFIX)]
    if len(ckpt_files) == 0:
        raise ValueError(f"No checkpoint files found in {hf_repo_id}")

    config_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=CONFIG_FILE, local_dir=repo_dir)
    ckpt_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=ckpt_files[0], local_dir=repo_dir)
    train_cfg = OmegaConf.load(config_path)
    train_cfg.model = DictConfig({
        "_component_": "models.mmllama3_8b",
        "use_clip": False,
        "perception_tokens": train_cfg.model.perception_tokens,
    })
    train_cfg.batch_size = 4
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


def get_model_score(hf_repo_id, mini_batch, local_dir):
    cleanup_gpu_memory()
    inference_recipe, config = load_ckpt_from_hf(hf_repo_id, local_dir)
    bt.logging.info(f"Scoring {hf_repo_id}...")
    log_gpu_memory('after model load')
    batch_dim = config.batch_size
    similarities = []

    for idx in range(0, len(mini_batch["video_embed"]), batch_dim):
        video_embed = torch.tensor(mini_batch["video_embed"][idx:idx+batch_dim])
        actual_captions = mini_batch["description"][idx:idx+batch_dim]
        generated_captions = inference_recipe.generate_batch(
            cfg=config,
            video_ib_embed=video_embed
        )
        text_embeddings = embed_text(
            inference_recipe._embed_model,
            generated_captions + actual_captions,
            device=inference_recipe._device
        )
        text_similarity = torch.nn.functional.cosine_similarity(
            text_embeddings[:video_embed.size(0)],
            text_embeddings[video_embed.size(0):],
            dim=-1
        )
        similarities.extend(text_similarity.tolist())

    mean_similarity = torch.tensor(similarities).mean().item()
    bt.logging.info(f"Scoring {hf_repo_id} complete: {mean_similarity:0.5f}")
    cleanup_gpu_memory()
    log_gpu_memory('after model clean-up')
    return mean_similarity


if __name__ == "__main__":
    from utilities.temp_dir_cache import TempDirCache
    temp_dir_cache = TempDirCache(10)
    for epoch in range(2):
        mini_batch = pull_latest_omega_dataset()
        for hf_repo_id in ["briggers/omega_a2a_test2", "salmanshahid/omega_a2a_test", "briggers/omega_a2a_test",]:
            local_dir = temp_dir_cache.get_temp_dir(hf_repo_id)
            local_dir = './model_cache' #temp_dir_cache.get_temp_dir(hf_repo_id)
            get_model_score(hf_repo_id, mini_batch, local_dir)
