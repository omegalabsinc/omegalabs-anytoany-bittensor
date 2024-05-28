import os
import argparse

from huggingface_hub import HfApi
from omegaconf import OmegaConf


MODEL_FILE_PT = "meta_model_{epoch}.pt"
ADAPTER_FILE_PT = "adapter_{epoch}.pt"
CONFIG_FILE = "training_config.yml"
README_FILE = "README.md"


def parse_args():
    # args should be ckpt_dir, epoch, hf_repo_id
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help="The path to checkpoint directory that you used as checkpointer.output_dir e.g. /path/to/ckpt_dir")
    parser.add_argument('--epoch', type=int, required=True, help="The epoch number to submit as your checkpoint to evaluate e.g. 10")
    parser.add_argument('--hf_repo_id', type=str, required=True, help="full HF repo ID you would like to upload the model to e.g. omegalabsinc/any_to_any_v1")
    args = parser.parse_args()
    return args


def get_required_files(epoch: int):
    return [
        MODEL_FILE_PT.format(epoch=epoch),
        ADAPTER_FILE_PT.format(epoch=epoch),
        CONFIG_FILE,
    ]


def validate_repo(ckpt_dir, epoch):
    for filename in get_required_files(epoch):
        filepath = os.path.join(ckpt_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file {filepath} not found in {ckpt_dir}")


def upload_checkpoint(ckpt_dir, epoch, hf_repo_id):
    files_to_upload = get_required_files(epoch) + [README_FILE]
    api = HfApi(token=os.environ["HF_ACCESS_TOKEN"])
    api.create_repo(
        repo_id=hf_repo_id,
        exist_ok=True,
        private=True,
    )
    for filename in files_to_upload:
        api.upload_file(repo_id=hf_repo_id, path_in_repo=filename, path_or_fileobj=os.path.join(ckpt_dir, filename))
    print(f"Successfully uploaded checkpoint '{ckpt_dir}' @ epoch={epoch} to {hf_repo_id}")


def check_config(ckpt_dir):
    config_file = os.path.join(ckpt_dir, CONFIG_FILE)
    cfg = OmegaConf.load(config_file)
    if cfg.model.use_clip:
        raise ValueError("Cannot upload checkpoints with CLIP embeddings")


def export_readme(ckpt_dir):
    readme_file = os.path.join(ckpt_dir, README_FILE)
    with open(readme_file, "w") as f:
        f.write(
            f"""---
license: mit
tags:
- any-to-any
- omega
- omegalabs
- bittensor
- agi
---

This is an Any-to-Any model checkpoint for the OMEGA Labs x Bittensor Any-to-Any subnet.

Check out the [git repo](https://github.com/omegalabsinc/omegalabs-bittensor-anytoany) and find us on X: [@omegalabsai](https://x.com/omegalabsai)."""
        )


def main():
    args = parse_args()
    validate_repo(args.ckpt_dir, args.epoch)
    check_config(args.ckpt_dir)
    export_readme(args.ckpt_dir)
    upload_checkpoint(args.ckpt_dir, args.epoch, args.hf_repo_id)


if __name__ == '__main__':
    main()
