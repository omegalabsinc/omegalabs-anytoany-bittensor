import os
from omegaconf import OmegaConf
from huggingface_hub import HfApi
from model.data import Model, ModelId
from model.storage.disk import utils
from constants import CompetitionParameters, MAX_HUGGING_FACE_BYTES

from model.storage.remote_model_store import RemoteModelStore
from huggingface_hub import HfApi, file_exists
from collections import defaultdict


MODEL_FILE_PT = "meta_model_{epoch}.pt"
ADAPTER_FILE_PT = "adapter_{epoch}.pt"
CONFIG_FILE = "training_config.yml"
README_FILE = "README.md"


def check_config(ckpt_dir):
    config_file = os.path.join(ckpt_dir, CONFIG_FILE)
    cfg = OmegaConf.load(config_file)
    if cfg.model.use_clip:
        raise ValueError("Cannot upload checkpoints with CLIP embeddings")


def get_required_files(epoch: int):
    return [
        MODEL_FILE_PT.format(epoch=epoch),
        CONFIG_FILE,
    ]


def export_readme(ckpt_dir: str):
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

Check out the [git repo](https://github.com/omegalabsinc/omegalabs-anytoany-bittensor) and find OMEGA on X: [@omegalabsai](https://x.com/omegalabsai).
"""
        )


class HuggingFaceModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")


    async def upload_model(
        self, model: Model, 
        competition_parameters: CompetitionParameters,
    ) -> ModelId:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceModelStore.assert_access_token_exists()
        api = HfApi(token=token)
        export_readme(model.local_repo_dir)
        files_to_upload = get_required_files(model.id.epoch) + [README_FILE]
        hf_repo_id = model.id.namespace + "/" + model.id.name
        api.create_repo(
            repo_id=hf_repo_id,
            exist_ok=True,
            private=True,
        )
        for filename in files_to_upload:
            commit_info = api.upload_file(repo_id=hf_repo_id, path_in_repo=filename, path_or_fileobj=os.path.join(model.local_repo_dir, filename))
        print(f"Successfully uploaded checkpoint '{model.local_repo_dir}' @ epoch={model.id.epoch} to {hf_repo_id}")
        
        model_id_with_commit = ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            epoch=model.id.epoch,
            hash=model.id.hash,
            commit=commit_info.oid,
            competition_id=model.id.competition_id,
        )
        
        return model_id_with_commit
        # # TODO consider skipping the redownload if a hash is already provided.
        # # To get the hash we need to redownload it at a local tmp directory after which it can be deleted.
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     model_with_hash = await self.download_model(
        #         model_id_with_commit, temp_dir, competition_parameters
        #     )
        #     # Return a ModelId with both the correct commit and hash.
        #     return model_with_hash.id

    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        model_parameters: CompetitionParameters,
    ) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")

        repo_id = model_id.namespace + "/" + model_id.name

        # Check ModelInfo for the size of model.safetensors file before downloading.
        try:
            token = HuggingFaceModelStore.assert_access_token_exists()
        except:
            token = None
        api = HfApi(token=token)
        model_info = api.model_info(
            repo_id=repo_id, revision=model_id.commit, timeout=10, files_metadata=True
        )
        size = sum(repo_file.size for repo_file in model_info.siblings)
        if size > MAX_HUGGING_FACE_BYTES:
            raise ValueError(
                f"Hugging Face repo over maximum size limit. Size {size}. Limit {MAX_HUGGING_FACE_BYTES}."
            )

        api.hf_hub_download(
            repo_id=repo_id,
            revision=model_id.commit,
            filename="checkpoint.safetensors",
            cache_dir=local_path,
        )

        # Get the directory the model was stored to.
        model_dir = utils.get_hf_download_path(local_path, model_id)

        # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
        utils.realize_symlinks_in_directory(model_dir)

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(model_dir)
        model_id_with_hash = ModelId(
            namespace=model_id.namespace,
            name=model_id.name,
            commit=model_id.commit,
            hash=model_hash,
            competition_id=model_id.competition_id,
        )

        return Model(id=model_id_with_hash, ckpt=model_dir)
