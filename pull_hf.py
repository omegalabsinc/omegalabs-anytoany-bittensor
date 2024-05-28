from datasets import load_dataset, Dataset
import huggingface_hub
import ulid
import time


HF_DATASET = "omegalabsinc/omega-multimodal"
prefix = "default/train/"
suffix = ".parquet"
TWELVE_HOURS = 12 * 60 * 60
MAX_FILES = 8


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(filename[len(prefix):filename.find(suffix)]).timestamp().timestamp


def pull_latest_omega_dataset() -> Dataset:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(prefix) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < TWELVE_HOURS
    ][:MAX_FILES]
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files)["train"]
    return omega_dataset


def load_ckpt_from_hf(hf_repo_id: str):
    pass


def evaluate_checkpoint(ckpt_path: str, dataset: Dataset):
    pass


def main():
    omega_dataset = pull_latest_omega_dataset()
    ckpt_path = "output/Meta-Llama-3-8B-Instruct-NO-CLIP/meta_model_0.pt"
    config_yaml = "config/8B_lora.yaml"
    hf_repo_id = ""
    ckpt = load_ckpt_from_hf(hf_repo_id)
    ckpt_score = evaluate_checkpoint(ckpt_path, omega_dataset)
    print(f"Checkpoint '{hf_repo_id}' score: {ckpt_score}")


if __name__ == "__main__":
    main()
