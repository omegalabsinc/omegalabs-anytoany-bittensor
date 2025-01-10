
import os
from transformers import MimiModel, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = "models/"

def download_models():
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Download evaluation models
    print("Downloading MIMI model...")
    MimiModel.from_pretrained("kyutai/mimi", cache_dir=MODELS_DIR)
    AutoFeatureExtractor.from_pretrained("kyutai/mimi", cache_dir=MODELS_DIR)

    print("Downloading Whisper model...")
    WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=MODELS_DIR)
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", cache_dir=MODELS_DIR)

    print("Downloading RawNet3...")
    hf_hub_download(repo_id="jungjee/RawNet3", filename="model.pt", local_dir=MODELS_DIR)

    # Download inference model
    print("Downloading Moshi model...")
    snapshot_download(
        repo_id="tezuesh/moshi_general",
        local_dir=os.path.join(MODELS_DIR, "moshi_general"),
        cache_dir=MODELS_DIR
    )

    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()