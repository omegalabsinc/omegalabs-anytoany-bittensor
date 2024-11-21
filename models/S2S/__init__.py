import numpy as np
from models.S2S.moshi.inference import InferenceRecipe as moshi_inference_recipe

def inference(model_id: str, hf_repo_id: str, device: str='cuda'):
    if model_id == "moshi":
        return moshi_inference_recipe(hf_repo_id, device)
    else:
        raise ValueError(f"Model {model_id} not found")
