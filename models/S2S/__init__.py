import numpy as np
from models.S2S.moshi.inference import inference as moshi_inference

def inference(model_id: str, audio_array: np.array, sample_rate: int):
    if model_id == "moshi":
        return moshi_inference(audio_array, sample_rate)
    else:
        raise ValueError(f"Model {model_id} not found")
