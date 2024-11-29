import os
import sys
from typing import Dict

import numpy as np
import librosa
import torch
import torch.nn.functional as F

from evaluation.S2S.rawnet.models.RawNet3 import RawNet3
from evaluation.S2S.rawnet.models.RawNetBasicBlock import Bottle2neck

from huggingface_hub import hf_hub_download

class RawNet3Inference:
    def __init__(self, model_name = 'jungjee/RawNet3', repo_dir = "models", device="cuda"):
        self.model = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1,
        )
        self.device = torch.device(device)

        
        model_path = repo_dir
        temp_location = hf_hub_download(repo_id=model_name, repo_type='model', filename='model.pt', local_dir=model_path)
        self.model.load_state_dict(torch.load(temp_location, weights_only=True)['model'])
        self.model.eval()

   
        self.model = self.model.to(self.device)

    def inference_utterance(self, input_file: str, n_segments: int, out_dir: str) -> None:
        output = self.extract_speaker_embd(
            fn=input_file,
            n_samples=48000,
            n_segments=n_segments,
        ).mean(0)

        return output

    def extract_speaker_embd(
        self, audio_arr: np.ndarray, sample_rate: int, n_samples: int, n_segments: int = 10
    ) -> np.ndarray:
        # audio, sample_rate = librosa.resample(fn, sr=None)
        if len(audio_arr.shape) > 1:
            audio_arr = np.mean(audio_arr, axis=1)

        if sample_rate != 16000:
            audio_arr = librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        if (
            len(audio_arr) < n_samples
        ):  # RawNet3 was trained using utterances of 3 seconds
            shortage = n_samples - len(audio_arr) + 1
            audio_arr = np.pad(audio_arr, (0, shortage), "wrap")
        
        audios = []
        startframe = np.linspace(0, len(audio_arr) - n_samples, num=n_segments)
        for asf in startframe:
            audios.append(audio_arr[int(asf) : int(asf) + n_samples])

        audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
        if self.device.type == "cuda":
            audios = audios.to(self.device)
        with torch.no_grad():
            output = self.model(audios)

        return output

if __name__ == "__main__":
    inference = RawNet3Inference()
    # Example usage:
    audio_pth = "/workspace/tezuesh/omega-v2v/.filtered/Fresh_Air_Remembering_Gospel_Singer_Cissy_Houston_MLB_Legend_Pete_Rose_sample/0000076993.wav"
    print(inference.extract_speaker_embd(audio_pth, 48000, 10).shape)
    # inference.inference_utterance("path/to/input.wav", 10, "./out.npy")
