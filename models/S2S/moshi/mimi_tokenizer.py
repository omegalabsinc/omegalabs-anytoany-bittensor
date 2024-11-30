from moshi import models
loaders = models.loaders
from huggingface_hub import hf_hub_download
import torch
from pydub import AudioSegment
import numpy as np

MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'kyutai/moshiko-pytorch-bf16'


device = "cuda" if torch.cuda.is_available() else "cpu"
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)

def encode_audio(mimi, wav, device):
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    all_codes = []
    with torch.no_grad(), mimi.streaming(batch_size=1):
        for offset in range(0, wav.shape[-1], frame_size):
            frame = wav[:, :, offset: offset + frame_size]
            codes = mimi.encode(frame.to(device))
            assert codes.shape[-1] == 1, codes.shape
            all_codes.append(codes)
    
    return all_codes


def load_audio(wav_path, mimi):
    audio = AudioSegment.from_wav(wav_path)
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32) / (2**15 if audio.sample_width == 2 else 2**31)
    wav = torch.from_numpy(samples).float().unsqueeze(0).unsqueeze(0)

    if audio.frame_rate != mimi.sample_rate:
        wav = torch.nn.functional.interpolate(wav, scale_factor=mimi.sample_rate/audio.frame_rate, mode='linear', align_corners=False)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    wav = wav[:, :, :(wav.shape[-1] // frame_size) * frame_size]

    return wav

gt, generated = "/workspace/tezuesh/omega-v2v/.extracted/Fresh_Air_Best_Of_Jeremy_Strong_Will_Harpers_Roadtrip_Across_America_sample/0000009512.wav", "/workspace/tezuesh/omega-v2v/.predictions_warmup/moshi/audio/Fresh_Air_Best_Of_Jeremy_Strong_Will_Harpers_Roadtrip_Across_America_sample/0000000013.wav"

gt_wav = load_audio(gt, mimi)
generated_wav = load_audio(generated, mimi)

print("gt_wav", gt_wav.shape)
print("generated_wav", generated_wav.shape)
mimi_tokens = encode_audio(mimi, gt_wav, device)
mimi_tokens = torch.cat(mimi_tokens, dim=-1)
print("GT TOKENS", mimi_tokens.shape)
# [0].shape)

print(mimi_tokens[0,0])
# mimi_tokens = encode_audio(mimi, load_audio(generated, mimi), device)
# print("GENERATED TOKENS", mimi_tokens)







