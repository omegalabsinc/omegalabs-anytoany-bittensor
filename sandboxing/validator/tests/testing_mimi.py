from transformers import MimiModel, AutoFeatureExtractor

print("Loading MIMI model and feature extractor...")
model = MimiModel.from_pretrained("kyutai/mimi")
model.config.num_quantizers = 4
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
import librosa

import numpy as np

print("Generating test audio signals...")
# Create two random audio signals
sample_rate = 16000  # Standard sample rate
duration = 3  # 3 seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# First audio: Mix of sine waves at different frequencies
print("Creating first audio signal with sine waves...")
audio1 = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
audio1 = audio1 / np.max(np.abs(audio1))  # Normalize
print(f"Audio 1 shape: {audio1.shape}")

# Second audio: White noise with some filtering
print("Creating second audio signal with filtered noise...")
audio2 = np.random.randn(len(t))
# Apply simple moving average to smooth the noise
window_size = 50
audio2 = np.convolve(audio2, np.ones(window_size)/window_size, mode='same')
audio2 = audio2 / np.max(np.abs(audio2))  # Normalize
print(f"Audio 2 shape: {audio2.shape}")

print("Resampling audio signals...")
gt_audio = librosa.resample(
    audio1, 
    orig_sr=sample_rate, 
    target_sr=feature_extractor.sampling_rate
)
generated_audio = librosa.resample(
    audio2, 
    orig_sr=sample_rate, 
    target_sr=feature_extractor.sampling_rate
)
print(f"Resampled audio shapes - GT: {gt_audio.shape}, Generated: {generated_audio.shape}")

print("Preparing inputs for MIMI model...")
gt_inputs = feature_extractor(
    raw_audio=gt_audio,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt"
)
generated_inputs = feature_extractor(
    raw_audio=generated_audio,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt"
)


print("Encoding audio with MIMI model...")
gt_encoding = model.encode(gt_inputs["input_values"], gt_inputs["padding_mask"])
generated_encoding = model.encode(generated_inputs["input_values"], generated_inputs["padding_mask"])

print("Audio codes shapes:")
print(f"Ground truth encoding shape: {gt_encoding.audio_codes.shape}")
print(f"Generated encoding shape: {generated_encoding.audio_codes.shape}")


gt_codes = gt_encoding.audio_codes
generated_codes = generated_encoding.audio_codes
import torchaudio.functional as F

for i in range(gt_codes.shape[1]):
    gt_seq = gt_codes[:, i]# Move to CPU for edit distance
    gen_seq = generated_codes[:, i]
    print(gt_seq.shape)
    print(gen_seq.shape)
    print(f"idx = {i} , edit_distance = {F.edit_distance(gt_seq[0], gen_seq[0])}")
