import os
from pydub import AudioSegment
from pydub.silence import detect_silence
import glob
from tqdm import tqdm
import numpy as np

def detect_and_trim_silence(audio_array, frame_rate, min_silence_duration=1000, silence_threshold=-40):
    # Load the audio file
    audio_array = audio_array.detach().cpu().numpy()
    audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
    
    # Create AudioSegment from raw audio bytes
    audio = AudioSegment(
        data=audio_bytes,
        sample_width=2,  # 16-bit audio = 2 bytes
        frame_rate=frame_rate,
        channels=1  # Mono audio
    )

    # Detect silence
    silence_intervals = detect_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_threshold
    )

    # Convert milliseconds to seconds
    silence_intervals_seconds = [(start / 1000, end / 1000) for start, end in silence_intervals]

    first_silence_end = silence_intervals_seconds[0][1]
    # Create audio from first silence end to end of audio
    trimmed_audio = audio_array[:,:,int(first_silence_end * frame_rate):]  # Slice audio from first silence end to the end

    # trimmed_audio.export(output_path, format="wav")
    return trimmed_audio

def process_all_audio_files(root_dir, output_dir):
    # Use glob to find all .wav files in the directory and its subdirectories
    wav_files = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
    for input_path in tqdm(wav_files, desc="Processing audio files"):
        relative_path = os.path.relpath(input_path, root_dir)
        output_dir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        detect_and_trim_silence(input_path, output_path)

# Use the function to process all audio files
if __name__ == "__main__":
    root_directory = "/workspace/tezuesh/omega-v2v/.predictions_warmup/moshi/audio/"
    output_directory = "/workspace/tezuesh/omega-v2v/.predictions_warmup/moshi/trimmed/"
    process_all_audio_files(root_directory, output_directory)

