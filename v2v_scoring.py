import os
from datasets import load_dataset
import huggingface_hub
from tempfile import TemporaryDirectory
import time
from typing import Optional
from datasets import Dataset
import ulid
import pandas as pd
import tempfile
from tqdm import tqdm
from models.S2S import inference as s2s_inference
import numpy as np

HF_DATASET = "tezuesh/diarization_dataset"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 48 * 60 * 60  # 48 hours
MAX_FILES = 1


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp


def pull_latest_diarization_dataset() -> Optional[Dataset]:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX)
    ][:MAX_FILES]

    print(recent_files)
    if len(recent_files) == 0:
        return None
    temp_dir = "./data_cache"
    # with TemporaryDirectory(dir='./data_cache') as temp_dir:
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir)["train"]
    omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
    return omega_dataset


def inference(model_id: str, audio_array: np.array, sample_rate: int):
    return s2s_inference(model_id, audio_array, sample_rate)


dataset = pull_latest_diarization_dataset()
print(dataset.keys())
# Initialize an empty list to store the data
data = []

# Create a temporary directory to store output files
temp_dir = tempfile.mkdtemp()

for i in tqdm(range(len(dataset))):
  

    youtube_id = dataset['youtube_id'][i]
    audio_array = np.array(dataset['audio_array'][i])
    # Create temporary files for output audio and text
    audio_output = os.path.join(temp_dir, f"{youtube_id}_output.wav")
    text_output = os.path.join(temp_dir, f"{youtube_id}_output.txt")
    diar_timestamps_start = np.array(dataset['diar_timestamps_start'][i])
    diar_timestamps_end = np.array(dataset['diar_timestamps_end'][i])
    diar_speakers = np.array(dataset['diar_speakers'][i])

    test_idx = 0
    diar_sample = audio_array[int(diar_timestamps_start[test_idx] * dataset['sample_rate'][i]):int(diar_timestamps_end[test_idx] * dataset['sample_rate'][i])]
    speaker = diar_speakers[test_idx]
    print(diar_sample.shape, audio_array.shape)
    # exit()


    
    # Perform inference
    result = inference(model_id="moshi", audio_array=diar_sample, sample_rate=dataset['sample_rate'][i])
    
    # Save the text output
    with open(text_output, 'w') as f:
        f.write(result['text'])
    
    # Append the data to our list
    data.append({
        'youtube_id': youtube_id,
        'output_audio': audio_output,
        'output_text': text_output
    })

# Create the DataFrame
df = pd.DataFrame(data)

# Print the first few rows of the DataFrame
print(df.head())


# # Load the dataset
# dataset = load_dataset(HF_DATASET)

# print(f"Dataset loaded successfully with {len(dataset['train'])} examples")
# first_row = dataset['train'][0]
# print("\nFirst row of dataset:")
# # print(first_row)
# print("\nKeys in dataset:")
# print("\nLength of values in first row:")
# for key in first_row.keys():
#     if isinstance(first_row[key], list):
#         print(f"{key}: {len(first_row[key])}")
#     else:
#         print(f"{key}: {first_row[key]}")



# import librosa
# import numpy as np
# audio_arr = first_row['audio_array']
# print(len(audio_arr), type(audio_arr))
# sr = first_row['sample_rate']
# audio = np.array(audio_arr)
# # exit()
# print(audio.shape)
# import soundfile as sf
# youtube_id = first_row['youtube_id']
# os.makedirs('Dataset_audios/Original', exist_ok=True)
# sf.write(f'Dataset_audios/Original/{youtube_id}.wav', audio, sr)

# diar_timestamps_start = first_row['diar_timestamps_start']
# diar_timestamps_end = first_row['diar_timestamps_end']
# diar_speakers = first_row['diar_speakers']

# for start, end, speaker in zip(diar_timestamps_start, diar_timestamps_end, diar_speakers):
#     # Calculate start and end samples
#     start_sample = int(start * sr)
#     end_sample = int(end * sr)
    
#     # Extract the clip
#     clip = audio[start_sample:end_sample]
    
#     # Create output directory if it doesn't exist
#     os.makedirs(f'Dataset_audios/Clips/{youtube_id}', exist_ok=True)
    
#     # Save the clip with speaker and timestamp info in filename
#     clip_filename = f'Dataset_audios/Clips/{youtube_id}/{speaker}_{start:08.2f}-{end:08.2f}.wav'
#     sf.write(clip_filename, clip, sr)
    

# # Create a list to store the diarization data
# diarization_data = []
# for start, end, speaker in zip(diar_timestamps_start, diar_timestamps_end, diar_speakers):
#     diarization_data.append({
#         'youtube_id': youtube_id,
#         'start_time': start,
#         'end_time': end, 
#         'speaker': speaker,
#         "duration": end - start
#     })

# # Convert to pandas DataFrame and save as CSV
# import pandas as pd
# df = pd.DataFrame(diarization_data)
# os.makedirs('Dataset_audios/Metadata', exist_ok=True)
# df.to_csv(f'Dataset_audios/Metadata/{youtube_id}_diarization.csv', index=False)