import torch
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download
import sentencepiece
from pydub import AudioSegment
from models.S2S.moshi.moshi.models import loaders, LMGen
from models.S2S.moshi.chunk_silence import detect_and_trim_silence
import os
from tqdm import tqdm
import glob
import random

# Initialize models and tokenizer


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

seed_all(42424242)

def initialize_models(hf_repo_id: str, repo_dir: str, device='cpu'):
    mimi_weight_path = hf_hub_download(repo_id=hf_repo_id, filename=loaders.MIMI_NAME, local_dir=repo_dir)
    mimi = loaders.get_mimi(mimi_weight_path, device=device)
    mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi

    text_tokenizer_path = hf_hub_download(repo_id=hf_repo_id, filename=loaders.TEXT_TOKENIZER_NAME, local_dir=repo_dir)
    text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer_path)

    moshi_weight_path = hf_hub_download(repo_id=hf_repo_id, filename=loaders.MOSHI_NAME, local_dir=repo_dir)
    moshi = loaders.get_moshi_lm(moshi_weight_path, device=device)
    lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)

    return {"models":[mimi, text_tokenizer, lm_gen], "paths":[mimi_weight_path, moshi_weight_path, text_tokenizer_path]}

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

def load_audio_from_array(audio_array: np.array, sample_rate: int, mimi):
    wav = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
    wav = torchaudio.transforms.Resample(sample_rate, mimi.sample_rate)(wav)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    wav = wav[:, :, :(wav.shape[-1] // frame_size) * frame_size]

    return wav

def pad_audio_codes(all_codes, mimi, device, time_seconds=30):
    min_frames = int(time_seconds * mimi.frame_rate)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    if len(all_codes) < min_frames:
        # Calculate how many more frames we need
        frames_to_add = min_frames - len(all_codes)
        
        # Generate additional codes using zero frames
        with torch.no_grad(), mimi.streaming(batch_size=1):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            for _ in range(frames_to_add):
                additional_code = mimi.encode(chunk)
                all_codes.append(additional_code)
    
    # Concatenate all codes into a single tensor
    # all_codes = torch.cat(all_codes, dim=2).to(device)
    return all_codes

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

def warmup(mimi, lm_gen, device):
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    for _ in range(1):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c:c + 1])
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:])
    torch.cuda.synchronize()

def generate_audio(mimi, lm_gen, text_tokenizer, all_codes, device):
    out_wav_chunks = []
    text_output = []
    with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
        for idx, code in enumerate(all_codes):
            # Assert the shape of the code
            assert code.shape == (1, 8, 1), f"Expected code shape (1, 8, 1), but got {code.shape}"
            tokens_out = lm_gen.step(code.to(device))
            if tokens_out is not None:
                wav_chunk = mimi.decode(tokens_out[:, 1:])
                out_wav_chunks.append(wav_chunk)
                text_token = tokens_out[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    text_output.append(_text)
    

    return torch.cat(out_wav_chunks, dim=-1), ''.join(text_output)

def save_audio(out_wav, sample_rate, output_path):
    out_wav_cpu = out_wav.cpu().squeeze(0).float()
    torchaudio.save(output_path, out_wav_cpu, sample_rate)
    
def process_folder(input_folder, audio_output_folder, text_output_folder, mimi, lm_gen, text_tokenizer, device):
    wav_files = glob.glob(os.path.join(input_folder, '**', '*.wav'), recursive=True)
    wav_files.sort()
    error_count = 0
    for input_path in tqdm(wav_files, desc="Processing files"):
        relative_path = os.path.relpath(input_path, input_folder)
        audio_output_path = os.path.join(audio_output_folder, relative_path)
        text_output_path = os.path.join(text_output_folder, relative_path.rsplit('.', 1)[0] + '.txt')

        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

        # Process the audio file
        try:
            wav = load_audio(input_path, mimi)
            all_codes = encode_audio(mimi, wav, device)
            all_codes = pad_audio_codes(all_codes, mimi, device)
            warmup(mimi, lm_gen, device)
            out_wav, text_output = generate_audio(mimi, lm_gen, text_tokenizer, all_codes, device)
        except Exception as e:
            print(f"Error processing file {input_path}: {str(e)}")
            error_count += 1
            continue
        
        # Save audio output
        save_audio(out_wav, mimi.sample_rate, audio_output_path)
        
        
        # Save text output
        with open(text_output_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_output)
        
    print(f"Error count: {error_count}/{len(wav_files)}")


class InferenceRecipe:
    def __init__(self, hf_repo_id: str, repo_dir: str, device: str='cuda'):
        self.hf_repo_id = hf_repo_id
        self.device = torch.device(device)
        mod_paths = initialize_models(hf_repo_id, repo_dir, self.device)
        self.mimi, self.text_tokenizer, self.lm_gen = mod_paths["models"]
        self.model_paths = mod_paths["paths"]
        
    def inference(self, audio_array: np.array, sample_rate: int):
        wav = load_audio_from_array(audio_array, sample_rate, self.mimi).to(self.device)
        all_codes = encode_audio(self.mimi, wav, self.device)
        all_codes = pad_audio_codes(all_codes, self.mimi, self.device)
        warmup(self.mimi, self.lm_gen, self.device)
        out_wav, text_output = generate_audio(self.mimi, self.lm_gen, self.text_tokenizer, all_codes, self.device)
        trimmed_wav = detect_and_trim_silence(out_wav, self.mimi.sample_rate, min_silence_duration=1000, silence_threshold=-40)
        return {"audio": trimmed_wav, "text": text_output}




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_repo_id = "kyutai/moshiko-pytorch-bf16"
    import tempfile
    repo_dir = tempfile.mkdtemp()
    mimi, text_tokenizer, lm_gen = initialize_models(hf_repo_id, repo_dir, device)

    # Set up input and output folders
    input_folder = '/workspace/tezuesh/omega-v2v/.extracted/'
    audio_output_folder = '/workspace/tezuesh/omega-v2v/.predictions_warmup/moshi/audio/'
    text_output_folder = '/workspace/tezuesh/omega-v2v/.predictions_warmup/moshi/text/'

    # Process all subfolders
    process_folder(input_folder, audio_output_folder, text_output_folder, mimi, lm_gen, text_tokenizer, device)


if __name__ == "__main__":
    main()
