from jiwer import wer
import librosa
import torch
import torchaudio.functional as F
from evaluation.S2S.rawnet.inference import RawNet3Inference
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from transformers import (
    AutoFeatureExtractor, 
    MimiModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import bittensor as bt
class S2SMetrics:
    def __init__(self):
        
        # Load the Whisper large model and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(self.device)
        self.mimi_model.config.num_quantizers = 4
        self.mimi_feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(self.device)
        
        self.nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
        self.wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        self.anti_spoofing_inference = RawNet3Inference(model_name = 'jungjee/RawNet3', device=self.device)
        
    
    def convert_audio(self, audio_arr, from_rate, to_rate, to_channels):
        try:
            audio = librosa.resample(audio_arr, orig_sr=from_rate, target_sr=to_rate)
            if to_channels == 1:
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=0, keepdims=True)
        except Exception as e:
            bt.logging.info(f"An error occurred while converting audio: {e}")
            return None
        return audio

    def transcribe_audio(self, audio_arr, sample_rate):
        """
        Transcribe the audio file using the Whisper large model and translate to English.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            str: Transcribed and translated text in English.
        """
        try:
            audio = librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=16000)
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            
            # Calculate the attention mask
            input_length = input_features.shape[-1]
            attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long)
            
            # Specify English as the target language
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
            
            predicted_ids = self.model.generate(
                input_features.to(self.device),
                attention_mask=attention_mask.to(self.device),
                forced_decoder_ids=forced_decoder_ids
            )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            return transcription[0]
        
        except Exception as e:
            bt.logging.info(f"An error occurred while transcribing the audio: {e}")
            return None

    

    def mimi_scoring(self, gt_audio_arr, generated_audio_arr, sample_rate_gt, sample_rate_generated):
        """
        Calculate the MIMI (Multi-channel Improved Metric for Intelligibility) score for a set of ground truth and generated audio paths.

        This method computes the MIMI score by comparing the tokenized representations of ground truth
        and generated audio samples using the Levenshtein distance.

        Args:
            gt_audio_paths (list): List of file paths to ground truth audio files.
            generated_audio_paths (list): List of file paths to generated audio files.

        Returns:
            float: The average MIMI score across all pairs of ground truth and generated audio samples.
                   Lower scores indicate higher similarity between the ground truth and generated audio.

        Note:
            - Audio samples shorter than 1000 frames receive a penalty instead of being skipped.
            - The score is calculated per channel and then averaged.
            - The final score is the average of all audio pair scores, including penalized ones.
        """
        
        gt_wav = self.convert_audio(gt_audio_arr, from_rate=sample_rate_gt, to_rate=24000, to_channels=1)
        generated_wav = self.convert_audio(generated_audio_arr, from_rate=sample_rate_generated, to_rate=24000, to_channels=1)

        if generated_wav is None:
            return 0
        if gt_wav is None:
            return 1

        # Squeeze the channel dimension since MIMI expects [batch, time] format
        gt_wav = torch.from_numpy(gt_wav)
        generated_wav = torch.from_numpy(generated_wav)
        
        metric = []
        if gt_wav.shape[0] < 1000 or generated_wav.shape[0] < 1000:
            # Apply a penalty instead of skipping
            penalty = 0  # You can adjust this value
            metric.append(penalty)
        else:
            gt_wav = self.mimi_feature_extractor(gt_wav, sampling_rate=24000, return_tensors="pt").to(self.device)
            generated_wav = self.mimi_feature_extractor(generated_wav, sampling_rate=24000, return_tensors="pt").to(self.device)

            gt_tokens = self.mimi_model.encode(gt_wav["input_values"], gt_wav["padding_mask"])['audio_codes']
            generated_tokens = self.mimi_model.encode(generated_wav["input_values"], generated_wav["padding_mask"])['audio_codes']
            
            num_channels = gt_tokens.shape[1]
            channel_losses = []
            for channel_idx in range(num_channels):
                gt_sample = gt_tokens[0, channel_idx, :]
                generated_sample = generated_tokens[0, channel_idx, :]
                edit_dist = F.edit_distance(gt_sample, generated_sample)
                channel_losses.append(1 - edit_dist/max(len(gt_sample), len(generated_sample)))
            metric.append(sum(channel_losses) / num_channels)
        return sum(metric) / len(metric)
    
    def compute_wer(self, gt_transcript, generated_transcript):
        """
        Compute the Word Error Rate (WER) between the ground truth transcript and the predicted transcript.

        This function calculates the WER between two given transcripts.

        Args:
            gt_transcript (str): The ground truth transcript.
            pred_transcript (str): The predicted transcript.

        Returns:
            float: The Word Error Rate between the ground truth and predicted transcripts.
        """
        if gt_transcript is None or generated_transcript is None:
            return 0
        if len(gt_transcript.split(" ")) < len(generated_transcript.split(" ")):
            gt_transcript, generated_transcript = generated_transcript, gt_transcript
        
        wer_score = wer(gt_transcript, generated_transcript)
        return 1 - wer_score


    def calculate_pesq(self, gt_audio_arr, generated_audio_arr, sample_rate_gt, sample_rate_generated):
        

        # Initialize PESQ metrics for narrowband and wideband
        
        """
        Calculate the Perceptual Evaluation of Speech Quality (PESQ) score using torchaudio.

        This function computes the PESQ score between the ground truth and generated audio files.
        PESQ is an objective measure for predicting the perceived quality of speech.

        Args:
            gt_audio_path (str): Path to the ground truth audio file.
            generated_audio_path (str): Path to the generated audio file.

        Returns:
            float: The calculated PESQ score. Higher scores indicate better quality.
                   Returns None if an error occurs.

        Note:
            This function requires torchaudio with PESQ support.
            Both audio files should have the same sampling rate (default is 16000 Hz).
        """
        try:
            # Convert both audio files to the required format (16kHz, mono)
            gt_audio = self.convert_audio(gt_audio_arr, from_rate=sample_rate_gt, to_rate=16000, to_channels=1)
            gen_audio = self.convert_audio(generated_audio_arr, from_rate=sample_rate_generated, to_rate=16000, to_channels=1)

            if gen_audio is None:
                return 0
            if gt_audio is None:
                return 1


            # Ensure both audio tensors have the same length by trimming to the shorter one
            min_length = min(gt_audio.shape[0], gen_audio.shape[0])
            gt_audio = gt_audio[:min_length]
            gen_audio = gen_audio[:min_length]
        
            # Calculate narrowband PESQ score
            # Convert numpy arrays to torch tensors
            gen_audio_tensor = torch.from_numpy(gen_audio).unsqueeze(0)
            gt_audio_tensor = torch.from_numpy(gt_audio).unsqueeze(0)
            # Calculate narrowband PESQ score
            nb_pesq_score = (self.nb_pesq(gen_audio_tensor, gt_audio_tensor).clip(-0.5, 4.5) + 0.5) / 5.0   # Normalize from [-0.5, 4.5] to [0, 1]

            # Calculate wideband PESQ score
            wb_pesq_score = (self.wb_pesq(gen_audio_tensor, gt_audio_tensor).clip(-0.5, 4.5) + 0.5) / 5.0   # Normalize from [-0.5, 4.5] to [0, 1]
            
            # Combine the scores 
            # Use appropriate weightage for narrowband and wideband PESQ scores
            # Wideband PESQ is generally considered more accurate for modern speech quality assessment
            combined_pesq_score = 0.3 * nb_pesq_score + 0.7 * wb_pesq_score
            
            return combined_pesq_score.detach().cpu().item()
        
        except Exception as e:
            bt.logging.info(f"An error occurred while calculating PESQ score: {e}")
            return 0
        
    def calculate_length_penalty(self, gt_audio_arr, generated_audio_arr, sample_rate_gt, sample_rate_generated):
        """
        Calculate the length penalty between ground truth and generated audio.

        This function computes a penalty based on the difference in duration
        between the ground truth and generated audio files. The penalty
        increases as the difference in duration increases.

        Args:
            gt_audio_arr (numpy.ndarray): Ground truth audio array
            generated_audio_arr (numpy.ndarray): Generated audio array 
            sample_rate_gt (int): Sample rate of ground truth audio
            sample_rate_generated (int): Sample rate of generated audio

        Returns:
            float: The calculated length penalty. A value of 1 indicates
                  perfect length match, with lower values for mismatched lengths.
        """
        try:
            gt_audio = self.convert_audio(gt_audio_arr, from_rate=sample_rate_gt, to_rate=16000, to_channels=1)
            generated_audio = self.convert_audio(generated_audio_arr, from_rate=sample_rate_generated, to_rate=16000, to_channels=1)

            if generated_audio is None:
                return 0
            if gt_audio is None:
                return 1

            gt_duration = gt_audio.shape[0] / 16000  # Assuming 16kHz sample rate
            generated_duration = generated_audio.shape[0] / 16000
            
            duration_diff = abs(gt_duration - generated_duration)/max(gt_duration, generated_duration)
            
            # Calculate penalty: 1 for perfect match, decreasing as difference increases
            penalty =  1 / (1 + duration_diff)
            
            return penalty
        
        except Exception as e:
            bt.logging.info(f"An error occurred while calculating length penalty: {e}")
            return 0  # Return 0 as the maximum penalty in case of error


    def calculate_anti_spoofing_score(self, generated_audio_arr, gt_audio_arr, gt_sample_rate, generated_sample_rate):
        try:
            generated_embedding = self.anti_spoofing_inference.extract_speaker_embd(generated_audio_arr,generated_sample_rate, 48000, 10)
            gt_embedding = self.anti_spoofing_inference.extract_speaker_embd(gt_audio_arr, gt_sample_rate, 48000, 10)
            return 1/(1+((generated_embedding - gt_embedding) ** 2).mean()).detach().cpu().item()
        except Exception as e:
            bt.logging.info(f"An error occurred while calculating anti-spoofing score: {e}")
            return 0
        
    def compute_distance(self, gt_audio_arrs, generated_audio_arrs):
        gt_transcripts = [self.transcribe_audio(gt_audio_arr[0], gt_audio_arr[1]) for gt_audio_arr in gt_audio_arrs]
        generated_transcripts = [self.transcribe_audio(generated_audio_arr[0], generated_audio_arr[1]) for generated_audio_arr in generated_audio_arrs]
   
        mimi_score = [self.mimi_scoring(gt_audio_arr[0], generated_audio_arr[0], gt_audio_arr[1], generated_audio_arr[1]) for gt_audio_arr, generated_audio_arr in zip(gt_audio_arrs, generated_audio_arrs)]
        wer_score = [self.compute_wer(gt_transcript, generated_transcript) for gt_transcript, generated_transcript in zip(gt_transcripts, generated_transcripts)]
        length_penalty = [self.calculate_length_penalty(gt_audio_arr[0], generated_audio_arr[0], gt_audio_arr[1], generated_audio_arr[1]) for gt_audio_arr, generated_audio_arr in zip(gt_audio_arrs, generated_audio_arrs)]    
        pesq_score = [self.calculate_pesq(gt_audio_arr[0], generated_audio_arr[0], gt_audio_arr[1], generated_audio_arr[1] ) for gt_audio_arr, generated_audio_arr in zip(gt_audio_arrs, generated_audio_arrs)]
        anti_spoofing_score = [self.calculate_anti_spoofing_score(generated_audio_arr[0], gt_audio_arr[0], generated_audio_arr[1], gt_audio_arr[1]) for generated_audio_arr, gt_audio_arr in zip(generated_audio_arrs, gt_audio_arrs)]
        # Calculate combined metric using geometric and arithmetic means
        combined_scores = []
        for m, w, l, p, a in zip(mimi_score, wer_score, length_penalty, pesq_score, anti_spoofing_score):
            # Calculate geometric mean for pesq and anti-spoofing scores
            try:
                geometric_mean = (p * a) ** (1/2)
                # Calculate arithmetic mean for mimi and inverted WER scores
                arithmetic_mean = (m + w) / 2
                # Combine geometric and arithmetic means, then apply length penalty
                combined = (geometric_mean * 0.6 + arithmetic_mean * 0.4) * l
                combined_scores.append(combined)
            except Exception as e:
                bt.logging.info(f"An error occurred while calculating combined score: {e}")
                combined_scores.append(0)
        
        return {
            'mimi_score': mimi_score,
            'wer_score': wer_score,
            'length_penalty': length_penalty,
            'pesq_score': pesq_score,
            'anti_spoofing_score': anti_spoofing_score,
            'combined_score': combined_scores
        }

 
 
if __name__ == "__main__":      
    # Example usage
    import librosa

    # Load audio files first
    gt_audio_arrs = []
    generated_audio_arrs = []
    
    gt_paths = [
        '/workspace/tezuesh/omega-v2v/.filtered/Fresh_Air_Remembering_Gospel_Singer_Cissy_Houston_MLB_Legend_Pete_Rose_sample/0000049013.wav',
        '/workspace/tezuesh/omega-v2v/.filtered/Fresh_Air_Remembering_Gospel_Singer_Cissy_Houston_MLB_Legend_Pete_Rose_sample/0000049013.wav'
    ]
    generated_paths = [
        '/workspace/tezuesh/omega-v2v/.filtered/Fresh_Air_Remembering_Gospel_Singer_Cissy_Houston_MLB_Legend_Pete_Rose_sample/0000049013.wav',
        '/workspace/tezuesh/omega-v2v/.filtered/Fresh_Air_Remembering_Gospel_Singer_Cissy_Houston_MLB_Legend_Pete_Rose_sample/0000223993.wav'
    ]

    # Load each audio file and store as (audio_array, sample_rate) tuples
    for gt_path, gen_path in zip(gt_paths, generated_paths):
        gt_audio, gt_sr = librosa.load(gt_path, sr=None)
        gen_audio, gen_sr = librosa.load(gen_path, sr=None)
        duration = librosa.get_duration(y=gt_audio, sr=gt_sr)
        print(f"Duration of {gt_path}: {duration} seconds")
        duration = librosa.get_duration(y=gen_audio, sr=gen_sr)
        print(f"Duration of {gen_path}: {duration} seconds")
        
        gt_audio_arrs.append((gt_audio, gt_sr))
        generated_audio_arrs.append((gen_audio, gen_sr))

    metric = S2SMetrics()
    print(metric.compute_distance(gt_audio_arrs, generated_audio_arrs))
