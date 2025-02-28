import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
import torch
import torchaudio.functional as F
from jiwer import wer
import librosa
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    MimiModel,
    AutoFeatureExtractor
)
from evaluation.S2S.rawnet.inference import RawNet3Inference
import bittensor as bt
import constants
import os

class S2SMetrics:
    """Speech-to-Speech evaluation metrics"""
    
    def __init__(self, cache_dir: str = ".checkpoints/"):
        """Initialize metrics with configuration"""
        # Setup cache directories
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MIMI model
        bt.logging.info("Loading MIMI model...")
        self.mimi = MimiModel.from_pretrained(
            "kyutai/mimi",
            cache_dir=self.cache_dir
        ).to(self.device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "kyutai/mimi",
            cache_dir=self.cache_dir
        )

        # Load RawNet3 (anti-spoofing)
        bt.logging.info("Loading RawNet3...")
        self.anti_spoofing_inference = RawNet3Inference(model_name = 'jungjee/RawNet3',repo_dir = self.cache_dir, device=self.device)
        

        # Load Whisper
        bt.logging.info("Loading Whisper...")
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v2", 
            cache_dir=self.cache_dir
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2", 
            cache_dir=self.cache_dir
        ).to(self.device).eval()
        
        # Initialize PESQ
        self.nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
        self.wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        
        

    def convert_audio(self, 
                     audio_arr: np.ndarray, 
                     from_rate: int, 
                     to_rate: int, 
                     to_channels: int = 1) -> Optional[np.ndarray]:
        """Convert audio array to target sample rate and channels"""
        try:
            audio = librosa.resample(audio_arr, orig_sr=from_rate, target_sr=to_rate)
            if to_channels == 1 and len(audio.shape) > 1:
                audio = audio.mean(axis=0, keepdims=True)
            return audio
        except Exception as e:
            bt.logging.error(f"Audio conversion error: {e}")
            return None

    def mimi_scoring(self, 
                    gt_audio_arr: np.ndarray,
                    generated_audio_arr: np.ndarray, 
                    sample_rate_gt: int,
                    sample_rate_generated: int) -> float:
        """Calculate MIMI-based similarity score between ground truth and generated audio"""
        try:
            # Convert to MIMI sample rate
            gt_audio = librosa.resample(
                gt_audio_arr, 
                orig_sr=sample_rate_gt, 
                target_sr=self.feature_extractor.sampling_rate
            )
            generated_audio = librosa.resample(
                generated_audio_arr, 
                orig_sr=sample_rate_generated, 
                target_sr=self.feature_extractor.sampling_rate
            )

            # Match lengths by truncating to shorter sequence
            min_len = min(len(gt_audio), len(generated_audio))
            gt_audio = gt_audio[:min_len]
            generated_audio = generated_audio[:min_len]

            # Prepare inputs
            gt_inputs = self.feature_extractor(
                raw_audio=gt_audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt"
            ).to(self.device)
            
            generated_inputs = self.feature_extractor(
                raw_audio=generated_audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt"
            ).to(self.device)

            # Get encodings
            with torch.no_grad():
                gt_encoding = self.mimi.encode(
                    gt_inputs["input_values"], 
                    gt_inputs["padding_mask"]
                )
                generated_encoding = self.mimi.encode(
                    generated_inputs["input_values"],
                    generated_inputs["padding_mask"]
                )

                # Compare codes
                gt_codes = gt_encoding.audio_codes
                generated_codes = generated_encoding.audio_codes
                
                # Calculate scores per codebook
                scores = []
                for i in range(gt_codes.shape[1]):
                    gt_seq = gt_codes[:, i][0]
                    gen_seq = generated_codes[:, i][0]
                    
                    # Calculate edit distance
                    edit_dist = F.edit_distance(gt_seq, gen_seq)
                    max_len = max(len(gt_seq), len(gen_seq))
                    score = 1 - (edit_dist / max_len if max_len > 0 else 0)
                    scores.append(score)
                
                return float(sum(scores) / len(scores))

        except Exception as e:
            bt.logging.error(f"MIMI scoring error: {e}")
            return 0.0

    def transcribe_audio(self, 
                        audio_arr: np.ndarray, 
                        sample_rate: int) -> Optional[str]:
        """Transcribe audio using Whisper"""
        try:
            # Resample to 16kHz for Whisper
            audio = librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=16000)
            
            # Prepare input features
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="en", 
                task="transcribe"
            )
            
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
            
            return self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
        except Exception as e:
            bt.logging.error(f"Transcription error: {e}")
            return None

    def compute_wer(self,
                   gt_transcript: Optional[str],
                   generated_transcript: Optional[str]) -> float:
        """Compute Word Error Rate between transcripts"""
        if not gt_transcript or not generated_transcript:
            return 0.0
            
        try:
            if len(gt_transcript.split(" ")) < len(generated_transcript.split(" ")):
                gt_transcript, generated_transcript = generated_transcript, gt_transcript
            return 1 - wer(gt_transcript, generated_transcript)
        except Exception as e:
            bt.logging.error(f"WER computation error: {e}")
            return 0.0

    def calculate_pesq(self,
                      gt_audio_arr: np.ndarray,
                      generated_audio_arr: np.ndarray, 
                      sample_rate_gt: int,
                      sample_rate_generated: int) -> float:
        """Calculate PESQ score"""
        try:
            # Check for invalid audio data
            if (np.isnan(gt_audio_arr).any() or 
                np.isnan(generated_audio_arr).any() or
                np.all(generated_audio_arr == 0) or
                np.all(gt_audio_arr == 0)):
                bt.logging.warning("Invalid audio data detected (NaN or all zeros)")
                return 0.0

            # Convert both to 16kHz mono
            gt_audio = self.convert_audio(gt_audio_arr, sample_rate_gt, 16000, 1)
            gen_audio = self.convert_audio(generated_audio_arr, sample_rate_generated, 16000, 1)

            if gen_audio is None or gt_audio is None:
                return 0.0

            # Normalize audio
            gt_audio = gt_audio / (np.max(np.abs(gt_audio)) + 1e-6)
            gen_audio = gen_audio / (np.max(np.abs(gen_audio)) + 1e-6)

            # Match lengths
            min_length = min(gt_audio.shape[-1], gen_audio.shape[-1])
            gt_audio = gt_audio[:min_length]
            gen_audio = gen_audio[:min_length]

            # Ensure minimum length for PESQ (at least 32ms at 16kHz = 512 samples)
            if min_length < 512:
                bt.logging.warning("Audio too short for PESQ calculation")
                return 0.0

            # Convert to tensors
            gt_tensor = torch.from_numpy(gt_audio).float().unsqueeze(0)
            gen_tensor = torch.from_numpy(gen_audio).float().unsqueeze(0)

            # Calculate scores
            nb_score = (self.nb_pesq(gen_tensor, gt_tensor).clip(-0.5, 4.5) + 0.5) / 5.0
            wb_score = (self.wb_pesq(gen_tensor, gt_tensor).clip(-0.5, 4.5) + 0.5) / 5.0
            
            # Weight narrow-band and wide-band scores
            return (0.3 * nb_score + 0.7 * wb_score).detach().cpu().item()
            
        except Exception as e:
            bt.logging.error(f"PESQ calculation error: {e}")
            return 0.0

    def calculate_anti_spoofing_score(self, generated_audio_arr, gt_audio_arr, gt_sample_rate, generated_sample_rate):
        try:
            generated_embedding = self.anti_spoofing_inference.extract_speaker_embd(generated_audio_arr, generated_sample_rate, 48000, 10)
            gt_embedding = self.anti_spoofing_inference.extract_speaker_embd(gt_audio_arr, gt_sample_rate, 48000, 10)
            return 1/(1+((generated_embedding - gt_embedding) ** 2).mean()).detach().cpu().item()
        except Exception as e:
            bt.logging.info(f"An error occurred while calculating anti-spoofing score: {e}")
            return 0
        
    def compute_distance(self, 
                        gt_audio_arrs: List[Tuple[np.ndarray, int]],
                        generated_audio_arrs: List[Tuple[np.ndarray, int]]) -> Dict[str, List[float]]:
        """Compute all metrics between ground truth and generated audio"""
        
        try:
            # Calculate all scores
            mimi_scores = []
            wer_scores = []
            pesq_scores = []
            anti_spoofing_scores = []
            
            for (gt_arr, gt_rate), (gen_arr, gen_rate) in zip(
                gt_audio_arrs, generated_audio_arrs
            ):
                bt.logging.info(f"Calculating metrics for {gt_arr} and {gen_arr}")  
                # MIMI score
                mimi_scores.append(
                    self.mimi_scoring(gt_arr, gen_arr, gt_rate, gen_rate)
                )
                
                # Transcription & WER
                gt_transcript = self.transcribe_audio(gt_arr, gt_rate)
                gen_transcript = self.transcribe_audio(gen_arr, gen_rate)
                bt.logging.info(f"Transcription: {gt_transcript} and {gen_transcript}")
                wer_scores.append(
                    self.compute_wer(gt_transcript, gen_transcript)
                )
                bt.logging.info(f"WER: {wer_scores}")   
                # PESQ
                pesq_scores.append(
                    self.calculate_pesq(gt_arr, gen_arr, gt_rate, gen_rate)
                )
                bt.logging.info(f"PESQ: {pesq_scores}")
                # Anti-spoofing
                anti_spoofing_scores.append(
                    self.calculate_anti_spoofing_score(
                        gen_arr, gt_arr, gen_rate, gt_rate
                    )
                )
                bt.logging.info(f"Anti-spoofing: {anti_spoofing_scores}")
            
            # Calculate combined scores
            length_penalty = [1.0] * len(mimi_scores)  # Simplified
            combined_scores = []
            
            for scores in zip(
                mimi_scores, 
                wer_scores, 
                length_penalty,
                pesq_scores,
                anti_spoofing_scores
            ):
                PESQ_COEFFICIENT = 0.2
                ANTI_SPOOFING_COEFFICIENT = 0.45
                MIMI_COEFFICIENT = 0.17
                WER_COEFFICIENT = 0.18
                try:
                    m, w, l, p, a = scores
                    combined = (
                        p * PESQ_COEFFICIENT +
                        a * ANTI_SPOOFING_COEFFICIENT +
                        m * MIMI_COEFFICIENT +
                        w * WER_COEFFICIENT
                    ) * l
                    combined_scores.append(combined)
                except Exception as e:
                    bt.logging.error(f"Combined score calculation error: {e}")
                    combined_scores.append(0.0)
            
            # Get compute efficiency score
            
            return {
                'mimi_score': mimi_scores,
                'wer_score': wer_scores,
                'length_penalty': length_penalty,
                'pesq_score': pesq_scores,
                'anti_spoofing_score': anti_spoofing_scores,
                'combined_score': combined_scores,
                # 'compute_score': compute_score
            }
            
        except Exception as e:
            bt.logging.error(f"Overall metric computation error: {e}")
            return {
                'mimi_score': [constants.penalty_score],
                'wer_score': [constants.penalty_score],
                'length_penalty': [constants.penalty_score],
                'pesq_score': [constants.penalty_score],
                'anti_spoofing_score': [constants.penalty_score],
                'combined_score': [constants.penalty_score],
                'compute_score': [constants.penalty_score]
            }

    def load_rawnet3(self, model_path: str, device: str = "cuda") -> RawNet3Inference:
        try:
            rawnet = RawNet3Inference(model_path, device)
            return rawnet
        except Exception as e:
            bt.logging.error(f"Error loading RawNet3: {str(e)}")
            raise