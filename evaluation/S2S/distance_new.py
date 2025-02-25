import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
import torch
import torchaudio.functional as F
import librosa
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    HubertModel,
    Wav2Vec2FeatureExtractor  # Updated - HuBERT uses Wav2Vec2FeatureExtractor
)
from evaluation.S2S.rawnet.inference import RawNet3Inference
import bittensor as bt
import constants
import os
from sklearn.metrics.pairwise import cosine_similarity

class MOSNet:
    """
    Simple implementation of a MOSNet-like feature extractor for speech naturalness assessment.
    This replaces ITU-T P.563 with a lightweight approach based on spectral features.
    """
    def __init__(self):
        self.sr = 16000  # Expected sample rate
    
    def score(self, audio_arr, sample_rate):
        """
        Calculate a naturalness score based on spectral features.
        Returns a score between 1 (poor) and 5 (excellent)
        """
        # Resample if needed
        if sample_rate != self.sr:
            audio_arr = librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=self.sr)
        
        # Normalize
        audio_arr = audio_arr / (np.max(np.abs(audio_arr)) + 1e-6)
        
        # Extract features that correlate with naturalness
        
        # 1. Spectral centroid (brightness, clarity)
        centroid = librosa.feature.spectral_centroid(y=audio_arr, sr=self.sr)[0]
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        
        # 2. Spectral contrast (speech formant structure)
        contrast = librosa.feature.spectral_contrast(y=audio_arr, sr=self.sr)
        contrast_mean = np.mean(contrast)
        
        # 3. Zero crossing rate (harshness/smoothness)
        zcr = librosa.feature.zero_crossing_rate(audio_arr)[0]
        zcr_mean = np.mean(zcr)
        
        # 4. RMS energy variations (prosody)
        rms = librosa.feature.rms(y=audio_arr)[0]
        rms_std = np.std(rms)
        
        # 5. MFCC for timbre characteristics
        mfccs = librosa.feature.mfcc(y=audio_arr, sr=self.sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Compute a weighted score (these weights are approximate and should be tuned)
        # Higher centroid → brighter speech → typically more natural up to a point
        centroid_score = 3.0 + min(1.0, max(-1.0, (centroid_mean - 3000) / 2000))
        
        # More spectral contrast → better formant structure → more natural
        contrast_score = 3.0 + min(1.0, max(-1.0, (contrast_mean - 15) / 10))
        
        # Moderate ZCR → not too harsh, not too muffled
        zcr_score = 5.0 - min(2.0, abs(zcr_mean - 0.05) * 50)
        
        # RMS variations → better prosody → more natural
        rms_score = 3.0 + min(1.0, rms_std * 10)
        
        # MFCC based score (simplified)
        mfcc_score = 3.0 + np.tanh(np.mean(mfccs_std[1:]) - 1.5)
        
        # Final score (weighted average)
        final_score = 0.25 * centroid_score + 0.25 * contrast_score + 0.2 * zcr_score + 0.15 * rms_score + 0.15 * mfcc_score
        
        # Clip to 1-5 range
        return min(5.0, max(1.0, final_score))

class HuBertEmbedder:
    def __init__(self, cache_dir=".checkpoints/"):
        # HuBERT uses the Wav2Vec2FeatureExtractor in newer transformers versions
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", cache_dir=cache_dir)
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960", cache_dir=cache_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embedding(self, audio_arr, sample_rate):
        # Check if audio array is valid
        if audio_arr is None or len(audio_arr) == 0:
            bt.logging.warning("Empty audio array provided to HuBertEmbedder")
            return None
            
        # Convert to float32 and make sure it's the right shape
        if isinstance(audio_arr, np.ndarray):
            if len(audio_arr.shape) > 1 and audio_arr.shape[0] > 1:
                # Convert stereo to mono by averaging channels
                audio_arr = np.mean(audio_arr, axis=0)
            
            # Ensure it's a 1D array
            audio_arr = audio_arr.squeeze()
        
        # Check if audio is too short
        if len(audio_arr) < 1000:  # Arbitrary minimum length
            bt.logging.warning("Audio too short for HuBert embedding")
            return None
            
        inputs = self.feature_extractor(
            audio_arr, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get last hidden states
            embeddings = outputs.last_hidden_state
            
            # Check if embeddings are valid
            if embeddings is None or embeddings.numel() == 0:
                bt.logging.warning("Empty embeddings from HuBert model")
                return None
                
            # Average over time dimension
            mean_embedding = torch.mean(embeddings, dim=1)
            
            # Move back to CPU for numpy conversion
            return mean_embedding.cpu().numpy().squeeze()

class S2SMetrics:
    """Speech-to-Speech evaluation metrics"""
    
    def __init__(self, cache_dir: str = ".checkpoints/"):
        """Initialize metrics with configuration"""
        # Setup cache directories
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load RawNet3 (anti-spoofing)
        bt.logging.info("Loading RawNet3...")
        self.anti_spoofing_inference = RawNet3Inference(model_name='jungjee/RawNet3', repo_dir=self.cache_dir, device=self.device)

        # Load Whisper (for potential future use, though WER is replaced)
        bt.logging.info("Loading Whisper...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=self.cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", cache_dir=self.cache_dir).to(self.device).eval()
        
        # Initialize PESQ
        self.nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
        self.wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        
        # Initialize HuBert for semantic similarity
        bt.logging.info("Loading HuBert...")
        self.hubert = HuBertEmbedder(cache_dir=self.cache_dir)
        
        # Initialize MOSNet for naturalness (replacement for P.563)
        self.mosnet = MOSNet()
        
        bt.logging.info("S2SMetrics initialization complete")

    def convert_audio(self, audio_arr: np.ndarray, from_rate: int, to_rate: int, to_channels: int = 1) -> Optional[np.ndarray]:
        """Convert audio array to target sample rate and channels"""
        if audio_arr is None:
            bt.logging.warning("None audio array provided to convert_audio")
            return None
            
        if len(audio_arr) == 0:
            bt.logging.warning("Empty audio array provided to convert_audio")
            return None
            
        # Check for NaN values
        if np.isnan(audio_arr).any():
            bt.logging.warning("NaN values detected in audio for conversion")
            return None
            
        # Check for valid sample rates
        if from_rate <= 0 or to_rate <= 0:
            bt.logging.warning(f"Invalid sample rates: from_rate={from_rate}, to_rate={to_rate}")
            return None
            
        # Handle multi-channel audio
        if len(audio_arr.shape) > 1 and audio_arr.shape[0] > 1 and to_channels == 1:
            audio_arr = np.mean(audio_arr, axis=0)
        
        # Resample if needed
        if from_rate != to_rate:
            audio = librosa.resample(audio_arr, orig_sr=from_rate, target_sr=to_rate)
        else:
            audio = audio_arr.copy()
            
        # Final check for valid audio
        if audio is None or len(audio) == 0 or np.isnan(audio).any():
            bt.logging.warning("Invalid audio after conversion")
            return None
            
        return audio

    def calculate_semantic_similarity(self, gt_audio_arr, generated_audio_arr, gt_sample_rate, generated_sample_rate) -> float:
        """Calculate HuBert-based semantic similarity score between ground truth and generated audio"""
        try:
            # Resample to HuBert's required sample rate (16000 Hz)
            gt_audio = self.convert_audio(gt_audio_arr, gt_sample_rate, 16000, 1)
            generated_audio = self.convert_audio(generated_audio_arr, generated_sample_rate, 16000, 1)
            
            if gt_audio is None or generated_audio is None:
                bt.logging.warning("Failed to convert audio for semantic similarity calculation")
                return constants.penalty_score
            
            # Check if audio arrays are empty or too short
            if len(gt_audio) < 1000 or len(generated_audio) < 1000:
                bt.logging.warning("Audio too short for semantic similarity calculation")
                return constants.penalty_score
            
            # Get embeddings
            gt_embedding = self.hubert.get_embedding(gt_audio, 16000)
            gen_embedding = self.hubert.get_embedding(generated_audio, 16000)
            
            # Check if embeddings are valid
            if gt_embedding is None or gen_embedding is None or len(gt_embedding) == 0 or len(gen_embedding) == 0:
                bt.logging.warning("Invalid embeddings for semantic similarity calculation")
                return constants.penalty_score
            
            # Compute cosine similarity
            similarity = cosine_similarity([gt_embedding], [gen_embedding])
            
            # Convert to float and ensure it's in [0,1]
            score = float(similarity[0][0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            bt.logging.error(f"Semantic similarity calculation error: {e}")
            return constants.penalty_score

    def calculate_pesq(self, gt_audio_arr: np.ndarray, generated_audio_arr: np.ndarray, gt_sample_rate: int, generated_sample_rate: int) -> float:
        """Calculate PESQ score"""
        try:
            # Check for empty or invalid arrays
            if (gt_audio_arr is None or generated_audio_arr is None or 
                len(gt_audio_arr) == 0 or len(generated_audio_arr) == 0):
                bt.logging.warning("Empty audio array provided to PESQ calculation")
                return constants.penalty_score
                
            # Check for invalid audio data
            if (np.isnan(gt_audio_arr).any() or 
                np.isnan(generated_audio_arr).any() or
                np.all(generated_audio_arr == 0) or
                np.all(gt_audio_arr == 0)):
                bt.logging.warning("Invalid audio data detected (NaN or all zeros)")
                return constants.penalty_score

            # Convert both to 16kHz mono
            gt_audio = self.convert_audio(gt_audio_arr, gt_sample_rate, 16000, 1)
            gen_audio = self.convert_audio(generated_audio_arr, generated_sample_rate, 16000, 1)

            if gen_audio is None or gt_audio is None:
                bt.logging.warning("Failed to convert audio for PESQ calculation")
                return constants.penalty_score

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
                return constants.penalty_score

            # Convert to tensors
            gt_tensor = torch.from_numpy(gt_audio).float().unsqueeze(0)
            gen_tensor = torch.from_numpy(gen_audio).float().unsqueeze(0)

            # Calculate scores
            nb_score = (self.nb_pesq(gen_tensor, gt_tensor).clip(-0.5, 4.5) + 0.5) / 5.0
            wb_score = (self.wb_pesq(gen_tensor, gt_tensor).clip(-0.5, 4.5) + 0.5) / 5.0
            
            # Weight narrow-band and wide-band scores
            result = (0.3 * nb_score + 0.7 * wb_score).detach().cpu().item()
            
            # Ensure the result is in [0,1]
            return max(0.0, min(1.0, result))
        except Exception as e:
            bt.logging.error(f"PESQ calculation error: {e}")
            return constants.penalty_score

    def calculate_anti_spoofing_score(self, generated_audio_arr, gt_audio_arr, gt_sample_rate, generated_sample_rate):
        """Calculate speaker similarity using RawNet3"""
        try:
            # Check for empty or invalid arrays
            if (generated_audio_arr is None or gt_audio_arr is None or 
                len(generated_audio_arr) == 0 or len(gt_audio_arr) == 0):
                bt.logging.warning("Empty audio array provided to anti-spoofing calculation")
                return constants.penalty_score
                
            # Check for NaN values
            if (np.isnan(generated_audio_arr).any() or np.isnan(gt_audio_arr).any()):
                bt.logging.warning("NaN values detected in audio for anti-spoofing calculation")
                return constants.penalty_score
                
            generated_embedding = self.anti_spoofing_inference.extract_speaker_embd(
                generated_audio_arr, generated_sample_rate, 48000, 10)
            gt_embedding = self.anti_spoofing_inference.extract_speaker_embd(
                gt_audio_arr, gt_sample_rate, 48000, 10)
            
            # Check if embeddings are valid
            if generated_embedding is None or gt_embedding is None:
                bt.logging.warning("Invalid embeddings from RawNet3")
                return constants.penalty_score
                
            # Calculate similarity as 1/(1 + MSE)
            similarity = 1 / (1 + ((generated_embedding - gt_embedding) ** 2).mean())
            
            # Ensure the result is a scalar
            result = similarity.detach().cpu().item()
            
            # Ensure the result is in [0,1]
            return max(0.0, min(1.0, result))
        except Exception as e:
            bt.logging.error(f"Anti-spoofing calculation error: {e}")
            return constants.penalty_score

    # def calculate_length_penalty(self, gt_length, gt_sample_rate, gen_length, gen_sample_rate):
    #     """Calculate length penalty using exp(-|log(L_gen / L_gt)|)"""
    #     try:
    #         # Check for invalid inputs
    #         if (gt_length is None or gen_length is None or 
    #             gt_sample_rate is None or gen_sample_rate is None or
    #             gt_sample_rate <= 0 or gen_sample_rate <= 0):
    #             bt.logging.warning("Invalid inputs for length penalty calculation")
    #             return constants.penalty_score
                
    #         gt_duration = gt_length / gt_sample_rate
    #         gen_duration = gen_length / gen_sample_rate
            
    #         # Avoid division by zero or very small values
    #         if gt_duration < 0.1 or gen_duration < 0.1:
    #             bt.logging.warning("Audio duration too short for reliable length penalty calculation")
    #             return constants.penalty_score
                
    #         ratio = gen_duration / gt_duration
            
    #         # Handle extreme ratios (too short or too long)
    #         if ratio < 0.1 or ratio > 10:
    #             bt.logging.warning(f"Extreme length ratio: {ratio:.2f}")
    #             return 0.1  # Severe penalty but not zero
                
    #         penalty = float(np.exp(-np.abs(np.log(ratio))))
            
    #         # Ensure the result is in [0,1]
    #         return max(0.0, min(1.0, penalty))
    #     except Exception as e:
    #         bt.logging.error(f"Length penalty calculation error: {e}")
    #         return constants.penalty_score

    def calculate_length_penalty(self, gt_length, gt_sample_rate, gen_length, gen_sample_rate):
        """Calculate length penalty using exp(-|log(L_gen / L_gt)|)"""
        gt_duration = gt_length / gt_sample_rate
        gen_duration = gen_length / gen_sample_rate

        print("gt_duration", gt_duration, "gen_duration", gen_duration)
        if gt_duration < 0.1 or gen_duration < 0.1:
            bt.logging.warning("Audio duration too short for reliable length penalty calculation")
            return constants.penalty_score
        ratio = gen_duration / gt_duration
        if ratio < 0.1 or ratio > 10:
            bt.logging.warning(f"Extreme length ratio: {ratio:.2f}")
            return 0.1
        # Soften penalty: 0.5 base + 0.5 scaled penalty
        penalty = 0.5 + 0.5 * np.exp(-np.abs(np.log(ratio)))
        return max(0.0, min(1.0, penalty))

    def calculate_naturalness(self, generated_audio_arr, generated_sample_rate):
        """Calculate naturalness score using MOSNet"""
        try:
            # Check for empty or invalid arrays
            if generated_audio_arr is None or len(generated_audio_arr) == 0:
                bt.logging.warning("Empty audio array provided to naturalness calculation")
                return constants.penalty_score
                
            # Check for NaN values
            if np.isnan(generated_audio_arr).any():
                bt.logging.warning("NaN values detected in audio for naturalness calculation")
                return constants.penalty_score
                
            # Check if audio is too short
            if len(generated_audio_arr) < 1000:  # Arbitrary minimum length
                bt.logging.warning("Audio too short for naturalness calculation")
                return constants.penalty_score
                
            # Get score from MOSNet (1-5 scale)
            score = self.mosnet.score(generated_audio_arr, generated_sample_rate)
            
            # Check if score is valid
            if score is None or np.isnan(score):
                bt.logging.warning("Invalid score from MOSNet")
                return constants.penalty_score
                
            # Normalize to [0,1]
            normalized_score = (score - 1) / 4
            
            # Ensure the result is in [0,1]
            return float(max(0.0, min(1.0, normalized_score)))
        except Exception as e:
            bt.logging.error(f"Naturalness calculation error: {e}")
            return constants.penalty_score

    def compute_distance(self, gt_audio_arrs: List[Tuple[np.ndarray, int]], generated_audio_arrs: List[Tuple[np.ndarray, int]]) -> Dict[str, List[float]]:
        """Compute all metrics between ground truth and generated audio"""
        try:
            # Check if input arrays are empty
            if not gt_audio_arrs or not generated_audio_arrs:
                bt.logging.error("Empty input arrays provided for metric computation")
                return {
                    'semantic_similarity_score': [constants.penalty_score],
                    'naturalness_score': [constants.penalty_score],
                    'pesq_score': [constants.penalty_score],
                    'anti_spoofing_score': [constants.penalty_score],
                    'length_penalty': [constants.penalty_score],
                    'combined_score': [constants.penalty_score]
                }
                
            # Initialize result arrays
            semantic_similarity_scores = []
            naturalness_scores = []
            pesq_scores = []
            anti_spoofing_scores = []
            length_penalties = []
            combined_scores = []
            
            for idx, ((gt_arr, gt_rate), (gen_arr, gen_rate)) in enumerate(zip(gt_audio_arrs, generated_audio_arrs)):
                bt.logging.info(f"Calculating metrics for pair {idx+1}/{len(gt_audio_arrs)}")
                
                # Skip empty arrays
                if gt_arr is None or gen_arr is None or len(gt_arr) == 0 or len(gen_arr) == 0:
                    bt.logging.warning(f"Skipping empty audio array in pair {idx+1}")
                    semantic_similarity_scores.append(constants.penalty_score)
                    naturalness_scores.append(constants.penalty_score)
                    pesq_scores.append(constants.penalty_score)
                    anti_spoofing_scores.append(constants.penalty_score)
                    length_penalties.append(constants.penalty_score)
                    combined_scores.append(constants.penalty_score)
                    continue
                
                # Semantic similarity (using HuBert)
                semantic_score = self.calculate_semantic_similarity(gt_arr, gen_arr, gt_rate, gen_rate)
                semantic_similarity_scores.append(semantic_score)
                
                # Naturalness score (using MOSNet instead of P.563)
                naturalness_score = self.calculate_naturalness(gen_arr, gen_rate)
                naturalness_scores.append(naturalness_score)
                
                # PESQ (audio clarity)
                pesq_score = self.calculate_pesq(gt_arr, gen_arr, gt_rate, gen_rate)
                pesq_scores.append(pesq_score)
                
                # Anti-spoofing (speaker identity preservation)
                anti_spoofing_score = self.calculate_anti_spoofing_score(gen_arr, gt_arr, gt_rate, gen_rate)
                anti_spoofing_scores.append(anti_spoofing_score)
                
                # Length penalty
                length_penalty = self.calculate_length_penalty(len(gt_arr), gt_rate, len(gen_arr), gen_rate)
                length_penalties.append(length_penalty)
                
                # Calculate combined score with weights
                # Get the scores we just calculated, or use penalty scores if they weren't added
                semantic_score = semantic_similarity_scores[-1] if semantic_similarity_scores else constants.penalty_score
                naturalness_score = naturalness_scores[-1] if naturalness_scores else constants.penalty_score
                pesq_score = pesq_scores[-1] if pesq_scores else constants.penalty_score
                anti_spoofing_score = anti_spoofing_scores[-1] if anti_spoofing_scores else constants.penalty_score
                length_penalty = length_penalties[-1] if length_penalties else constants.penalty_score
                
                # - Semantic (content) and Naturalness form the arithmetic mean
                # - PESQ (clarity) and Anti-spoofing (voice match) form the geometric mean
                # - Length penalty is applied as a multiplier
                content_score = (semantic_score + naturalness_score) / 2
                
                # Ensure positive values for geometric mean
                pesq_score_safe = max(constants.penalty_score, pesq_score)
                anti_spoofing_score_safe = max(constants.penalty_score, anti_spoofing_score)
                technical_score = (pesq_score_safe * anti_spoofing_score_safe) ** 0.5
                
                # Final score with weighted combination and length penalty
                combined_score = (0.6 * technical_score + 0.4 * content_score) * length_penalty
                combined_scores.append(combined_score)
                
                bt.logging.info(f"Scores for pair {idx+1}: semantic={semantic_score:.3f}, "
                               f"naturalness={naturalness_score:.3f}, pesq={pesq_score:.3f}, "
                               f"anti_spoofing={anti_spoofing_score:.3f}, length={length_penalty:.3f}, "
                               f"combined={combined_score:.3f}")
            
            # Ensure we have at least one score in each list
            if not semantic_similarity_scores:
                semantic_similarity_scores = [constants.penalty_score]
            if not naturalness_scores:
                naturalness_scores = [constants.penalty_score]
            if not pesq_scores:
                pesq_scores = [constants.penalty_score]
            if not anti_spoofing_scores:
                anti_spoofing_scores = [constants.penalty_score]
            if not length_penalties:
                length_penalties = [constants.penalty_score]
            if not combined_scores:
                combined_scores = [constants.penalty_score]
                
            # Ensure all scores are scalar values, not nested lists
            semantic_similarity_scores = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in semantic_similarity_scores]
            naturalness_scores = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in naturalness_scores]
            pesq_scores = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in pesq_scores]
            anti_spoofing_scores = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in anti_spoofing_scores]
            length_penalties = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in length_penalties]
            combined_scores = [float(score) if isinstance(score, (int, float)) else constants.penalty_score for score in combined_scores]
                
            return {
                'semantic_similarity_score': semantic_similarity_scores,
                'naturalness_score': naturalness_scores,
                'pesq_score': pesq_scores,
                'anti_spoofing_score': anti_spoofing_scores,
                'length_penalty': length_penalties,
                'combined_score': combined_scores
            }
        except Exception as e:
            bt.logging.error(f"Overall metric computation error: {e}")
            return {
                'semantic_similarity_score': [constants.penalty_score],
                'naturalness_score': [constants.penalty_score],
                'pesq_score': [constants.penalty_score],
                'anti_spoofing_score': [constants.penalty_score],
                'length_penalty': [constants.penalty_score],
                'combined_score': [constants.penalty_score]
            }

            
if __name__ == "__main__":
    metrics = S2SMetrics()
    result = metrics.compute_distance([(np.random.randn(16000), 16000)], [(np.random.randn(16000), 16000)])
    bt.logging.info(f"Test result: {result}")