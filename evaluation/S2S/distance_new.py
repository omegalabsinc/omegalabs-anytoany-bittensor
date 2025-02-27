import os
import numpy as np
import librosa
import bittensor as bt
import pandas as pd
import time
import torch
import torchaudio.functional as F
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    HubertModel,
    Wav2Vec2FeatureExtractor
)
from typing import Dict, Optional, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import constants

# Define penalty_score if not in constants
if not hasattr(constants, 'penalty_score'):
    constants.penalty_score = 0.0

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
            
        try:
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
                embeddings = outputs.last_hidden_state
                
                # Check if embeddings are valid
                if embeddings is None or embeddings.numel() == 0:
                    bt.logging.warning("Empty embeddings from HuBert model")
                    return None
                    
                # Average over time dimension
                mean_embedding = torch.mean(embeddings, dim=1)
                
                # Move back to CPU for numpy conversion
                return mean_embedding.cpu().numpy().squeeze()
        except Exception as e:
            bt.logging.error(f"Error in HuBert embedding: {e}")
            return None

class S2SMetrics:
    def __init__(self, cache_dir: str = ".checkpoints/"):
        """Initialize metrics with configuration"""
        # Setup cache directories
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Whisper
        bt.logging.info("Loading Whisper...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", cache_dir=cache_dir).to(self.device).eval()
        
        # Initialize HuBert for semantic similarity
        bt.logging.info("Loading HuBert...")
        self.hubert = HuBertEmbedder(cache_dir=cache_dir)
        
        # Initialize DNSMOS for naturalness
        bt.logging.info("Loading DNSMOS...")
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(16000, False)
        
        # Noise detection thresholds
        self.zcr_threshold = 0.35
        self.flatness_threshold = 0.3
        
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
        
        try:
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
        except Exception as e:
            bt.logging.error(f"Audio conversion error: {e}")
            return None

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

    def calculate_length_penalty(self, gt_length, gt_sample_rate, gen_length, gen_sample_rate):
        """Calculate length penalty using exp(-|log(L_gen / L_gt)|)"""
        try:
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
        except Exception as e:
            bt.logging.error(f"Length penalty calculation error: {e}")
            return constants.penalty_score

    def calculate_naturalness(self, generated_audio_arr, generated_sample_rate):
        """Calculate naturalness score using DNSMOS"""
        try:
            # Check for empty or invalid arrays
            if (generated_audio_arr is None or len(generated_audio_arr) == 0 or 
                np.isnan(generated_audio_arr).any() or len(generated_audio_arr) < 8000):
                bt.logging.warning("Invalid audio for naturalness calculation")
                return constants.penalty_score
                
            audio = self.convert_audio(generated_audio_arr, generated_sample_rate, 16000, 1)
            if audio is None:
                bt.logging.warning("Audio conversion failed for naturalness")
                return constants.penalty_score
                
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Calculate noise metrics
            zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
            if len(audio) >= 512:
                flatness = librosa.feature.spectral_flatness(y=audio, n_fft=512)[0].mean()
            else:
                flatness = 0.0
                
            with torch.no_grad():
                scores = self.dnsmos(audio_tensor)
                
            if scores is None or torch.isnan(scores[0]).any():
                bt.logging.warning("DNSMOS returned invalid scores")
                return constants.penalty_score
                
            p808_mos, sig_mos, bak_mos, ovr_mos = [s.item() for s in scores]
            weighted_mos = 0.4 * p808_mos + 0.4 * ovr_mos + 0.2 * sig_mos
            
            # Detailed logging
            print("p808_mos", p808_mos)
            print(f"DNSMOS raw scores: p808={p808_mos:.4f}, sig={sig_mos:.4f}, bak={bak_mos:.4f}, ovr={ovr_mos:.4f}")
            print(f"Weighted MOS before penalty: {weighted_mos:.4f}")
            
            # Apply noise penalty if detected
            if zcr > self.zcr_threshold and flatness > self.flatness_threshold:
                weighted_mos *= 0.2
                print("noise detected")
                
            # Normalize to [0,1] based on DNSMOS 1-5 scale
            base_score = (0.5*p808_mos + 0.5*ovr_mos - 1.0) / 4.0
            print(f"Base score (normalized): {base_score:.4f}")
            
            # Ensure the result is in [0,1]
            return max(0.0, min(1.0, base_score))
        except Exception as e:
            bt.logging.error(f"Naturalness calculation error: {e}")
            return constants.penalty_score

    def compute_distance(self, gt_audio_arrs: List[Tuple[np.ndarray, int]], generated_audio_arrs: List[Tuple[np.ndarray, int]]) -> Dict[str, List[float]]:
        """Compute all metrics between ground truth and generated audio"""
        try:
            # Check if input arrays are empty
            if not gt_audio_arrs or not generated_audio_arrs:
                bt.logging.error("Empty input arrays provided for metric computation")
                return self._default_metrics()
                
            # Initialize result arrays
            results = {
                'semantic_similarity_score': [],
                'naturalness_score': [],
                'length_penalty': [],
                'combined_score': []
            }
            
            for idx, ((gt_arr, gt_rate), (gen_arr, gen_rate)) in enumerate(zip(gt_audio_arrs, generated_audio_arrs)):
                bt.logging.info(f"Calculating metrics for pair {idx+1}/{len(gt_audio_arrs)}")
                
                # Skip empty arrays
                if gt_arr is None or gen_arr is None or len(gt_arr) == 0 or len(gen_arr) == 0:
                    bt.logging.warning(f"Skipping empty audio array in pair {idx+1}")
                    results['semantic_similarity_score'].append(constants.penalty_score)
                    results['naturalness_score'].append(constants.penalty_score)
                    results['length_penalty'].append(constants.penalty_score)
                    results['combined_score'].append(constants.penalty_score)
                    continue
                
                # Calculate individual metrics
                semantic_score = self.calculate_semantic_similarity(gt_arr, gen_arr, gt_rate, gen_rate)
                naturalness_score = self.calculate_naturalness(gen_arr, gen_rate)
                length_penalty = self.calculate_length_penalty(len(gt_arr), gt_rate, len(gen_arr), gen_rate)
                
                # Calculate combined score
                content_score = 0.4 * semantic_score + 0.6 * naturalness_score
                length_penalty = 1.0
                combined_score = content_score * length_penalty
                
                # Store results
                results['semantic_similarity_score'].append(semantic_score)
                results['naturalness_score'].append(naturalness_score)
                results['length_penalty'].append(length_penalty)
                results['combined_score'].append(combined_score)
                
                bt.logging.info(f"Scores for pair {idx+1}: semantic={semantic_score:.4f}, "
                              f"naturalness={naturalness_score:.4f}, length={length_penalty:.4f}, "
                              f"combined={combined_score:.4f}")
            
            # Ensure all scores are scalar values
            return {k: [float(v) if isinstance(v, (int, float)) else constants.penalty_score for v in val] 
                    for k, val in results.items()}
        except Exception as e:
            bt.logging.error(f"Overall metric computation error: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, List[float]]:
        """Return default metrics when computation fails"""
        return {
            'semantic_similarity_score': [constants.penalty_score],
            'naturalness_score': [constants.penalty_score],
            'length_penalty': [constants.penalty_score],
            'combined_score': [constants.penalty_score]
        }

# if __name__ == "__main__":
    # # Enable detailed logging
    # # bt.logging.set_level(bt.logging.INFO)  # or DEBUG for more detail
    # main()