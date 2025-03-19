import os
import numpy as np
import librosa
import bittensor as bt
import pandas as pd
import torch
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
)
from sentence_transformers import SentenceTransformer
import jiwer
from typing import Dict, Optional, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import constants

# Define penalty_score if not in constants
if not hasattr(constants, 'penalty_score'):
    constants.penalty_score = 0.0

class WhisperTranscriber:
    """Class for transcribing audio using Whisper and calculating text similarity."""
    
    def __init__(self, cache_dir=".checkpoints/", model_size="large-v2"):
        """
        Initialize WhisperTranscriber with specified model size.
        
        Args:
            cache_dir: Directory for caching model files
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2')
        """
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Whisper for transcription
        bt.logging.info(f"Loading Whisper {model_size} for transcription...")
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}", cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Load SentenceTransformer for text similarity
        bt.logging.info("Loading SentenceBERT for text similarity...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        self.sentence_transformer.to(self.device)
    
    def transcribe(self, audio, sample_rate):
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcription text
        """
        try:
            # Ensure audio is at 16kHz for Whisper
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Process with Whisper
            with torch.no_grad():
                input_features = self.processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features
                input_features = input_features.to(self.device)
                
                # Generate transcription
                generated_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
            return transcription
        except Exception as e:
            bt.logging.error(f"Transcription error: {e}")
            return ""
    
    def calculate_text_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using SentenceBERT.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0
                
            # Get embeddings
            embedding1 = self.sentence_transformer.encode([text1])[0]
            embedding2 = self.sentence_transformer.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            return float(similarity)
        except Exception as e:
            bt.logging.error(f"Text similarity calculation error: {e}")
            return 0.0
    
    def calculate_wer(self, reference, hypothesis):
        """
        Calculate Word Error Rate between reference and hypothesis.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            WER score (lower is better)
        """
        try:
            if not reference:
                return 1.0
                
            wer = jiwer.wer(reference, hypothesis)
            return min(wer, 1.0)  # Cap at 1.0
        except Exception as e:
            bt.logging.error(f"WER calculation error: {e}")
            return 1.0

class S2SMetrics:
    def __init__(self, cache_dir: str = ".checkpoints/"):
        """Initialize metrics with configuration"""
        # Setup cache directories
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Whisper transcriber for semantic similarity
        bt.logging.info("Loading WhisperTranscriber...")
        self.transcriber = WhisperTranscriber(cache_dir=cache_dir, model_size="large-v2")
        
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
        """Calculate semantic similarity based on Whisper transcription and text comparison"""
        try:
            # Check for valid audio
            if gt_audio_arr is None or generated_audio_arr is None:
                bt.logging.warning("Invalid audio arrays provided for semantic similarity")
                return constants.penalty_score
                
            # Ensure audio is not too short
            min_length = 8000  # At least 0.5 second at 16kHz
            if len(gt_audio_arr) / gt_sample_rate < 0.5 or len(generated_audio_arr) / generated_sample_rate < 0.5:
                bt.logging.warning("Audio too short for reliable transcription")
                return constants.penalty_score
            
            # Convert audio for processing
            gt_audio = self.convert_audio(gt_audio_arr, gt_sample_rate, 16000, 1)
            gen_audio = self.convert_audio(generated_audio_arr, generated_sample_rate, 16000, 1)
            
            if gt_audio is None or gen_audio is None:
                bt.logging.warning("Audio conversion failed for semantic similarity")
                return constants.penalty_score
            
            # Transcribe both audios
            bt.logging.info("Transcribing ground truth audio...")
            gt_transcription = self.transcriber.transcribe(gt_audio, 16000)
            
            bt.logging.info("Transcribing generated audio...")
            gen_transcription = self.transcriber.transcribe(gen_audio, 16000)
            
            # Log transcriptions
            bt.logging.info(f"Ground truth transcription: '{gt_transcription}'")
            bt.logging.info(f"Generated transcription: '{gen_transcription}'")
            
            # Calculate text similarity
            text_similarity = self.transcriber.calculate_text_similarity(gt_transcription, gen_transcription)
            bt.logging.info(f"Text semantic similarity: {text_similarity:.4f}")
            
            combined_score = text_similarity
            bt.logging.info(f"Combined semantic similarity score: {combined_score:.4f}")
            
            return combined_score
        except Exception as e:
            bt.logging.error(f"Semantic similarity calculation error: {e}")
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
            bt.logging.info(f"DNSMOS raw scores: p808={p808_mos:.4f}, sig={sig_mos:.4f}, bak={bak_mos:.4f}, ovr={ovr_mos:.4f}")
            bt.logging.info(f"Weighted MOS before penalty: {weighted_mos:.4f}")
            
            # Apply noise penalty if detected
            if zcr > self.zcr_threshold and flatness > self.flatness_threshold:
                weighted_mos *= 0.2
                bt.logging.info("Excessive noise detected, applying penalty")
                
            # Normalize to [0,1] based on DNSMOS 1-5 scale
            base_score = (weighted_mos - 1.0) / 4.0
            bt.logging.info(f"Naturalness score (normalized): {base_score:.4f}")
            
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
                'combined_score': []
            }
            
            for idx, ((gt_arr, gt_rate), (gen_arr, gen_rate)) in enumerate(zip(gt_audio_arrs, generated_audio_arrs)):
                bt.logging.info(f"Calculating metrics for pair {idx+1}/{len(gt_audio_arrs)}")
                
                # Skip empty arrays
                if gt_arr is None or gen_arr is None or len(gt_arr) == 0 or len(gen_arr) == 0:
                    bt.logging.warning(f"Skipping empty audio array in pair {idx+1}")
                    for key in results:
                        results[key].append(constants.penalty_score)
                    continue
                
                # Calculate individual metrics
                semantic_score = self.calculate_semantic_similarity(gt_arr, gen_arr, gt_rate, gen_rate)
                naturalness_score = self.calculate_naturalness(gen_arr, gen_rate)
                combined_score = semantic_score*0.2 + naturalness_score*0.8
                
                # Store individual metrics
                results['semantic_similarity_score'].append(semantic_score)
                results['naturalness_score'].append(naturalness_score)
                results['combined_score'].append(combined_score)
               
            # Ensure all scores are scalar values
            return {k: [float(v) if isinstance(v, (int, float)) else constants.penalty_score for v in val] 
                    for k, val in results.items()}
        except Exception as e:
            bt.logging.error(f"Overall metric computation error: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, List[float]]:
        """Return default metrics when computation fails"""
        default_metrics = {
            'semantic_similarity_score': [constants.penalty_score],
            'naturalness_score': [constants.penalty_score],
            'combined_score': [constants.penalty_score]
        }
            
        return default_metrics
        
    def evaluate_and_display(self, gt_audio_arrs, generated_audio_arrs, model_name="Model"):
        """Evaluate and display metrics in a formatted table"""
        metrics = self.compute_distance(gt_audio_arrs, generated_audio_arrs)
        
        # Create a DataFrame for nicer display
        df = pd.DataFrame({
            'semantic_score': metrics['semantic_similarity_score'],
            'natural_score': metrics['naturalness_score'],
            'combined_score': metrics['combined_score']
        })
        
        # Print header
        print(f"Scores for model: {model_name}")
        print("-" * 80)
        
        # Print DataFrame
        print(df.round(4))
        
        # Print footer and summary
        print("=" * 80)
        
        # Return metrics for further analysis
        return metrics, df