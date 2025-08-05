"""
VoiceBench Adapter for Docker-based Models

This module provides an adapter that allows Docker-containerized voice models
to be evaluated using the VoiceBench framework. It bridges the interface gap
between VoiceBench's expected model API and the Docker container HTTP API.
"""

import os
import sys
import subprocess
import tempfile
import json
import numpy as np
import time
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
import bittensor as bt

# Add VoiceBench to Python path
VOICEBENCH_PATH = "/home/salman/anmol/VoiceBench"
if VOICEBENCH_PATH not in sys.path:
    sys.path.insert(0, VOICEBENCH_PATH)

from datasets import load_dataset, Audio
from neurons.docker_manager import DockerManager
from neurons.miner_model_assistant import MinerModelAssistant, create_miner_assistant
from neurons.llm_judge import calculate_llm_scores


class MinerModelAdapter:
    """
    Adapter that makes miner model APIs compatible with VoiceBench evaluation.
    
    This class implements the VoiceBench model interface while internally calling
    miner model HTTP APIs for inference.
    """
    
    def __init__(self, api_url: str, timeout: int = 600):
        """
        Initialize the adapter.
        
        Args:
            api_url: URL of the miner model API
            timeout: Request timeout in seconds
        """
        self.miner_assistant = MinerModelAssistant(api_url=api_url, timeout=timeout)
        
    def generate_audio(self, audio_input: Dict[str, Any]) -> str:
        """
        Generate response from audio input (VoiceBench interface).
        
        Args:
            audio_input: Audio data dict with 'array' and 'sampling_rate' keys
            
        Returns:
            Text response from the model
        """
        try:
            # Validate input
            if not isinstance(audio_input, dict):
                raise ValueError("audio_input must be a dictionary")
            
            if 'array' not in audio_input or 'sampling_rate' not in audio_input:
                raise ValueError("audio_input must contain 'array' and 'sampling_rate' keys")
            
            # Call the miner model assistant
            response = self.miner_assistant.generate_audio(audio_input)
            
            if not response:
                bt.logging.warning("Empty response from miner model")
                return ""
            
            return response.strip()
            
        except Exception as e:
            bt.logging.error(f"Error in miner model inference: {e}")
            return ""
    
    def generate_text(self, text_input: str) -> str:
        """
        Generate response from text input.
        
        Args:
            text_input: Text prompt
            
        Returns:
            Text response (error message for audio-only models)
        """
        return self.miner_assistant.generate_text(text_input)
    
    def generate_ttft(self, audio_input: Dict[str, Any]) -> str:
        """
        Generate response for time-to-first-token measurement.
        
        Args:
            audio_input: Audio data dict
            
        Returns:
            Text response
        """
        return self.generate_audio(audio_input)


class DockerModelAdapter:
    """
    Adapter that makes Docker-containerized models compatible with VoiceBench evaluation.
    
    This class implements the VoiceBench model interface while internally managing
    Docker containers for model inference.
    """
    
    def __init__(self, container_url: str, docker_manager: DockerManager):
        """
        Initialize the adapter.
        
        Args:
            container_url: URL of the running Docker container
            docker_manager: DockerManager instance for container operations
        """
        self.container_url = container_url
        self.docker_manager = docker_manager
        
    def generate_audio(self, audio_input: Dict[str, Any]) -> str:
        """
        Generate response from audio input (VoiceBench interface).
        
        Args:
            audio_input: Audio data dict with 'array' and 'sampling_rate' keys
            
        Returns:
            Text response from the model
        """
        try:
            # Validate input
            if not isinstance(audio_input, dict):
                raise ValueError("audio_input must be a dictionary")
            
            if 'array' not in audio_input or 'sampling_rate' not in audio_input:
                raise ValueError("audio_input must contain 'array' and 'sampling_rate' keys")
            
            audio_array = np.array(audio_input['array'])
            sample_rate = audio_input['sampling_rate']
            
            # Validate audio data
            if audio_array.size == 0:
                bt.logging.warning("Empty audio array provided")
                return ""
            
            if sample_rate <= 0:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
            
            # Call Docker container inference
            result = self.docker_manager.inference_v2v(
                url=self.container_url,
                audio_array=audio_array,
                sample_rate=sample_rate
            )
            
            # Validate and extract text response
            if not isinstance(result, dict):
                bt.logging.warning(f"Unexpected result type: {type(result)}")
                return ""
            
            # Handle voice-to-voice models that may not return text
            text_response = result.get('text', '')
            
            # If no text response, this might be a voice-to-voice only model
            if not text_response and 'audio' in result:
                bt.logging.info("Model returned audio but no text - this appears to be a voice-to-voice only model")
                # For VoiceBench evaluation, we need text responses
                # We could potentially use speech-to-text here, but for now return empty
                return "[VOICE_OUTPUT_NO_TEXT]"
            
            if not isinstance(text_response, str):
                text_response = str(text_response) if text_response is not None else ""
            
            return text_response.strip()
            
        except Exception as e:
            bt.logging.error(f"Error in Docker model inference: {e}")
            return ""
    
    def generate_text(self, text_input: str) -> str:
        """
        Generate response from text input (fallback for text-only evaluation).
        
        Args:
            text_input: Text prompt
            
        Returns:
            Text response (empty for audio-only models)
        """
        return ""
    
    def generate_ttft(self, audio_input: Dict[str, Any]) -> str:
        """
        Generate response for time-to-first-token measurement.
        
        Args:
            audio_input: Audio data dict
            
        Returns:
            Text response
        """
        return self.generate_audio(audio_input)


class VoiceBenchEvaluator:
    """
    Main evaluator that runs VoiceBench evaluation on Docker-containerized models.
    """
    
    # All available VoiceBench datasets with their appropriate splits
    VOICEBENCH_DATASETS = {
        'alpacaeval': ['test'],
        'alpacaeval_full': ['test'], 
        'commoneval': ['test'],
        'wildvoice': ['test'],
        'openbookqa': ['test'],
        'mmsu': ['physics'],  # Use one specific domain instead of 'test'
        'sd-qa': ['usa'],  # Using USA dataset only
        'mtbench': ['test'],
        'ifeval': ['test'],
        'bbh': ['test'],
        'advbench': ['test']
    }
    
    def __init__(self, voicebench_path: str = VOICEBENCH_PATH):
        """
        Initialize the VoiceBench evaluator.
        
        Args:
            voicebench_path: Path to VoiceBench installation
        """
        self.voicebench_path = Path(voicebench_path)
        self.setup_environment()
    
    def setup_environment(self):
        """Set up the VoiceBench environment."""
        # Ensure VoiceBench is in Python path
        if str(self.voicebench_path) not in sys.path:
            sys.path.insert(0, str(self.voicebench_path))
    
    def _generate_with_timeout(self, func, *args, timeout=30):
        """
        Execute a function with timeout.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            timeout: Timeout in seconds
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If function takes longer than timeout
        """
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("Function call timed out")
        
        # Set up timeout handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(*args)
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutException:
            raise TimeoutError(f"Function call timed out after {timeout} seconds")
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
    
    def evaluate_model(
        self,
        container_url: str,
        docker_manager: DockerManager,
        datasets: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        modality: str = 'audio'
    ) -> Dict[str, Any]:
        """
        Evaluate a Docker-containerized model using VoiceBench.
        
        Args:
            container_url: URL of the running Docker container
            docker_manager: DockerManager instance
            datasets: List of VoiceBench datasets to evaluate on (None = all datasets)
            splits: List of data splits to use (None = use dataset-specific splits)
            modality: Evaluation modality ('audio', 'text', or 'ttft')
            
        Returns:
            Dictionary containing evaluation results
        """
        adapter = DockerModelAdapter(container_url, docker_manager)
        results = {}
        
        # Use all datasets if none specified
        if datasets is None:
            datasets_to_eval = self.VOICEBENCH_DATASETS
        else:
            datasets_to_eval = {k: self.VOICEBENCH_DATASETS[k] for k in datasets if k in self.VOICEBENCH_DATASETS}
        
        for dataset_name, dataset_splits in datasets_to_eval.items():
            # Use dataset-specific splits if none provided
            splits_to_use = splits if splits is not None else dataset_splits
            
            for split in splits_to_use:
                try:
                    # Load VoiceBench dataset
                    dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
                    
                    # For testing, limit to first 6 samples
                    if len(dataset) > 6:
                        dataset = dataset.select([0,1,2,3,4,5])
                    
                    # Run evaluation
                    dataset_results = self._evaluate_dataset(
                        adapter, dataset, dataset_name, split, modality
                    )
                    
                    results[f"{dataset_name}_{split}"] = dataset_results
                    
                except Exception as e:
                    bt.logging.error(f"Error evaluating {dataset_name}_{split}: {e}")
                    results[f"{dataset_name}_{split}"] = {"error": str(e)}
        
        return results
    
    def _evaluate_with_adapter(
        self,
        model_adapter,
        datasets: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        modality: str = 'audio'
    ) -> Dict[str, Any]:
        """
        Evaluate model using a pre-created adapter.
        
        Args:
            model_adapter: Pre-configured model adapter
            datasets: List of VoiceBench datasets to evaluate on (None = all datasets)
            splits: List of data splits to use (None = use dataset-specific splits)
            modality: Evaluation modality ('audio', 'text', or 'ttft')
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {}
        
        # Use all datasets if none specified
        if datasets is None:
            datasets_to_eval = self.VOICEBENCH_DATASETS
        else:
            datasets_to_eval = {k: self.VOICEBENCH_DATASETS[k] for k in datasets if k in self.VOICEBENCH_DATASETS}
        
        for dataset_name, dataset_splits in datasets_to_eval.items():
            # Use dataset-specific splits if none provided
            splits_to_use = splits if splits is not None else dataset_splits
            
            for split in splits_to_use:
                try:
                    # Load VoiceBench dataset
                    dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
                    
                    # For testing, limit to first 6 samples
                    if len(dataset) > 6:
                        dataset = dataset.select([0,1,2,3,4,5])
                    
                    # Run evaluation
                    dataset_results = self._evaluate_dataset(
                        model_adapter, dataset, dataset_name, split, modality
                    )
                    
                    results[f"{dataset_name}_{split}"] = dataset_results
                    
                except Exception as e:
                    bt.logging.error(f"Error evaluating {dataset_name}_{split}: {e}")
                    results[f"{dataset_name}_{split}"] = {"error": str(e)}
        
        return results
    
    def _evaluate_dataset(
        self,
        model_adapter: DockerModelAdapter,
        dataset,
        dataset_name: str,
        split: str,
        modality: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            model_adapter: DockerModelAdapter instance
            dataset: Loaded dataset
            dataset_name: Name of the dataset
            split: Data split
            modality: Evaluation modality
            
        Returns:
            Evaluation results for this dataset
        """
        responses = []
        
        # Generate responses with timeout and retry logic
        for i, item in enumerate(dataset):
            max_retries = 2
            retry_delay = 5
            
            for attempt in range(max_retries + 1):
                try:
                    # Add timeout for individual inference calls
                    start_time = time.time()
                    
                    if modality == 'audio':
                        # Check if audio field exists
                        if 'audio' not in item:
                            bt.logging.warning(f"No audio field in item {i+1}, skipping")
                            response = ""
                        else:
                            response = self._generate_with_timeout(
                                model_adapter.generate_audio, item['audio'], timeout=60
                            )
                    elif modality == 'text':
                        response = self._generate_with_timeout(
                            model_adapter.generate_text, item['prompt'], timeout=30
                        )
                    elif modality == 'ttft':
                        response = self._generate_with_timeout(
                            model_adapter.generate_ttft, item['audio'], timeout=30
                        )
                    else:
                        raise ValueError(f"Unsupported modality: {modality}")
                    
                    # Validate response
                    if not isinstance(response, str):
                        response = str(response) if response is not None else ""
                    
                    inference_time = time.time() - start_time
                    
                    responses.append({
                        'prompt': item.get('prompt', ''),
                        'response': response,
                        'reference': item.get('output', ''),
                        'inference_time': inference_time,
                        'attempt': attempt + 1,
                        **{k: v for k, v in item.items() if k not in ['audio', 'prompt', 'output']}
                    })
                    
                    # Success, break retry loop
                    break
                    
                except Exception as e:
                    bt.logging.warning(f"Error processing item {i+1}/{len(dataset)} (attempt {attempt+1}): {e}")
                    
                    if attempt < max_retries:
                        bt.logging.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final attempt failed
                        responses.append({
                            'prompt': item.get('prompt', ''),
                            'response': '',
                            'reference': item.get('output', ''),
                            'error': str(e),
                            'failed_attempts': attempt + 1
                        })
                        break
        
        # Calculate basic metrics
        total_responses = len(responses)
        successful_responses = len([r for r in responses if 'error' not in r and r['response']])
        success_rate = successful_responses / total_responses if total_responses > 0 else 0
        
        return {
            'dataset': dataset_name,
            'split': split,
            'modality': modality,
            'total_samples': total_responses,
            'successful_responses': successful_responses,
            'success_rate': success_rate,
            'responses': responses
        }
    
    def run_gpt_evaluation(self, results_file: str) -> Dict[str, Any]:
        """
        Run GPT-4 evaluation on generated responses.
        
        Args:
            results_file: Path to results file
            
        Returns:
            GPT evaluation results
        """
        try:
            # Run VoiceBench GPT evaluation
            cmd = [
                sys.executable,
                str(self.voicebench_path / "api_judge.py"),
                "--src_file", results_file
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.voicebench_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Load evaluation results
                eval_file = results_file.replace('.jsonl', '_eval.jsonl')
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        eval_results = [json.loads(line) for line in f]
                    return {'gpt_evaluation': eval_results}
                else:
                    return {'gpt_evaluation': 'completed', 'message': 'Evaluation file not found'}
            else:
                return {'error': f"GPT evaluation failed: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {'error': 'GPT evaluation timed out'}
        except Exception as e:
            return {'error': f'GPT evaluation error: {str(e)}'}
    
    def calculate_final_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate final VoiceBench scores with dataset-specific weighting.
        
        Args:
            results: Complete evaluation results
            
        Returns:
            Dictionary of final scores
        """
        scores = {}
        dataset_weights = {
            # Core conversational datasets (higher weight)
            'alpacaeval_test': 2.0,
            'commoneval_test': 2.0,
            'wildvoice_test': 2.0,
            
            # Reasoning and QA datasets
            'bbh_test': 1.5,
            'mmsu_test': 1.5,
            'openbookqa_test': 1.0,
            
            # Safety and instruction following
            'advbench_test': 1.5,
            'ifeval_test': 1.5,
            
            # Multi-turn and regional
            'mtbench_test': 1.0,
            'sd-qa_aus': 0.8,
            'sd-qa_usa': 0.8,
            'sd-qa_gbr': 0.8,
            'sd-qa_ind': 0.8,
            
            # Full alpaca dataset
            'alpacaeval_full_test': 1.0
        }
        
        weighted_scores = {}
        total_weight = 0.0
        
        for dataset_key, dataset_results in results.items():
            if 'error' in dataset_results:
                scores[dataset_key] = 0.0
                continue
            
            # Calculate dataset score based on success rate and response quality
            success_rate = dataset_results.get('success_rate', 0.0)
            
            # For datasets with responses, calculate average response length as quality indicator
            responses = dataset_results.get('responses', [])
            if responses:
                valid_responses = [r for r in responses if 'error' not in r and r.get('response', '').strip()]
                voice_only_responses = [r for r in responses if 'error' not in r and r.get('response', '') == '[VOICE_OUTPUT_NO_TEXT]']
                
                if valid_responses:
                    avg_response_length = sum(len(r.get('response', '')) for r in valid_responses) / len(valid_responses)
                    # Normalize response length (assume good responses are 50-500 chars)
                    length_score = min(1.0, max(0.1, avg_response_length / 200.0))
                    # Combine success rate with response quality
                    dataset_score = (success_rate * 0.7) + (length_score * 0.3)
                elif voice_only_responses:
                    # Handle voice-to-voice only models
                    # They get partial credit for producing audio output
                    bt.logging.info(f"Dataset {dataset_key}: Voice-to-voice only model detected ({len(voice_only_responses)} voice responses)")
                    dataset_score = success_rate * 0.3  # Reduced score for voice-only models in text evaluation
                else:
                    dataset_score = success_rate * 0.5  # Penalize for no valid responses
            else:
                dataset_score = success_rate
            
            scores[dataset_key] = dataset_score
            
            # Apply weighting for overall score calculation
            weight = dataset_weights.get(dataset_key, 1.0)
            weighted_scores[dataset_key] = dataset_score * weight
            total_weight += weight
        
        # Calculate weighted overall score
        if total_weight > 0:
            scores['overall'] = sum(weighted_scores.values()) / total_weight
        else:
            scores['overall'] = 0.0
            
        return scores


def run_voicebench_evaluation_miner(
    api_url: str,
    datasets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Convenience function to run VoiceBench evaluation on a miner model API.
    
    Args:
        api_url: URL of the miner model API
        datasets: List of datasets to evaluate (None = all VoiceBench datasets)
        splits: List of splits to evaluate (None = use dataset-specific splits)
        timeout: Request timeout for API calls
        
    Returns:
        Complete evaluation results including scores
    """
    evaluator = VoiceBenchEvaluator()
    
    bt.logging.info("Starting comprehensive VoiceBench evaluation via miner API...")
    
    # Create miner model adapter
    adapter = MinerModelAdapter(api_url=api_url, timeout=timeout)
    
    # Run model evaluation on all VoiceBench datasets
    results = evaluator._evaluate_with_adapter(
        model_adapter=adapter,
        datasets=datasets,  # None means all datasets
        splits=splits,      # None means dataset-specific splits
        modality='audio'
    )
    
    # Calculate traditional scores
    scores = evaluator.calculate_final_scores(results)
    
    # Calculate LLM-based scores
    bt.logging.info("Starting LLM-based evaluation...")
    llm_scores = calculate_llm_scores(results)
    
    bt.logging.info(f"VoiceBench evaluation completed.")
    bt.logging.info(f"Traditional score: {scores.get('overall', 0.0):.3f}")
    bt.logging.info(f"🤖 LLM-based score (PRIMARY): {llm_scores.get('overall', 0.0):.3f}")
    
    # Use LLM scores as the primary VoiceBench score
    primary_scores = llm_scores.copy()
    
    return {
        'voicebench_results': results,
        'voicebench_scores': primary_scores,  # LLM scores as primary
        'traditional_scores': scores,         # Keep traditional for comparison
        'llm_scores': llm_scores
    }


def run_voicebench_evaluation(
    container_url: str,
    docker_manager: DockerManager,
    datasets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run VoiceBench evaluation on a Docker model.
    
    Args:
        container_url: URL of the running Docker container
        docker_manager: DockerManager instance
        datasets: List of datasets to evaluate (None = all VoiceBench datasets)
        splits: List of splits to evaluate (None = use dataset-specific splits)
        
    Returns:
        Complete evaluation results including scores
    """
    evaluator = VoiceBenchEvaluator()
    
    bt.logging.info("Starting comprehensive VoiceBench evaluation across all datasets...")
    
    # Run model evaluation on all VoiceBench datasets
    results = evaluator.evaluate_model(
        container_url=container_url,
        docker_manager=docker_manager,
        datasets=datasets,  # None means all datasets
        splits=splits,      # None means dataset-specific splits
        modality='audio'
    )
    
    # Calculate traditional scores
    scores = evaluator.calculate_final_scores(results)
    
    # Calculate LLM-based scores
    bt.logging.info("Starting LLM-based evaluation...")
    llm_scores = calculate_llm_scores(results)
    
    bt.logging.info(f"VoiceBench evaluation completed.")
    bt.logging.info(f"Traditional score: {scores.get('overall', 0.0):.3f}")
    bt.logging.info(f"🤖 LLM-based score (PRIMARY): {llm_scores.get('overall', 0.0):.3f}")
    
    # Use LLM scores as the primary VoiceBench score
    primary_scores = llm_scores.copy()
    
    return {
        'voicebench_results': results,
        'voicebench_scores': primary_scores,  # LLM scores as primary
        'traditional_scores': scores,         # Keep traditional for comparison
        'llm_scores': llm_scores
    }