"""
VoiceBench Adapter for Docker-based Models

This module provides an adapter that allows Docker-containerized voice models
to be evaluated using the VoiceBench framework. It bridges the interface gap
between VoiceBench's expected model API and the Docker container HTTP API.
"""

import json
import time
import signal
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import bittensor as bt
import random

# Remove VoiceBench dependency - everything is now in neurons/

from datasets import load_dataset, Audio
from neurons.docker_manager import DockerManager
from neurons.miner_model_assistant import MinerModelAssistant
from neurons.llm_judge import evaluate_responses_with_llm
from neurons.voicebench_evaluators import (
    OpenEvaluator, MCQEvaluator, IFEvaluator, 
    BBHEvaluator, HarmEvaluator
)
from constants import VOICEBENCH_MAX_SAMPLES, SAMPLES_PER_DATASET

# Dataset to evaluator mapping based on VoiceBench DATASETS_CONFIG
DATASET_EVALUATOR_MAP = {
    # Open-ended QA datasets (need GPT evaluation)
    'alpacaeval': 'open',
    'alpacaeval_full': 'open',
    'commoneval': 'open',
    'wildvoice': 'open',
    
    # Multiple-choice QA datasets (no GPT needed)
    'openbookqa': 'mcq',
    'mmsu': 'mcq',  # All MMSU subjects use MCQ
    
    # Instruction following (no GPT needed)
    'ifeval': 'ifeval',
    
    # Reasoning (no GPT needed)
    'bbh': 'bbh',
    
    # Safety (no GPT needed)
    'advbench': 'harm',
}

# Datasets that need LLM scoring first
NEEDS_LLM_JUDGE = ['alpacaeval', 'alpacaeval_full', 'commoneval', 'wildvoice']


def evaluate_dataset_with_proper_evaluator(
    dataset_name: str,
    dataset_results: Dict[str, Any]
) -> tuple[float, Dict[str, Any]]:
    """
    Evaluate dataset responses using the correct VoiceBench evaluator.
    
    Returns:
        Tuple of (score, status_dict)
        
        status = {
                'dataset': dataset_name,
                'total_samples': dataset_results.get('total_samples', 0),
                'successful_responses': dataset_results.get('successful_responses', 0),
                'success_rate': dataset_results.get('success_rate', 0.0),
                'evaluator_used': None,
                'evaluation_status': 'pending',
                'evaluation_error': None,
                'evaluation_details': None,
                'score': 0.0,
                'sample_details': []  # NEW: List of per-sample details
            }
    """
    status = {
        'dataset': dataset_name,
        'total_samples': dataset_results.get('total_samples', 0),
        'successful_responses': dataset_results.get('successful_responses', 0),
        'success_rate': dataset_results.get('success_rate', 0.0),
        'evaluator_used': None,
        'evaluation_status': 'pending',
        'evaluation_error': None,
        'evaluation_details': None,
        'evaluation_time': 0.0,
        'score': 0.0,
        'sample_details': []  # NEW: Initialize empty list for sample details
    }
    
    start = time.perf_counter()
    
    # Check for dataset-level error
    if 'error' in dataset_results:
        status['evaluation_status'] = 'failed'
        status['evaluation_error'] = dataset_results['error']
        return 0.0, status
    
    # Get responses
    responses = dataset_results.get('responses', [])
    if not responses:
        status['evaluation_status'] = 'no_responses'
        return 0.0, status
    
    # Determine evaluator type
    dataset_base = dataset_name.split('_')[0]  # Handle dataset_split format
    evaluator_type = DATASET_EVALUATOR_MAP.get(dataset_base, 'open')
    status['evaluator_used'] = evaluator_type
    bt.logging.info(f"Using {evaluator_type} evaluator for {dataset_name}")
    try:
        # Prepare data for evaluation
        eval_data = []
        
        if evaluator_type == 'open':
            # First get LLM scores
            #TODO: Need some status of llm api calls. how many failed? which model failed?
            bt.logging.info(f"Getting LLM scores for {dataset_name}")
            llm_responses = evaluate_responses_with_llm(responses)
            
            # Build sample details for open evaluator datasets
            for i, response in enumerate(llm_responses):
                # Extract the llm_score from the response dict
                # # Convert score to string format expected by OpenEvaluator
                scores = response.get('score', ['0'])
                eval_data.append({
                    'score':  scores  # OpenEvaluator expects list of score strings
                })
                
                # Build sample detail
                sample_detail = {
                    'hf_index': response.get('hf_index', i),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': response.get('llm_raw_response', ''),
                    'llm_scores': scores if isinstance(scores, list) else [scores],
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
                    
            bt.logging.info(f"Got LLM scores for {dataset_name} for {len(llm_responses)} responses")
        elif evaluator_type == 'mcq':
            # MCQEvaluator needs response and reference
            for response in responses:
                eval_data.append({
                    'response': response.get('response', ''),
                    'reference': response.get('reference', '')
                })
                
        elif evaluator_type == 'ifeval':
            # IFEvaluator needs special format
            for response in responses:
                eval_data.append({
                    'key': response.get('id', 0),
                    'instruction_id_list': response.get('instruction_id_list', []),
                    'prompt': response.get('prompt', ''),
                    'response': response.get('response', ''),
                    'kwargs': response.get('kwargs', [])
                })
                
                # Build sample detail for IFEval
                sample_detail = {
                    'hf_index': response.get('hf_index', 0),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': '',  # IFEval doesn't use LLM judge
                    'llm_scores': [],  # No LLM scores for IFEval
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
                
        elif evaluator_type == 'bbh':
            # BBHEvaluator needs id and reference
            for response in responses:
                # BBH task IDs have specific format like 'bbh_navigate_147'
                task_id = response.get('id', response.get('task_id', 'unknown'))
                if isinstance(task_id, str) and task_id.startswith('bbh_'):
                    # Use the existing BBH task ID
                    bbh_id = task_id
                else:
                    # Create a default BBH task ID
                    bbh_id = f'bbh_unknown_{task_id}'
                    
                eval_data.append({
                    'id': bbh_id,
                    'response': response.get('response', ''),
                    'reference': response.get('reference', '')
                })
                
        elif evaluator_type == 'harm':
            # HarmEvaluator just needs response
            for response in responses:
                eval_data.append({
                    'response': response.get('response', '')
                })
                
                # Build sample detail for Harm evaluator (advbench)
                # Note: We'll add the per-sample harm score after evaluation
                sample_detail = {
                    'hf_index': response.get('hf_index', 0),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': '',  # Harm doesn't use LLM judge
                    'llm_scores': [],  # Will be populated with harm score after evaluation
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
        
        # Create and run evaluator
        if evaluator_type == 'open':
            evaluator = OpenEvaluator()
        elif evaluator_type == 'mcq':
            evaluator = MCQEvaluator()
        elif evaluator_type == 'ifeval':
            evaluator = IFEvaluator()
        elif evaluator_type == 'bbh':
            evaluator = BBHEvaluator()
        elif evaluator_type == 'harm':
            evaluator = HarmEvaluator()
        else:
            bt.logging.warning(f"Unknown evaluator type {evaluator_type}, using OpenEvaluator")
            evaluator = OpenEvaluator()
        
        # Evaluate
        eval_result = evaluator.evaluate(eval_data)
        
        # Store per-sample scores for harm evaluator
        if evaluator_type == 'harm' and 'per_sample_scores' in eval_result:
            per_sample_scores = eval_result['per_sample_scores']
            for i, score in enumerate(per_sample_scores):
                # Store as 1.0 for safe/refused, 0.0 for unsafe
                status['sample_details'][i]['llm_scores'] = [float(score)]
            eval_result.pop('per_sample_scores', None)

        # Extract score based on evaluator output format
        if 'gpt' in eval_result:
            # OpenEvaluator returns 1-5, normalize to 0-1
            score = eval_result['gpt'] / 5.0
        elif 'acc' in eval_result:
            # MCQ/BBH return 0-100 percentage
            score = eval_result['acc'] / 100.0
        elif 'final' in eval_result:
            # IFEvaluator returns final score 0-1
            score = eval_result['final']
        elif 'refusal_rate' in eval_result:
            # HarmEvaluator returns refusal rate (higher is better for safety)
            score = eval_result['refusal_rate']
        elif 'strict-prompt' in eval_result:
            # IFEvaluator alternative
            score = eval_result['strict-prompt']
        else:
            bt.logging.warning(f"Unknown evaluation result format: {eval_result}")
            score = 0.0
        
        # limit the sample_details to any random 10 samples
        if len(status['sample_details']) > 10:
            status['sample_details'] = random.sample(status['sample_details'], 10)

        status['score'] = score
        status['evaluation_status'] = 'completed'
        status['evaluation_details'] = eval_result
        status['evaluation_time'] = time.perf_counter() - start

        bt.logging.info(f"Dataset {dataset_name} evaluated with {evaluator_type}: score={score:.3f}")
        
        return score, status
        
    except Exception as e:
        bt.logging.error(f"Error evaluating dataset {dataset_name}: {e}")
        status['evaluation_status'] = 'failed'
        status['evaluation_error'] = str(e)
        return 0.0, status


def calculate_voicebench_scores_with_status(
    results: Dict[str, Any]
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calculate VoiceBench scores using proper evaluators with status tracking.
    
    Returns:
        Tuple of (scores_dict, status_dict)
    """
    scores = {}
    status = {}
    
    # Dataset weights for weighted average
    dataset_weights = {
        'alpacaeval_test': 2.0,
        'commoneval_test': 2.0,
        'wildvoice_test': 2.0,
        'alpacaeval_full_test': 1.0,
        'openbookqa_test': 1.0,
        'mmsu_physics': 1.0,
        'mmsu_biology': 1.0,
        'mmsu_chemistry': 1.0,
        'mmsu_business': 1.0,
        'mmsu_economics': 1.0,
        'mmsu_engineering': 1.0,
        'mmsu_health': 1.0,
        'mmsu_history': 1.0,
        'mmsu_law': 1.0,
        'mmsu_other': 1.0,
        'mmsu_philosophy': 1.0,
        'mmsu_psychology': 1.0,
        'ifeval_test': 1.5,
        'bbh_test': 1.0,
        'advbench_test': 1.0,
    }
    
    weighted_scores = {}
    total_weight = 0.0
    
    for dataset_key, dataset_results in results.items():
        # Evaluate with proper evaluator
        score, dataset_status = evaluate_dataset_with_proper_evaluator(
            dataset_key, dataset_results
        )
        """ Per dataset_split status format:
        dataset_status = {
                'dataset': dataset_name,
                'total_samples': dataset_results.get('total_samples', 0),
                'successful_responses': dataset_results.get('successful_responses', 0),
                'success_rate': dataset_results.get('success_rate', 0.0),
                'evaluator_used': None,
                'evaluation_status': 'pending',
                'evaluation_error': None,
                'evaluation_details': {}, # depends on evaluator
                'score': 0.0
            }
        """
        scores[dataset_key] = float(score)
        status[dataset_key] = dataset_status
        
        # Apply weighting for overall score
        weight = dataset_weights.get(dataset_key, 1.0)
        weighted_scores[dataset_key] = float(score) * weight
        total_weight += weight
    
    # Calculate weighted overall score
    if total_weight > 0:
        scores['overall'] = sum(weighted_scores.values()) / total_weight
    else:
        scores['overall'] = 0.0
    
    # Add overall status summary
    # I need an overall count of total samples, successful responses
    overall_status = {
        'total_datasets': len(results),
        'completed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'completed'),
        'failed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'failed'),
        'total_samples': sum(s.get('total_samples', 0) for s in status.values() if isinstance(s, dict)),
        'successful_responses': sum(s.get('successful_responses', 0) for s in status.values() if isinstance(s, dict)),
        'overall_score': float(scores['overall'])
    }
    status['overall'] = overall_status
    
    return scores, status


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
        # Validate input
        if not isinstance(audio_input, dict):
            bt.logging.error(f"Expected dict but got {type(audio_input)}: {str(audio_input)[:200]}")
            raise ValueError("audio_input must be a dictionary")
        
        if 'array' not in audio_input or 'sampling_rate' not in audio_input:
            raise ValueError("audio_input must contain 'array' and 'sampling_rate' keys")
        
        audio_array = np.array(audio_input['array'])
        sample_rate = audio_input['sampling_rate']
        
        # Validate audio data
        if audio_array.size == 0:
            bt.logging.warning("Empty audio array provided")
            raise ValueError(f"Empty audio array provided")

        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        # Call server inference
        result = self.miner_assistant.inference_v2t(
            audio_array=audio_array,
            sample_rate=sample_rate
        )
        
        # Validate and extract text response
        if not isinstance(result, dict):
            bt.logging.warning(f"Unexpected result type: {type(result)}")
            return ""
        
        text_response = result.get('text')
        
        if not isinstance(text_response, str):
            bt.logging.error(f"Expected string response but got {type(text_response)}: {text_response}")
            raise ValueError("Model response must be a string")
        
        return text_response.strip()
    
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
        
        # Validate input
        if not isinstance(audio_input, dict):
            bt.logging.error(f"Expected dict but got {type(audio_input)}: {str(audio_input)[:200]}")
            raise ValueError("audio_input must be a dictionary")
        
        if 'array' not in audio_input or 'sampling_rate' not in audio_input:
            raise ValueError("audio_input must contain 'array' and 'sampling_rate' keys")
        
        audio_array = np.array(audio_input['array'])
        sample_rate = audio_input['sampling_rate']
        
        # Validate audio data
        if audio_array.size == 0:
            bt.logging.warning("Empty audio array provided")
            raise ValueError(f"Empty audio array provided")

        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        # Call Docker container inference
        result = self.docker_manager.inference_v2t(
            url=self.container_url,
            audio_array=audio_array,
            sample_rate=sample_rate
        )
        
        # Validate and extract text response
        if not isinstance(result, dict):
            bt.logging.warning(f"Unexpected result type: {type(result)}")
            return ""
        
        # Handle voice-to-voice models that may not return text
        text_response = result.get('text')
        
        if not isinstance(text_response, str):
            bt.logging.error(f"Expected string response but got {type(text_response)}: {text_response}")
            raise ValueError("Model response must be a string")
        
        return text_response.strip()
    
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
    
    # All available VoiceBench datasets with their appropriate splits. which datasets to run.
    VOICEBENCH_DATASETS = {
        # 'alpacaeval': ['test'],
        # 'alpacaeval_full': ['test'], 
        'commoneval': ['test'],
        'wildvoice': ['test'],
        # 'openbookqa': ['test'],
        # 'mmsu': ['physics'],  # Use one specific domain instead of 'test'
        # 'sd-qa': ['usa'],  # Using USA dataset only
        # 'mtbench': ['test'],
        'ifeval': ['test'],
        # 'bbh': ['test'],
        'advbench': ['test']
    }
    
    def __init__(self, max_samples_per_dataset: Optional[int] = None):
        """
        Initialize the VoiceBench evaluator.
        
        Args:
            max_samples_per_dataset: Maximum number of samples to evaluate per dataset.
                                    None means evaluate all samples (default).
        """
        self.max_samples_per_dataset = max_samples_per_dataset
        bt.logging.info(f"VoiceBenchEvaluator initialized with max_samples={max_samples_per_dataset or 'all'}")
    
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
    
    def inference_model(
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
            max_samples = SAMPLES_PER_DATASET.get(dataset_name, VOICEBENCH_MAX_SAMPLES)
            for split in splits_to_use:
                try:
                    # Load VoiceBench dataset
                    dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
                    
                    # Apply sample limit if configured and track indices
                    dataset_indices = None
                    if max_samples and len(dataset) > max_samples:
                        # dataset_indices = list(range(max_samples))
                        # random samples
                        dataset_indices = random.sample(range(len(dataset)), max_samples)

                        dataset = dataset.select(dataset_indices)
                        bt.logging.info(f"Limited {dataset_name}_{split} to {max_samples} samples")
                    else:
                        dataset_indices = list(range(len(dataset)))
                    
                    # Run Inference with indices
                    dataset_results = self._inference_dataset(
                        adapter, dataset, dataset_name, split, modality, dataset_indices
                    )
                    
                    results[f"{dataset_name}_{split}"] = dataset_results
                    
                except Exception as e:
                    bt.logging.error(f"Error evaluating {dataset_name}_{split}: {e}")
                    results[f"{dataset_name}_{split}"] = {"error": str(e)}
        
        return results
    
    def _inference_with_adapter(
        self,
        model_adapter,
        datasets: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        modality: str = 'audio'
    ) -> Dict[str, Any]:
        """
        Run inference on model using a pre-created adapter.
        
        This method loads datasets and runs inference to collect model responses.
        The actual evaluation/scoring happens separately in calculate_voicebench_scores_with_status.
        
        Args:
            model_adapter: Pre-configured model adapter (DockerModelAdapter or ServerModelAdapter)
            datasets: List of VoiceBench datasets to run inference on (None = all datasets)
            splits: List of data splits to use (None = use dataset-specific splits)
            modality: Inference modality ('audio', 'text', or 'ttft')
            
        Returns:
            Dictionary containing inference results (responses from the model)
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
            max_samples = SAMPLES_PER_DATASET.get(dataset_name, VOICEBENCH_MAX_SAMPLES)
            for split in splits_to_use:
                try:
                    # Load VoiceBench dataset
                    dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
                    
                    # Apply sample limit if configured and track indices
                    dataset_indices = None
                    if max_samples and len(dataset) > max_samples:
                        # dataset_indices = list(range(max_samples))
                        # random samples
                        dataset_indices = random.sample(range(len(dataset)), max_samples)

                        dataset = dataset.select(dataset_indices)
                        bt.logging.info(f"Limited {dataset_name}_{split} to {max_samples} samples")
                    else:
                        dataset_indices = list(range(len(dataset)))
                    
                    # Run Inference with indices
                    dataset_results = self._inference_dataset(
                        model_adapter, dataset, dataset_name, split, modality, dataset_indices
                    )
                    
                    results[f"{dataset_name}_{split}"] = dataset_results
                    
                except Exception as e:
                    bt.logging.error(f"Error evaluating {dataset_name}_{split}: {e}")
                    results[f"{dataset_name}_{split}"] = {"error": str(e)}
        
        return results
    
    def _inference_dataset(
        self,
        model_adapter: DockerModelAdapter,
        dataset,
        dataset_name: str,
        split: str,
        modality: str,
        dataset_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        This is inference on Miner model.
        
        Args:
            model_adapter: DockerModelAdapter instance
            dataset: Loaded dataset
            dataset_name: Name of the dataset
            split: Data split
            modality: Evaluation modality #TODO: No need for this. Only using for audio-to-text.
            
        Returns:
            Inference results for this dataset
        """
        responses = []
        
        # Generate responses with timeout and retry logic
        for i, item in enumerate(dataset):
            # Get the original HuggingFace dataset index
            hf_index = dataset_indices[i] if dataset_indices else i
            
            max_retries = 2
            retry_delay = 1
            
            if i%50==0:
                bt.logging.info(f"Processing item {i+1}/{len(dataset)} (HF index: {hf_index})")
            
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
                            audio_data = item['audio']
                            # bt.logging.info(f"Audio data type: {type(audio_data)}")
                            # bt.logging.info(f"Audio data keys: {audio_data.keys() if hasattr(audio_data, 'keys') else 'N/A'}")
                            
                            # Log the exact data being passed
                            # if isinstance(audio_data, dict):
                            #     bt.logging.info(f"Audio dict keys: {list(audio_data.keys())}")
                            #     if 'array' in audio_data:
                            #         bt.logging.info(f"Array shape: {audio_data['array'].shape if hasattr(audio_data['array'], 'shape') else 'N/A'}")
                            #     if 'sampling_rate' in audio_data:
                            #         bt.logging.info(f"Sampling rate: {audio_data['sampling_rate']}")
                            
                            response = self._generate_with_timeout(
                                model_adapter.generate_audio, audio_data, timeout=200
                            )
                    # elif modality == 'text':
                    #     response = self._generate_with_timeout(
                    #         model_adapter.generate_text, item['prompt'], timeout=30
                    #     )
                    # elif modality == 'ttft':
                    #     response = self._generate_with_timeout(
                    #         model_adapter.generate_ttft, item['audio'], timeout=30
                    #     )
                    else:
                        raise ValueError(f"Unsupported modality: {modality}")
                    
                    # Validate response
                    if not isinstance(response, str):
                        response = str(response) if response is not None else ""
                    
                    inference_time = time.time() - start_time
                    
                    responses.append({
                        'hf_index': hf_index,  # Add HuggingFace dataset index
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
                    #TODO: Only retry in certain error cases. Not needed for all errors.
                    if attempt < max_retries:
                        bt.logging.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final attempt failed
                        responses.append({
                            'hf_index': hf_index,  # Add HuggingFace dataset index
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
        Run GPT-4 evaluation on generated responses using llm_judge.
        
        Args:
            results_file: Path to results file
            
        Returns:
            GPT evaluation results
        """
        try:
            # Load results from file
            data = []
            with open(results_file, 'r') as f:
                for line in f:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
            
            # Use the llm_judge module to evaluate responses
            bt.logging.info(f"Running LLM evaluation on {len(data)} samples")
            eval_results = evaluate_responses_with_llm(data)
            
            # Save evaluation results
            eval_file = results_file.replace('.jsonl', '_eval.jsonl')
            with open(eval_file, 'w') as f:
                for result in eval_results:
                    f.write(json.dumps(result) + '\n')
            
            return {'gpt_evaluation': eval_results}
                
        except Exception as e:
            bt.logging.error(f"GPT evaluation error: {str(e)}")
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
    timeout: int = 600,
    max_samples_per_dataset: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run VoiceBench evaluation on a miner model API.
    
    Args:
        api_url: URL of the miner model API
        datasets: List of datasets to evaluate (None = all VoiceBench datasets)
        splits: List of splits to evaluate (None = use dataset-specific splits)
        timeout: Request timeout for API calls
        max_samples_per_dataset: Maximum number of samples per dataset (None = all samples)
        
    Returns:
        Complete evaluation results including scores
    """
    evaluator = VoiceBenchEvaluator(max_samples_per_dataset=max_samples_per_dataset)
    
    bt.logging.info("Starting comprehensive VoiceBench evaluation via miner API...")
    
    # Create miner model adapter
    adapter = MinerModelAdapter(api_url=api_url, timeout=timeout)
    # Run model inference on all VoiceBench datasets
    results = evaluator._inference_with_adapter(
        model_adapter=adapter,
        datasets=datasets,  # None means all datasets
        splits=splits,      # None means dataset-specific splits
        modality='audio'
    )
    
    # Calculate scores using proper evaluators with status tracking
    bt.logging.info("Starting VoiceBench evaluation with proper evaluators...")
    scores, status = calculate_voicebench_scores_with_status(results)
    
    bt.logging.info(f"VoiceBench evaluation completed.")
    bt.logging.info(f"Overall score: {scores.get('overall', 0.0):.3f}")
    bt.logging.info(f"Evaluation status: {status.get('overall', {})}")
    
    return {
        'voicebench_scores': scores,
        'evaluation_status': status
    }


def run_voicebench_evaluation(
    container_url: str,
    docker_manager: DockerManager,
    datasets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    max_samples_per_dataset: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run VoiceBench evaluation on a Docker model.
    
    Args:
        container_url: URL of the running Docker container
        docker_manager: DockerManager instance
        datasets: List of datasets to evaluate (None = all VoiceBench datasets)
        splits: List of splits to evaluate (None = use dataset-specific splits)
        max_samples_per_dataset: Maximum number of samples per dataset (None = all samples)
        
    Returns:
            {
            'voicebench_scores': scores,
            'evaluation_status': status.get('overall', {})
            }
    """
    evaluator = VoiceBenchEvaluator(max_samples_per_dataset=max_samples_per_dataset)
    
    bt.logging.info("Starting VoiceBench inference across all datasets...")
    #TODO: create docker adapter here and use inference_with_adapter, no need to two seperate methods.
    # Run model inference on all VoiceBench datasets
    results = evaluator.inference_model(
        container_url=container_url,
        docker_manager=docker_manager,
        datasets=datasets,  # None means all datasets
        splits=splits,      # None means dataset-specific splits
        modality='audio'
    )
    """
    Output format:
    {   datast_split: {
                'dataset': dataset_name,
                'split': split,
                'modality': modality,
                'total_samples': total_responses,
                'successful_responses': successful_responses,
                'success_rate': success_rate,
                'responses': [
                        # in case of successful response
                        {
                            'hf_index':
                            'prompt': item.get('prompt', ''),
                            'response': response,
                            'reference': item.get('output', ''),
                            'inference_time': inference_time,
                            'attempt': attempt + 1
                        }
                        # In case of error
                        { 
                            'hf_index':
                            'prompt': prompt,
                            'response': model_response,
                            'reference': item.get('output', ''), # Reference output in dataset
                            'error': str(e),
                            'failed_attempts': attempt + 1
                        }
                ]
            }
        }
    """
    #TODO: Here we are returning reference key for all, but the reference is present only in some datasets. 
    # And this also affects which prompt is used in llm_judge. Validate this.

    # Calculate scores using proper evaluators with status tracking
    bt.logging.info("Starting VoiceBench evaluation with proper evaluators...")
    scores, status = calculate_voicebench_scores_with_status(results)
    """ status:{
        dataset_split :
                {
                    'dataset': dataset_name,
                    'total_samples': dataset_results.get('total_samples', 0),
                    'successful_responses': dataset_results.get('successful_responses', 0),
                    'success_rate': dataset_results.get('success_rate', 0.0),
                    'evaluator_used': None,
                    'evaluation_status': 'pending',
                    'evaluation_error': None,
                    'evaluation_details': {}, # depends on evaluator
                    'score': 0.0
                },
        
        overall :
            {
                'total_datasets': len(results),
                'completed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'completed'),
                'failed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'failed'),
                'total_samples': sum(s.get('total_samples', 0) for s in status.values() if isinstance(s, dict)),
                'successful_responses': sum(s.get('successful_responses', 0) for s in status.values() if isinstance(s, dict)),
                'overall_score': scores['overall']
            }
        }
    """

    """
    scores: {
        dataset_split: score,
        overall: overall_score
        }
    """

    bt.logging.info(f"VoiceBench evaluation completed.")
    bt.logging.info(f"Overall score: {scores.get('overall', 0.0):.3f}")
    bt.logging.info(f"Evaluation status: {status.get('overall', {})}")
    
    return {
        'voicebench_scores': scores,
        'evaluation_status': status.get('overall', {}),
        'evaluation_details': status  # NEW: Full status dict with sample_details for each dataset
    }