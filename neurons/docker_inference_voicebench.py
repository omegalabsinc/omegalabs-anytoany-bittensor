"""
Hybrid Docker Inference with VoiceBench Integration

This module provides a comprehensive evaluation approach that combines:
1. Traditional S2S metrics (PESQ, WER, MIMI, anti-spoofing) from existing pipeline
2. VoiceBench evaluation across all 11 datasets and 9 task categories

The hybrid approach ensures backward compatibility while adding comprehensive
voice assistant benchmarking capabilities.
"""

import os
import time
import traceback
import tempfile
import shutil
import random
from typing import Dict, Any
import bittensor as bt
from pathlib import Path
from datasets import load_dataset, Audio

# Existing imports
from neurons.docker_manager import DockerManager
from evaluation.S2S.distance import S2SMetrics
from utilities.gpu import log_gpu_memory, cleanup_gpu_memory
from constants import penalty_score, VOICEBENCH_MAX_SAMPLES
from utilities.compare_block_and_model import compare_block_and_model

# VoiceBench integration
from neurons.voicebench_adapter import run_voicebench_evaluation

# Existing dataset import for fallback compatibility
from neurons.docker_inference_v2v import (
    pull_latest_diarization_dataset,
    pull_latest_diarization_dataset_fallback
)



def _verify_hotkey(hf_repo_id: str, hotkey: str, local_dir: str) -> bool:
    """Verify miner hotkey matches repository hotkey."""
    try:
        import huggingface_hub
        target_file_path = huggingface_hub.hf_hub_download(
            repo_id=hf_repo_id,
            filename="hotkey.txt",
            local_dir=Path(local_dir) / hf_repo_id
        )
        with open(target_file_path, 'r') as file:
            hotkey_contents = file.read().strip()
        return hotkey_contents == hotkey
    except Exception as e:
        bt.logging.error(f"Hotkey verification failed: {e}")
        return False


def _compute_s2s_legacy_metrics(container_url: str, docker_manager: DockerManager, hf_repo_id: str) -> Dict[str, Any]:
    """
    Compute legacy S2S metrics for backward compatibility.
    
    This function maintains the original S2S evaluation logic but as a component
    of the hybrid evaluation system.
    """
    # Pull diarization dataset (existing logic)
    mini_batch = pull_latest_diarization_dataset()
    if mini_batch is None:
        bt.logging.info("Pulling fallback diarization dataset")
        mini_batch = pull_latest_diarization_dataset_fallback()
        
    if mini_batch is None:
        bt.logging.error("No diarization dataset available")
        return {"combined_score": penalty_score, "error": "no_dataset"}
    
    # Initialize S2S metrics
    s2s_metrics = S2SMetrics()
    
    # Process samples (simplified version of original logic)
    try:
        total_scores = []
        
        for i in range(min(len(mini_batch['audio']), 3)):  # Limit samples for efficiency
            audio_array = mini_batch['audio'][i]['array']
            sample_rate = mini_batch['audio'][i]['sampling_rate']
            
            # Run inference through Docker container
            result = docker_manager.inference_v2v(
                url=container_url,
                audio_array=audio_array,
                sample_rate=sample_rate
            )
            
            if 'audio' in result:
                # Compute S2S metrics
                metrics_dict = s2s_metrics.compute_distance(
                    gt_audio_arrs=[[audio_array, sample_rate]],
                    generated_audio_arrs=[[result['audio'], sample_rate]]
                )
                
                # Extract score (simplified scoring logic)
                score = metrics_dict.get('combined_score', 0.0)
                total_scores.append(score)
        
        # Calculate average score
        if total_scores:
            combined_score = sum(total_scores) / len(total_scores)
        else:
            combined_score = 0.0
            
        return {
            "combined_score": combined_score,
            "sample_count": len(total_scores),
            "individual_scores": total_scores
        }
        
    except Exception as e:
        bt.logging.error(f"S2S metrics computation failed: {e}")
        return {"combined_score": penalty_score, "error": str(e)}


def run_voicebench_scoring(
    hf_repo_id: str,
    hotkey: str,
    block: int,
    model_tracker,
    local_dir: str,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Main scoring function that replaces run_v2v_scoring with VoiceBench evaluation.
    
    This function provides the same interface as the original run_v2v_scoring
    but uses comprehensive VoiceBench evaluation instead of S2S metrics.
    """
    if not compare_block_and_model(block, hf_repo_id):
        bt.logging.info(f"Block {block} is older than model {hf_repo_id}. Penalizing model.")
        return {"combined_score": penalty_score}
    
    log_gpu_memory('before hybrid evaluation')
    
    # Initialize Docker manager
    docker_manager = DockerManager(base_cache_dir=local_dir)
    container_url = None
    
    try:
        # Sanitize repository name for Docker
        sanitized_name = hf_repo_id.split('/')[-1].lower()
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in sanitized_name)
        container_uid = f"{sanitized_name}_{int(time.time())}"
        
        # Start Docker container
        container_url = docker_manager.start_container(
            uid=container_uid,
            repo_id=hf_repo_id,
            gpu_id=0 if device == 'cuda' else None
        )
        
        bt.logging.info(f"Started container {container_uid} for hybrid evaluation of {hf_repo_id}")
        
        # Verify hotkey if provided
        if hotkey is not None:
            if not _verify_hotkey(hf_repo_id, hotkey, local_dir):
                bt.logging.warning(f"Hotkey verification failed for {hf_repo_id}")
                return {"combined_score": penalty_score, "error": "hotkey_mismatch"}
        
        results = {
            "hf_repo_id": hf_repo_id,
            "hotkey": hotkey,
            "block": block
        }
        
        # Run VoiceBench evaluation
        voicebench_score = 0.0
        bt.logging.info("Running comprehensive VoiceBench evaluation...")
        # Get max samples from environment or use default
        max_samples = VOICEBENCH_MAX_SAMPLES
        
        voicebench_results = run_voicebench_evaluation(
            container_url=container_url,
            docker_manager=docker_manager,
            datasets=None,  # Use all datasets
            splits=None,    # Use dataset-specific splits
            max_samples_per_dataset=max_samples
        )
        
        voicebench_score = voicebench_results['voicebench_scores'].get('overall', 0.0)
        results['voicebench_scores'] = voicebench_results['voicebench_scores']
        results['evaluation_status'] = voicebench_results['evaluation_status']
        results['evaluation_details'] = voicebench_results.get('evaluation_details', {})  # NEW: Include full details with sample_details
        
        bt.logging.info(f"VoiceBench evaluation completed. Score: {voicebench_score:.3f}")

        # Run Voice MOS evaluation
        bt.logging.info("Running Voice MOS evaluation...")
        voice_mos_results = run_voice_mos_evaluation(
            container_url=container_url,
            docker_manager=docker_manager,
            max_samples_per_dataset=max_samples
        )

        voice_mos_score = voice_mos_results.get('voice_mos_score', 0.0)
        results['voice_mos_score'] = voice_mos_score
        results['voice_mos_details'] = voice_mos_results.get('voice_mos_details', {})

        bt.logging.info(f"Voice MOS evaluation completed. Score: {voice_mos_score:.3f}")

        # Combine scores (configurable weights)
        voicebench_weight = 0.7  # 70% text accuracy
        voice_mos_weight = 0.3   # 30% voice quality
        combined_score = (voicebench_score * voicebench_weight) + (voice_mos_score * voice_mos_weight)

        results['combined_score'] = combined_score
        results['raw_scores'] = {
            'voicebench': voicebench_score,
            'voice_mos': voice_mos_score
        }

        bt.logging.info(f"Combined score: {combined_score:.3f} (VB: {voicebench_score:.3f}, MOS: {voice_mos_score:.3f})")
        bt.logging.info(f"Evaluation completed for {hf_repo_id}")

        return results
        
    except Exception as e:
        bt.logging.error(f"Error in hybrid evaluation for {hf_repo_id}: {e}")
        bt.logging.error(traceback.format_exc())
        raise e
        
    finally:
        # Cleanup
        try:
            if container_url and docker_manager:
                docker_manager.stop_container(container_uid)
                bt.logging.info(f"Stopped container {container_uid}")
        except Exception as e:
            bt.logging.error(f"Error cleaning up container: {e}")
        
        try:
            docker_manager.cleanup_docker_resources()
        except Exception as e:
            bt.logging.error(f"Error cleaning up Docker resources: {e}")
            
        cleanup_gpu_memory()
        log_gpu_memory('after hybrid evaluation cleanup')


def run_voice_mos_evaluation(
    container_url: str,
    docker_manager: DockerManager,
    max_samples_per_dataset: int = 10
) -> Dict[str, Any]:
    """
    Run voice quality evaluation using v2v endpoint and MOS scoring.
    """
    temp_audio_dir = None
    try:
        # Create temp directory for audio files
        temp_audio_dir = tempfile.mkdtemp(prefix="voice_mos_")
        bt.logging.info(f"Created temp audio directory: {temp_audio_dir}")

        # Inference phase - call v2v endpoint, save audio files
        inference_results = inference_voice_mos(
            container_url=container_url,
            docker_manager=docker_manager,
            temp_audio_dir=temp_audio_dir,
            max_samples_per_dataset=max_samples_per_dataset
        )

        # Scoring phase - get MOS score from saved audio
        mos_score = get_mos_score(temp_audio_dir)

        return {
            'voice_mos_score': mos_score,
            'voice_mos_details': {
                'samples_processed': inference_results['total_samples'],
                'datasets_evaluated': inference_results['datasets'],
                'inference_success_rate': inference_results['success_rate'],
                'sample_details': inference_results['sample_details']
            }
        }

    except Exception as e:
        bt.logging.error(f"Voice MOS evaluation failed: {e}")
        bt.logging.error(traceback.format_exc())
        return {
            'voice_mos_score': 0.0,
            'voice_mos_details': {
                'error': str(e),
                'samples_processed': 0,
                'datasets_evaluated': [],
                'inference_success_rate': 0.0,
                'sample_details': []
            }
        }

    finally:
        # Ensure cleanup even on errors
        if temp_audio_dir and os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)
            bt.logging.info(f"Cleaned up temp audio directory: {temp_audio_dir}")


def inference_voice_mos(
    container_url: str,
    docker_manager: DockerManager,
    temp_audio_dir: str,
    max_samples_per_dataset: int
) -> Dict[str, Any]:
    """
    Call v2v endpoint on VoiceBench samples and save audio outputs.
    """
    # VoiceBench datasets (same as used in voicebench_adapter.py)
    VOICEBENCH_DATASETS = {
        'commoneval': ['test'],
        'wildvoice': ['test'],
        'ifeval': ['test'],
        'advbench': ['test']
    }

    total_samples = 0
    success_count = 0
    sample_details = []
    datasets_evaluated = []

    for dataset_name, splits in VOICEBENCH_DATASETS.items():
        bt.logging.info(f"Processing dataset: {dataset_name}")
        datasets_evaluated.append(dataset_name)

        for split in splits:
            # Load dataset
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            # Get random samples
            dataset_size = len(dataset)
            sample_count = min(max_samples_per_dataset, dataset_size)
            sample_indices = random.sample(range(dataset_size), sample_count)

            for sample_idx in sample_indices:
                total_samples += 1
                sample = dataset[sample_idx]

                # Get audio data
                audio_data = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']

                # Call v2v endpoint
                start_time = time.time()
                response = docker_manager.inference_v2v(
                    url=container_url,
                    audio_array=audio_data,
                    sample_rate=sample_rate,
                    timeout=30
                )
                inference_time = time.time() - start_time

                # Check if we got audio response
                if 'audio_wav_bytes' in response and response['audio_wav_bytes'] is not None:
                    # Save audio file with flat naming
                    filename = f"{dataset_name}_{split}_{sample_idx:04d}.wav"
                    audio_path = os.path.join(temp_audio_dir, filename)

                    # Save WAV bytes directly to file
                    with open(audio_path, 'wb') as f:
                        f.write(response['audio_wav_bytes'])

                    success_count += 1
                    bt.logging.debug(f"Saved audio: {filename}")

                    sample_details.append({
                        'dataset': dataset_name,
                        'split': split,
                        'sample_index': sample_idx,
                        'filename': filename,
                        'inference_time': inference_time,
                        'success': True
                    })

                else:
                    bt.logging.warning(f"No audio in response for {dataset_name}_{sample_idx}")
                    sample_details.append({
                        'dataset': dataset_name,
                        'split': split,
                        'sample_index': sample_idx,
                        'filename': None,
                        'inference_time': inference_time,
                        'success': False,
                        'error': 'No audio in response'
                    })

    success_rate = success_count / total_samples if total_samples > 0 else 0.0
    bt.logging.info(f"Voice MOS inference completed: {success_count}/{total_samples} samples successful ({success_rate:.2%})")

    return {
        'total_samples': total_samples,
        'success_count': success_count,
        'success_rate': success_rate,
        'datasets': datasets_evaluated,
        'sample_details': sample_details
    }


def get_mos_score(temp_dir_path: str) -> float:
    """
    Calculate MOS score using UTMOS-v2 model.

    Args:
        temp_dir_path: Path to directory containing audio files

    Returns:
        Average MOS score across all audio files
    """
    import utmosv2

    # List audio files
    audio_files = [f for f in os.listdir(temp_dir_path) if f.endswith('.wav')]
    bt.logging.info(f"Found {len(audio_files)} audio files for MOS scoring")

    if len(audio_files) == 0:
        bt.logging.warning("No audio files found for MOS scoring")
        return 0.0

    # Load model (fresh each time to free GPU for other tasks)
    start_time = time.time()
    bt.logging.info("Loading UTMOS-v2 model...")
    model = utmosv2.create_model(pretrained=True)
    model_load_time = time.time() - start_time
    bt.logging.info(f"UTMOS-v2 model loaded in {model_load_time:.2f} seconds")

    # Run MOS scoring
    start_time = time.time()
    bt.logging.info(f"Running MOS scoring on {len(audio_files)} files...")
    mos_results = model.predict(input_dir=temp_dir_path)
    scoring_time = time.time() - start_time
    bt.logging.info(f"MOS scoring completed in {scoring_time:.2f} seconds")

    # Calculate average MOS score
    mos_scores = [result['predicted_mos'] for result in mos_results]
    average_mos = sum(mos_scores) / len(mos_scores)

    bt.logging.info(f"MOS scoring results:")
    bt.logging.info(f"  Files processed: {len(mos_results)}")
    bt.logging.info(f"  Average MOS: {average_mos:.3f}")
    bt.logging.info(f"  MOS range: {min(mos_scores):.3f} - {max(mos_scores):.3f}")
    bt.logging.info(f"  Total time: {model_load_time + scoring_time:.2f}s (load: {model_load_time:.2f}s, scoring: {scoring_time:.2f}s)")

    # Log individual file scores for debugging
    for result in mos_results[:3]:  # Log first 3 files
        bt.logging.debug(f"  {result['file_path']}: {result['predicted_mos']:.3f}")

    # scale MOS to 0-1 range
    average_mos = average_mos/5.0
    bt.logging.info(f"Scaled Average MOS (0-1): {average_mos:.3f}")
    return average_mos


if __name__ == "__main__":
    # Test the hybrid evaluation
    test_repos = ["eggmoo/omega_gQdQiVq", "tezuesh/moshi_general"]
    
    for hf_repo_id in test_repos:
        bt.logging.info(f"Testing hybrid evaluation on {hf_repo_id}")
        results = run_voicebench_scoring(
            hf_repo_id=hf_repo_id,
            hotkey=None,
            block=5268488,
            model_tracker=None,
            local_dir="./model_cache"
        )
        
        print(f"Results for {hf_repo_id}:")
        print(f"  Combined Score: {results.get('combined_score', 0.0):.3f}")
        print(f"  VoiceBench Score: {results.get('voicebench_score', 0.0):.3f}")
        
        if 'voicebench_scores' in results:
            print("  Dataset Scores:")
            for dataset, score in results['voicebench_scores'].items():
                if dataset != 'overall':
                    print(f"    {dataset}: {score:.3f}")
        print()