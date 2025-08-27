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
from typing import Optional, Dict, Any
import bittensor as bt
from pathlib import Path

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
        
        results['combined_score'] = voicebench_score
        
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