"""
Runs voicebench evaluation
Runs voice mos evaluation
"""

import os
import time
import traceback
from typing import Dict, Any
import bittensor as bt
from pathlib import Path

# Existing imports
from neurons.docker_manager import DockerManager
from utilities.gpu import log_gpu_memory, cleanup_gpu_memory
from constants import penalty_score, VOICEBENCH_WEIGHT, VOICE_MOS_WEIGHT
from utilities.compare_block_and_model import compare_block_and_model

# scorings
from neurons.voicebench_adapter import run_voicebench_evaluation
from neurons.mos_scoring import run_voice_mos_evaluation


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


def run_full_scoring(
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
        
        voicebench_results = run_voicebench_evaluation(
            container_url=container_url
        )
        
        voicebench_score = voicebench_results['voicebench_scores'].get('overall', 0.0)
        results['voicebench_scores'] = voicebench_results['voicebench_scores']
        results['evaluation_status'] = voicebench_results['evaluation_status']
        results['evaluation_details'] = voicebench_results.get('evaluation_details', {})  # NEW: Include full details with sample_details
        
        bt.logging.info(f"VoiceBench evaluation completed. Score: {voicebench_score:.3f}")

        # Run Voice MOS evaluation
        bt.logging.info("Running Voice MOS evaluation...")
        voice_mos_results = run_voice_mos_evaluation(
            container_url=container_url
        )

        voice_mos_score = voice_mos_results.get('voice_mos_score', 0.0)
        results['voice_mos_score'] = voice_mos_score
        results['voice_mos_details'] = voice_mos_results.get('voice_mos_details', {})

        bt.logging.info(f"Voice MOS evaluation completed. Score: {voice_mos_score:.3f}")

        combined_score = (voicebench_score * VOICEBENCH_WEIGHT) + (voice_mos_score * VOICE_MOS_WEIGHT)

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


if __name__ == "__main__":
    # Test the hybrid evaluation
    test_repos = ["eggmoo/omega_gQdQiVq", "tezuesh/moshi_general"]
    
    for hf_repo_id in test_repos:
        bt.logging.info(f"Testing hybrid evaluation on {hf_repo_id}")
        results = run_full_scoring(
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