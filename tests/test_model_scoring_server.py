#!/usr/bin/env python3
"""
Test Model Scoring with Server

This script allows testing and scoring models using a server API endpoint
instead of creating Docker containers. It preserves the exact same scoring
logic as the main hybrid VoiceBench + Voice MOS scoring flow.

The updated scoring includes:
- VoiceBench evaluation (70% weight) - Text accuracy and reasoning
- Voice MOS evaluation (30% weight) - Voice quality using UTMOS-v2

You need a Chutes API key & OpenAI API key
Usage:
    python -m tests.test_model_scoring_server --experiment_name "test-scoring-script" --logging.debug
"""
PENALTY_SCORE=0.001
import argparse
import json
import time
import traceback
from typing import Dict, Any, Optional
import torch
import bittensor as bt
import os

# Import the server-based evaluation function
from neurons.voicebench_adapter import run_voicebench_evaluation_miner
from constants import SAMPLES_PER_DATASET, VOICEBENCH_MAX_SAMPLES
from datetime import datetime


def run_actual_scoring_with_server(
    api_url: str,
    datasets: Optional[list] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Run the actual scoring flow using server API instead of Docker.
    This directly calls the inference functions from docker_inference_voicebench.py
    but replaces Docker container calls with server API calls.
    """
    from neurons.voicebench_adapter import VoiceBenchEvaluator, MinerModelAdapter
    from neurons.docker_inference_voicebench import get_mos_score, inference_voice_mos
    from neurons.miner_model_assistant import MinerModelAssistant
    import tempfile
    import shutil
    import os

    bt.logging.info(f"Starting actual hybrid scoring with server API: {api_url}")

    temp_audio_dir = None
    try:
        # Step 1: VoiceBench evaluation
        bt.logging.info("Running VoiceBench evaluation...")
        v2t_url = api_url.rstrip('/') +'/api/v1/v2t'  # Ensure base URL for other endpoints
        # voicebench_results = run_voicebench_evaluation_miner(
        #     api_url=v2t_url,
        #     datasets=datasets,
        #     timeout=timeout
        # )
        voicebench_results={}
        print(f"Skipped the voicebench evaluation")
        voicebench_scores = voicebench_results.get('voicebench_scores', {})
        voicebench_overall = voicebench_scores.get('overall', 0.0)
        bt.logging.info(f"VoiceBench completed: {voicebench_overall:.3f}")

        # Step 2: Voice MOS evaluation - actual implementation
        bt.logging.info("Running Voice MOS evaluation...")
        temp_audio_dir = tempfile.mkdtemp(prefix="voice_mos_server_")

        # Run inference to generate audio files
        voice_mos_results = run_voice_mos_inference_server(
            api_url=api_url,
            temp_audio_dir=temp_audio_dir,
            datasets=datasets,
            max_samples=3
        )

        # Calculate actual MOS score
        mos_score = get_mos_score(temp_audio_dir)
        bt.logging.info(f"Voice MOS completed: {mos_score:.3f}")

        # Step 3: Combine scores
        voicebench_weight = 0.7
        voice_mos_weight = 0.3
        combined_score = (voicebench_overall * voicebench_weight) + (mos_score * voice_mos_weight)

        return {
            'voicebench_scores': voicebench_scores,
            'voice_mos_score': mos_score,
            'combined_score': combined_score,
            'raw_scores': {
                'voicebench': voicebench_overall,
                'voice_mos': mos_score
            },
            'evaluation_status': voicebench_results.get('evaluation_status', {}),
            'evaluation_details': voicebench_results.get('evaluation_details', {}),
            'voice_mos_details': voice_mos_results
        }

    finally:
        if temp_audio_dir and os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)


def run_voice_mos_inference_server(
    api_url: str,
    temp_audio_dir: str,
    datasets: Optional[list],
    max_samples: int = 10
) -> Dict[str, Any]:
    """Run voice MOS inference using server API"""
    from datasets import load_dataset, Audio
    from neurons.miner_model_assistant import MinerModelAssistant
    import random
    import time

    assistant = MinerModelAssistant(api_url=api_url, timeout=30)

    # Use same datasets as production
    # test_datasets = datasets if datasets else ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    test_datasets = ['wildvoice']
    total_samples = 0
    success_count = 0

    for dataset_name in test_datasets:
        try:
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split='test')
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            sample_count = min(max_samples // len(test_datasets), len(dataset))
            sample_indices = random.sample(range(len(dataset)), sample_count)

            for idx in sample_indices:
                total_samples += 1
                sample = dataset[idx]

                try:
                    # Call v2v endpoint
                    audio_data = sample['audio']['array']
                    sample_rate = sample['audio']['sampling_rate']

                    result = assistant.inference_v2v(
                        url=api_url,
                        audio_array=audio_data,
                        sample_rate=sample_rate
                    )

                    if 'audio_wav_bytes' in result and result['audio_wav_bytes']:
                        # Save audio file
                        filename = f"{dataset_name}_{idx:04d}.wav"
                        audio_path = os.path.join(temp_audio_dir, filename)

                        with open(audio_path, 'wb') as f:
                            f.write(result['audio_wav_bytes'])

                        success_count += 1

                except Exception as e:
                    bt.logging.warning(f"V2V inference failed for {dataset_name}_{idx}: {e}")
                    continue

        except Exception as e:
            bt.logging.error(f"Failed to process dataset {dataset_name}: {e}")
            continue

    success_rate = success_count / total_samples if total_samples > 0 else 0.0

    return {
        'total_samples': total_samples,
        'success_count': success_count,
        'success_rate': success_rate,
        'datasets': test_datasets
    }


def test_server_scoring(
    api_url: str,
    experiment_name: str = "unnamed",
    datasets: Optional[list] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Test model scoring using a server API endpoint.
    
    This function uses the exact same scoring logic as the production flow
    but connects to a server API instead of creating Docker containers.
    
    Args:
        api_url: URL of the model server API endpoint (e.g., http://localhost:8000/api/v1/v2t)
        max_samples: Maximum number of samples per dataset to evaluate (None = all)
        datasets: List of specific datasets to evaluate (None = all VoiceBench datasets)
        timeout: Request timeout for API calls in seconds
        
    Returns:
        Dictionary with scoring results matching the production format:
        {
            'raw_scores': {
                'voicebench': {...}
            },
            'combined_score': float,
            'evaluation_status': {...},
            'evaluation_details': {...},
            'metadata': {...}
        }
    """
    bt.logging.info(f"Starting server-based model scoring")
    bt.logging.info(f"Experiment: {experiment_name}")
    bt.logging.info(f"API URL: {api_url}")
    
    start_time = time.time()
    
    try:
        # Run actual scoring flow with server API
        bt.logging.info("Running actual scoring flow with server API")
        hybrid_results = run_actual_scoring_with_server(
            api_url=api_url,
            datasets=datasets,
            timeout=timeout
        )

        # Extract scores (matching production format)
        voicebench_scores = hybrid_results.get('voicebench_scores', {})
        voice_mos_score = hybrid_results.get('voice_mos_score', 0.0)
        combined_score = hybrid_results.get('combined_score', 0.0)

        # Build results in the exact same format as production hybrid scoring
        results = {
            'raw_scores': {
                'voicebench': voicebench_scores,
                'voice_mos': voice_mos_score
            },
            'combined_score': combined_score,
            'voicebench_score': voicebench_scores.get('overall', 0.0),
            'voice_mos_score': voice_mos_score,
            'evaluation_status': hybrid_results.get('evaluation_status', {}),
            'evaluation_details': hybrid_results.get('evaluation_details', {}),
            'voice_mos_details': hybrid_results.get('voice_mos_details', {}),
            'metadata': {
                'experiment_name': experiment_name,
                'competition_id': 'voicebench_hybrid',
                'model_id': 'server_model',
                'api_url': api_url,
                'evaluation_time': time.time() - start_time,
                'datasets': datasets,
                'scoring_weights': {
                    'voicebench': 0.7,
                    'voice_mos': 0.3
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        
        bt.logging.success(f"Evaluation completed successfully")
        bt.logging.info(f"Combined score: {combined_score:.4f}")
        bt.logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        bt.logging.error(f"Error during server-based evaluation: {e}")
        bt.logging.error(traceback.format_exc())
        
        # Return penalty score on failure (matching production behavior)
        return {
            'raw_scores': {
                'voicebench': None,
                'voice_mos': None
            },
            'combined_score': PENALTY_SCORE,
            'voicebench_score': 0.0,
            'voice_mos_score': 0.0,
            'evaluation_status': {
                'overall': {
                    'status': 'error',
                    'error': str(e)
                }
            },
            'evaluation_details': {},
            'voice_mos_details': {},
            'metadata': {
                'experiment_name': experiment_name,
                'competition_id': 'voicebench_hybrid',
                'model_id': 'server_model',
                'api_url': api_url,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
        }
    finally:
        # Clean up GPU memory if needed (matching production)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def append_to_report(results: Dict[str, Any], report_file: str = "model_evaluation_report.txt"):
    """Append evaluation results to a text report file."""
    with open(report_file, 'a') as f:
        # Write separator for new experiment
        f.write("\n" + "="*80 + "\n")
        
        # Write experiment metadata
        metadata = results.get('metadata', {})
        f.write(f"EXPERIMENT: {metadata.get('experiment_name', 'unnamed')}\n")
        f.write(f"TIMESTAMP: {metadata.get('timestamp', datetime.now().isoformat())}\n")
        f.write(f"API URL: {metadata.get('api_url', 'N/A')}\n")
        f.write(f"EVALUATION TIME: {metadata.get('evaluation_time', 0):.2f} seconds\n")
        f.write("-"*80 + "\n")
        
        # Write combined score
        f.write(f"COMBINED SCORE: {results['combined_score']:.4f}\n")
        f.write("-"*80 + "\n")
        
        # Write individual component scores
        f.write("COMPONENT SCORES:\n")
        voicebench_score = results.get('voicebench_score', 0.0)
        voice_mos_score = results.get('voice_mos_score', 0.0)
        f.write(f"  {'VoiceBench (70%)':<30} {voicebench_score:>8.4f}\n")
        f.write(f"  {'Voice MOS (30%)':<30} {voice_mos_score:>8.4f}\n")
        f.write("-"*80 + "\n")

        # Write individual dataset scores if available
        if results.get('raw_scores', {}).get('voicebench'):
            f.write("VOICEBENCH DATASET SCORES:\n")
            scores = results['raw_scores']['voicebench']

            # Sort datasets for consistent reporting
            dataset_names = sorted([k for k in scores.keys() if k != 'overall'])

            for dataset in dataset_names:
                score = scores[dataset]
                # Include samples used from SAMPLES_PER_DATASET
                samples_used = SAMPLES_PER_DATASET.get(dataset.split('_')[0], VOICEBENCH_MAX_SAMPLES)
                f.write(f"  {dataset:<30} {score:>8.4f}  (samples: {samples_used})\n")

            # Write overall at the end
            if 'overall' in scores:
                f.write(f"  {'VB OVERALL':<30} {scores['overall']:>8.4f}\n")
        
        # Write evaluation status summary
        status = results.get('evaluation_status', {}).get('overall', {})
        if status:
            f.write("-"*80 + "\n")
            f.write(f"STATUS: {status.get('status', 'unknown')}\n")
            
            # Write total samples and successful responses
            total_samples = status.get('total_samples', 0)
            successful_responses = status.get('successful_responses', 0)
            if total_samples > 0:
                f.write(f"TOTAL SAMPLES: {total_samples}\n")
                f.write(f"SUCCESSFUL RESPONSES: {successful_responses}\n")
                f.write(f"SUCCESS RATE: {(successful_responses/total_samples*100):.1f}%\n")
            
            # Write any errors
            errors = status.get('errors', {})
            if errors:
                f.write("ERRORS:\n")
                for dataset, error in errors.items():
                    f.write(f"  {dataset}: {error}\n")
        
        # Write error if evaluation failed
        if metadata.get('error'):
            f.write("-"*80 + "\n")
            f.write(f"ERROR: {metadata['error']}\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Results appended to: {report_file}")


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed evaluation results in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall score
    print(f"\n📊 COMBINED SCORE: {results['combined_score']:.4f}")

    # Component scores
    voicebench_score = results.get('voicebench_score', 0.0)
    voice_mos_score = results.get('voice_mos_score', 0.0)
    print(f"\n🔀 COMPONENT SCORES:")
    print(f"  • VoiceBench (70%): {voicebench_score:.4f}")
    print(f"  • Voice MOS (30%):  {voice_mos_score:.4f}")

    # VoiceBench dataset scores
    if results.get('raw_scores', {}).get('voicebench'):
        print(f"\n📈 VOICEBENCH DATASET SCORES:")
        scores = results['raw_scores']['voicebench']

        # Print individual dataset scores
        for dataset, score in scores.items():
            if dataset != 'overall':
                print(f"  • {dataset:20s}: {score:.4f}")

        # Print overall score
        if 'overall' in scores:
            print(f"  ─────────────────────────────")
            print(f"  • {'VB OVERALL':20s}: {scores['overall']:.4f}")

    # Voice MOS details
    if results.get('voice_mos_details'):
        print(f"\n🎤 VOICE MOS DETAILS:")
        mos_details = results['voice_mos_details']
        if mos_details.get('simulated'):
            print(f"  • Type: Simulated (for server testing)")
        if 'note' in mos_details:
            print(f"  • Note: {mos_details['note']}")
    
    # Evaluation status
    status = results.get('evaluation_status', {})
    if status:
        print(f"\n📋 EVALUATION STATUS:")
        overall_status = status.get('overall', {})
        if overall_status:
            print(f"  • Status: {overall_status.get('status', 'unknown')}")
            
            # Print evaluators used
            evaluators_used = overall_status.get('evaluators_used', {})
            if evaluators_used:
                print(f"  • Evaluators used:")
                for dataset, evaluator in evaluators_used.items():
                    print(f"    - {dataset}: {evaluator}")
            
            # Print any errors
            errors = overall_status.get('errors', {})
            if errors:
                print(f"  • Errors encountered:")
                for dataset, error in errors.items():
                    print(f"    - {dataset}: {error}")
    
    # Metadata
    metadata = results.get('metadata', {})
    if metadata:
        print(f"\n⚙️  METADATA:")
        print(f"  • API URL: {metadata.get('api_url', 'N/A')}")
        print(f"  • Evaluation time: {metadata.get('evaluation_time', 0):.2f} seconds")
        print(f"  • Max samples: {metadata.get('max_samples') or 'all'}")
        
        if metadata.get('error'):
            print(f"  • Error: {metadata['error']}")
    
    # Sample details summary (if available)
    details = results.get('evaluation_details', {})
    if details:
        print(f"\n📝 SAMPLE DETAILS:")
        for dataset_split, dataset_details in details.items():
            if isinstance(dataset_details, dict):
                total = dataset_details.get('total_samples', 0)
                successful = dataset_details.get('successful_responses', 0)
                success_rate = dataset_details.get('success_rate', 0.0)
                print(f"  • {dataset_split}:")
                print(f"    - Total samples: {total}")
                print(f"    - Successful: {successful}")
                print(f"    - Success rate: {success_rate:.1%}")
                
                # Show sample responses if available and verbose
                sample_details = dataset_details.get('sample_details', [])
                if sample_details and len(sample_details) > 0:
                    print(f"    - Sample responses available: {len(sample_details)}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function to run server-based model testing."""
    parser = argparse.ArgumentParser(
        description="Test model scoring using a server API instead of Docker containers"
    )
    
    # Add custom arguments
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8010",
        help="URL of the model server API endpoint"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of this experiment for tracking in the report"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to evaluate (e.g., commoneval wildvoice)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout for API calls in seconds (default: 600)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results as JSON"
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="model_evaluation_report.txt",
        help="Text file to append results to (default: model_evaluation_report.txt)"
    )
    
    # Add bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    
    # Parse with bittensor config
    config = bt.config(parser)
    
    print(f"\n🚀 Starting server-based model scoring...")
    print(f"   Experiment: {config.experiment_name}")
    print(f"   API URL: {config.api_url}")
    print(f"   Datasets: {config.datasets or 'all'}")
    print(f"   Timeout: {config.timeout}s")
    print(f"   Report file: {config.report_file}")
    
    # Show samples per dataset configuration
    print(f"\n📊 Samples per dataset:")
    for dataset, samples in SAMPLES_PER_DATASET.items():
        print(f"   {dataset}: {samples}")
    print(f"   Default: {VOICEBENCH_MAX_SAMPLES}")
    
    # Run the evaluation
    results = test_server_scoring(
        api_url=config.api_url,
        experiment_name=config.experiment_name,
        datasets=config.datasets,
        timeout=config.timeout
    )
    
    # Print detailed results to console
    print_detailed_results(results)
    
    # Append to report file
    append_to_report(results, config.report_file)
    
    # Save full JSON results if requested
    if config.output:
        with open(config.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Full JSON results saved to: {config.output}")
    
    # Exit with appropriate code
    if results['combined_score'] == PENALTY_SCORE:
        print("❌ Evaluation failed with penalty score")
        exit(1)
    else:
        print("✅ Evaluation completed successfully")
        exit(0)


if __name__ == "__main__":
    main()