#!/usr/bin/env python3
"""
Test Model Scoring with Server

Simplified script that tests models using a server API endpoint.
Uses the same evaluation methods as docker_inference_voicebench.py
but with a server URL instead of Docker containers.

Usage:
    python -m tests.test_model_scoring_server --api_url http://localhost:8010
"""

import argparse
import json
import time
import traceback
from typing import Dict, Any
import bittensor as bt

# Import the refactored evaluation functions
from neurons.voicebench_adapter import run_voicebench_evaluation
from neurons.mos_scoring import run_voice_mos_evaluation
from constants import VOICEBENCH_WEIGHT, VOICE_MOS_WEIGHT


def test_scoring_with_server(api_url: str) -> Dict[str, Any]:
    """
    Test scoring using server API with the refactored evaluation methods.
    Uses the same methods as docker_inference_voicebench.py but with server URL.
    """
    bt.logging.info(f"Starting server-based scoring: {api_url}")

    results = {}

    try:
        # Run VoiceBench evaluation
        bt.logging.info("Running VoiceBench evaluation...")
        voicebench_results = run_voicebench_evaluation(container_url=api_url)

        voicebench_score = voicebench_results['voicebench_scores'].get('overall', 0.0)
        results['voicebench_scores'] = voicebench_results['voicebench_scores']
        results['evaluation_status'] = voicebench_results['evaluation_status']
        results['evaluation_details'] = voicebench_results.get('evaluation_details', {})

        bt.logging.info(f"VoiceBench evaluation completed. Score: {voicebench_score:.3f}")

        # Run Voice MOS evaluation
        bt.logging.info("Running Voice MOS evaluation...")
        voice_mos_results = run_voice_mos_evaluation(container_url=api_url)

        voice_mos_score = voice_mos_results.get('voice_mos_score', 0.0)
        results['voice_mos_score'] = voice_mos_score
        results['voice_mos_details'] = voice_mos_results.get('voice_mos_details', {})

        bt.logging.info(f"Voice MOS evaluation completed. Score: {voice_mos_score:.3f}")

        # Calculate combined score
        combined_score = (voicebench_score * VOICEBENCH_WEIGHT) + (voice_mos_score * VOICE_MOS_WEIGHT)

        results['combined_score'] = combined_score
        results['raw_scores'] = {
            'voicebench': voicebench_score,
            'voice_mos': voice_mos_score
        }

        bt.logging.info(f"Combined score: {combined_score:.3f} (VB: {voicebench_score:.3f}, MOS: {voice_mos_score:.3f})")

        return results

    except Exception as e:
        bt.logging.error(f"Error in server evaluation: {e}")
        bt.logging.error(traceback.format_exc())
        raise e


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Overall score
    print(f"\nCOMBINED SCORE: {results['combined_score']:.4f}")

    # Component scores
    voicebench_score = results['raw_scores']['voicebench']
    voice_mos_score = results['raw_scores']['voice_mos']
    print(f"\nCOMPONENT SCORES:")
    print(f"  VoiceBench ({VOICEBENCH_WEIGHT:.0%}): {voicebench_score:.4f}")
    print(f"  Voice MOS ({VOICE_MOS_WEIGHT:.0%}):   {voice_mos_score:.4f}")

    # VoiceBench dataset scores
    if 'voicebench_scores' in results:
        print(f"\nVOICEBENCH DATASET SCORES:")
        scores = results['voicebench_scores']
        for dataset, score in scores.items():
            if dataset != 'overall':
                print(f"  {dataset:20s}: {score:.4f}")
        if 'overall' in scores:
            print(f"  {'VB OVERALL':20s}: {scores['overall']:.4f}")

    # Evaluation status
    if 'evaluation_status' in results:
        status = results['evaluation_status']
        print(f"\nEVALUATION STATUS:")
        if 'overall' in status:
            overall = status['overall']
            print(f"  Status: {overall.get('status', 'unknown')}")
            if 'total_samples' in overall:
                print(f"  Total samples: {overall['total_samples']}")
            if 'successful_responses' in overall:
                print(f"  Successful responses: {overall['successful_responses']}")

    # Voice MOS details
    if 'voice_mos_details' in results:
        mos_details = results['voice_mos_details']
        print(f"\nVOICE MOS DETAILS:")
        for key, value in mos_details.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


def main():
    """Main function to run server-based model testing."""
    parser = argparse.ArgumentParser(
        description="Test model scoring using a server API"
    )

    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8010",
        help="URL of the model server API endpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scoring_results.json",
        help="Output file to save results as JSON (default: scoring_results.json)"
    )

    # Add bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse with bittensor config
    config = bt.config(parser)

    print(f"\nStarting server-based model scoring...")
    print(f"API URL: {config.api_url}")

    start_time = time.time()

    try:
        # Run the evaluation
        results = test_scoring_with_server(config.api_url)

        # Print detailed results
        print_detailed_results(results)

        # Save results to JSON
        with open(config.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {config.output}")

        print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()