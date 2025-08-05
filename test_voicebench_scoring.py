#!/usr/bin/env python3
"""
Test script for VoiceBench scoring integration
"""

import sys
import os
import asyncio
import time
sys.path.insert(0, '.')

from neurons.scoring_manager import ScoringManager, ScoreModelInputs, get_scoring_config
import bittensor as bt

def test_scoring():
    """Test the VoiceBench scoring with a sample model"""
    
    # Configure logging
    bt.logging.set_trace(True)
    bt.logging.set_debug(True)
    
    print("🚀 Starting VoiceBench Scoring Test")
    
    # Get configuration
    config = get_scoring_config()
    config.offline = True
    config.wandb.off = True
    
    # Initialize scoring manager
    print("📋 Initializing scoring manager...")
    scoring_manager = ScoringManager(config)
    
    # Test models - replace with actual HF model repos
    test_models = [
        {
            "hf_repo_id": "microsoft/DialoGPT-medium",
            "competition_id": "v2v",
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "block": 1000000
        },
        # Add more test models if needed
    ]
    
    for i, model_data in enumerate(test_models, 1):
        print(f"\n🔄 Testing Model {i}/{len(test_models)}: {model_data['hf_repo_id']}")
        
        # Create scoring inputs
        scoring_inputs = ScoreModelInputs(**model_data)
        
        print(f"📥 Scoring inputs: {scoring_inputs}")
        
        # Start scoring
        print("⏳ Starting VoiceBench evaluation...")
        start_time = time.time()
        
        try:
            # Run scoring directly (synchronous for testing)
            result = scoring_manager._score_model(scoring_inputs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Scoring completed in {duration:.2f} seconds")
            print(f"📊 Results:")
            
            if result:
                print(f"   - Combined Score: {result.get('combined_score', 'N/A')}")
                print(f"   - VoiceBench Score: {result.get('voicebench_score', 'N/A')}")
                
                if 'voicebench_scores' in result:
                    print(f"   - Dataset Scores:")
                    for dataset, score in result['voicebench_scores'].items():
                        if dataset != 'overall':
                            print(f"     * {dataset}: {score:.3f}")
                
                if 'voicebench_results' in result:
                    print(f"   - Evaluation Details:")
                    for dataset_key, dataset_result in result['voicebench_results'].items():
                        if 'total_samples' in dataset_result:
                            success_rate = dataset_result.get('success_rate', 0.0)
                            total = dataset_result.get('total_samples', 0)
                            successful = dataset_result.get('successful_responses', 0)
                            print(f"     * {dataset_key}: {successful}/{total} ({success_rate:.1%})")
            else:
                print("   - No results returned")
                
        except Exception as e:
            print(f"❌ Error during scoring: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_scoring()