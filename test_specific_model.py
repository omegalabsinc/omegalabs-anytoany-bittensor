#!/usr/bin/env python3
"""
Test VoiceBench scoring with the specific model LandCruiser/sn21_omg_1806_22
"""

import sys
import time
sys.path.insert(0, '.')

from neurons.scoring_manager import ScoringManager, ScoreModelInputs, get_scoring_config
import bittensor as bt

def test_landcruiser_model():
    """Test the VoiceBench scoring with LandCruiser model"""
    
    # Configure logging
    bt.logging.set_trace(True)
    bt.logging.set_debug(True)
    
    print("🚀 Testing VoiceBench Scoring with LandCruiser Model")
    print("=" * 60)
    
    # Get configuration
    config = get_scoring_config()
    config.offline = True
    config.wandb.off = True
    
    # Initialize scoring manager
    print("📋 Initializing scoring manager...")
    scoring_manager = ScoringManager(config)
    
    # Test with the specific LandCruiser model
    test_model = {
        "hf_repo_id": "LandCruiser/sn21_omg_1806_22",
        "competition_id": "v2v",  # Voice-to-voice competition
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "block": 1000000
    }
    
    print(f"🔄 Testing Model: {test_model['hf_repo_id']}")
    print(f"📝 Competition: {test_model['competition_id']}")
    print(f"🔑 Hotkey: {test_model['hotkey']}")
    print(f"📦 Block: {test_model['block']}")
    
    # Create scoring inputs
    scoring_inputs = ScoreModelInputs(**test_model)
    
    print(f"\n📥 Starting VoiceBench evaluation...")
    print("⚠️  Note: This model may only support voice-to-voice (no text output)")
    
    start_time = time.time()
    
    try:
        # Run scoring directly
        result = scoring_manager._score_model(scoring_inputs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Scoring completed in {duration:.2f} seconds")
        print(f"📊 Results:")
        
        if result:
            print(f"   🎯 Combined Score: {result.get('combined_score', 'N/A')}")
            print(f"   🎤 VoiceBench Score: {result.get('voicebench_score', 'N/A')}")
            
            # Display detailed VoiceBench results
            if 'voicebench_scores' in result:
                print(f"\n   📈 Dataset Scores:")
                for dataset, score in result['voicebench_scores'].items():
                    if dataset == 'overall':
                        print(f"     🏆 {dataset.upper()}: {score:.3f}")
                    else:
                        print(f"     📊 {dataset}: {score:.3f}")
            
            # Display evaluation details
            if 'voicebench_results' in result:
                print(f"\n   🔍 Evaluation Details:")
                for dataset_key, dataset_result in result['voicebench_results'].items():
                    if 'total_samples' in dataset_result:
                        success_rate = dataset_result.get('success_rate', 0.0)
                        total = dataset_result.get('total_samples', 0)
                        successful = dataset_result.get('successful_responses', 0)
                        print(f"     📋 {dataset_key}: {successful}/{total} samples ({success_rate:.1%} success)")
                        
                        # Show any errors
                        if 'error' in dataset_result:
                            print(f"     ❌ Error: {dataset_result['error']}")
            
            # Display any errors or warnings
            if 'voicebench_error' in result:
                print(f"\n   ⚠️  VoiceBench Error: {result['voicebench_error']}")
            
            if 's2s_error' in result:
                print(f"   ⚠️  S2S Error: {result['s2s_error']}")
                
        else:
            print("   ❌ No results returned (likely an error occurred)")
            
    except Exception as e:
        print(f"\n❌ Error during scoring: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_landcruiser_model()