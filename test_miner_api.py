#!/usr/bin/env python3
"""
Test VoiceBench scoring with miner model API (LandCruiser approach)
"""

import sys
import time
sys.path.insert(0, '.')

from neurons.voicebench_adapter import run_voicebench_evaluation_miner
from neurons.miner_model_assistant import MinerModelAssistant
import bittensor as bt

def test_miner_api():
    """Test VoiceBench evaluation with miner model API"""
    
    # Configure logging
    bt.logging.set_trace(True)
    bt.logging.set_debug(True)
    
    print("🚀 Testing VoiceBench with Miner Model API")
    print("=" * 60)
    
    # Configuration for your miner model API
    api_url = "http://localhost:8010/api/v1/v2t"  # Adjust as needed
    timeout = 600  # 10 minutes
    
    print(f"🔗 API URL: {api_url}")
    print(f"⏱️  Timeout: {timeout} seconds")
    
    # Test basic API connectivity first
    print("\n🔍 Testing API connectivity...")
    try:
        assistant = MinerModelAssistant(api_url=api_url, timeout=30)
        
        # Create a simple test audio (1 second of silence)
        import numpy as np
        test_audio = {
            'array': np.zeros(16000, dtype=np.float32),  # 1 second at 16kHz
            'sampling_rate': 16000
        }
        
        response = assistant.generate_audio(test_audio)
        if response:
            print(f"✅ API is responsive. Test response: {response[:50]}...")
        else:
            print("⚠️  API responded but returned empty response")
            
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")
        print("Make sure your miner model is running on the specified URL")
        return
    
    # Run full VoiceBench evaluation
    print(f"\n🎯 Starting comprehensive VoiceBench evaluation...")
    print("📊 This will evaluate across all 11 VoiceBench datasets")
    print("⏳ Expected duration: 15-30 minutes")
    
    start_time = time.time()
    
    try:
        # Run VoiceBench evaluation
        results = run_voicebench_evaluation_miner(
            api_url=api_url,
            datasets=None,  # All datasets
            splits=None,    # Dataset-specific splits
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ VoiceBench evaluation completed in {duration/60:.1f} minutes")
        print("=" * 60)
        print("📊 FINAL RESULTS:")
        print("=" * 60)
        
        # Display overall scores
        voicebench_scores = results.get('voicebench_scores', {})
        overall_score = voicebench_scores.get('overall', 0.0)
        
        print(f"🤖 PRIMARY LLM-BASED SCORE: {overall_score:.3f}")
        
        # Show traditional score if available
        traditional_scores = results.get('traditional_scores', {})
        if traditional_scores:
            print(f"📊 Traditional Score: {traditional_scores.get('overall', 0.0):.3f}")
        print()
        
        # Display dataset-specific scores
        print("📈 Dataset Scores:")
        for dataset, score in voicebench_scores.items():
            if dataset != 'overall':
                print(f"   📊 {dataset:<20}: {score:.3f}")
        
        print()
        
        # Display evaluation details
        voicebench_results = results.get('voicebench_results', {})
        print("🔍 Evaluation Details:")
        
        total_samples_all = 0
        total_successful_all = 0
        
        for dataset_key, dataset_result in voicebench_results.items():
            if isinstance(dataset_result, dict) and 'total_samples' in dataset_result:
                total = dataset_result.get('total_samples', 0)
                successful = dataset_result.get('successful_responses', 0)
                success_rate = dataset_result.get('success_rate', 0.0)
                
                total_samples_all += total
                total_successful_all += successful
                
                print(f"   📋 {dataset_key:<20}: {successful:3d}/{total:3d} ({success_rate:6.1%})")
                
                # Show errors if any
                if 'error' in dataset_result:
                    print(f"      ❌ Error: {dataset_result['error']}")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Samples Processed: {total_samples_all}")
        print(f"   Successful Responses: {total_successful_all}")
        if total_samples_all > 0:
            overall_success_rate = total_successful_all / total_samples_all
            print(f"   Overall Success Rate: {overall_success_rate:.1%}")
        
        print("\n" + "=" * 60)
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during VoiceBench evaluation: {e}")
        import traceback
        traceback.print_exc()


def test_single_dataset():
    """Test with just one dataset for quick validation"""
    
    print("🧪 Quick test with single dataset (alpacaeval)")
    
    api_url = "http://localhost:8010/api/v1/v2t"
    
    try:
        results = run_voicebench_evaluation_miner(
            api_url=api_url,
            datasets=['alpacaeval'],  # Just one dataset
            splits=['test'],
            timeout=1000
        )
        
        print("✅ Single dataset test completed")
        print(f"🤖 LLM Score (Primary): {results['voicebench_scores'].get('overall', 0.0):.3f}")
        if 'traditional_scores' in results:
            print(f"📊 Traditional Score: {results['traditional_scores'].get('overall', 0.0):.3f}")
        
    except Exception as e:
        print(f"❌ Single dataset test failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VoiceBench with miner API")
    parser.add_argument('--quick', action='store_true', help='Run quick test with single dataset')
    parser.add_argument('--api-url', default='http://localhost:8010/api/v1/v2t', help='Miner API URL')
    
    args = parser.parse_args()
    
    if args.quick:
        test_single_dataset()
    else:
        test_miner_api()