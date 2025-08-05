#!/usr/bin/env python3
"""
Test complete validator workflow with Hugging Face miner model
Testing with siddhantoon/miner_v2t from Hugging Face
"""

import sys
import time
import os
sys.path.insert(0, '.')

from neurons.scoring_manager import ScoringManager, ScoreModelInputs, get_scoring_config
import bittensor as bt

def test_hf_miner_model():
    """Test the complete validator workflow with HF miner model"""
    
    # Configure logging
    bt.logging.set_trace(True)
    bt.logging.set_debug(True)
    
    print("🚀 Testing Complete Validator with Hugging Face Miner Model")
    print("=" * 80)
    print(f"🤖 Model: siddhantoon/miner_v2t")
    print(f"🎯 Competition: voicebench (with LLM judge)")
    print("=" * 80)
    
    # Get configuration with proper model parameters
    import argparse
    import bittensor as bt
    
    # Create parser to mimic command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtensor.network", default="finney")
    parser.add_argument("--netuid", type=int, default=21)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--wandb.off", action="store_true", default=True)
    parser.add_argument("--hf_repo_id", type=str, default="siddhantoon/miner_v2t")
    parser.add_argument("--competition_id", type=str, default="voicebench")
    parser.add_argument("--hotkey", type=str, default="5FTXYCvdLXS4tD6i87PD2dVFXSUJw4BdGomoecSfATr8hNCL")
    parser.add_argument("--block", type=int, default=6155120)
    
    # Parse config
    config = bt.config(parser)
    config.offline = True
    config.wandb.off = True
    
    # Initialize scoring manager
    print("📋 Initializing scoring manager...")
    scoring_manager = ScoringManager(config)
    
    # Use config values for the test
    print(f"🔄 Testing Model: {config.hf_repo_id}")
    print(f"📝 Competition: {config.competition_id}")
    print(f"🔑 Hotkey: {config.hotkey}")
    print(f"📦 Block: {config.block}")
    
    # Create scoring inputs from config
    scoring_inputs = ScoreModelInputs(
        hf_repo_id=config.hf_repo_id,
        competition_id=config.competition_id,
        hotkey=config.hotkey,
        block=config.block
    )
    
    print(f"\n📥 Starting complete validator evaluation...")
    print("🔧 This will:")
    print("   1. Download the HF model")
    print("   2. Deploy it in Docker container")
    print("   3. Run VoiceBench evaluation across multiple datasets")
    print("   4. Use LLM judge for quality scoring")
    print("   5. Calculate weighted final scores")
    print("⏳ Expected duration: 20-40 minutes")
    
    start_time = time.time()
    
    try:
        # Run scoring through the complete validator pipeline
        result = scoring_manager._score_model(scoring_inputs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Complete validator evaluation completed in {duration/60:.1f} minutes")
        print("=" * 80)
        print("📊 FINAL VALIDATOR RESULTS:")
        print("=" * 80)
        
        if result:
            # Overall scores
            combined_score = result.get('combined_score', 'N/A')
            voicebench_score = result.get('voicebench_score', 'N/A')
            
            print(f"🏆 COMBINED SCORE: {combined_score}")
            print(f"🤖 VOICEBENCH LLM SCORE: {voicebench_score}")
            print()
            
            # Detailed VoiceBench results
            if 'voicebench_scores' in result:
                print(f"📈 VoiceBench Dataset Scores (LLM-based):")
                for dataset, score in result['voicebench_scores'].items():
                    if dataset == 'overall':
                        print(f"   🏆 {dataset.upper():<20}: {score:.3f}")
                    else:
                        print(f"   🤖 {dataset:<20}: {score:.3f}")
                print()
            
            # Traditional scores for comparison
            if 'traditional_scores' in result:
                print(f"📊 Traditional Scores (Success Rate-based):")
                for dataset, score in result['traditional_scores'].items():
                    if dataset == 'overall':
                        print(f"   📊 {dataset.upper():<20}: {score:.3f}")
                    else:
                        print(f"   📋 {dataset:<20}: {score:.3f}")
                print()
            
            # Evaluation details
            if 'voicebench_results' in result:
                print(f"🔍 Evaluation Details:")
                voicebench_results = result['voicebench_results']
                
                total_samples_all = 0
                total_successful_all = 0
                total_llm_evaluated = 0
                
                for dataset_key, dataset_result in voicebench_results.items():
                    if isinstance(dataset_result, dict) and 'total_samples' in dataset_result:
                        total = dataset_result.get('total_samples', 0)
                        successful = dataset_result.get('successful_responses', 0)
                        success_rate = dataset_result.get('success_rate', 0.0)
                        llm_score = dataset_result.get('llm_score', 0.0)
                        llm_valid = dataset_result.get('llm_valid_count', 0)
                        
                        total_samples_all += total
                        total_successful_all += successful
                        total_llm_evaluated += llm_valid
                        
                        print(f"   📋 {dataset_key:<20}: {successful:3d}/{total:3d} samples ({success_rate:6.1%} success) | LLM: {llm_score:.3f} ({llm_valid} evaluated)")
                
                print(f"\n📊 OVERALL STATISTICS:")
                print(f"   Total Samples Processed: {total_samples_all}")
                print(f"   Successful API Responses: {total_successful_all}")
                print(f"   LLM Evaluated Responses: {total_llm_evaluated}")
                if total_samples_all > 0:
                    overall_success_rate = total_successful_all / total_samples_all
                    print(f"   Overall API Success Rate: {overall_success_rate:.1%}")
            
            # Show any errors
            error_keys = [k for k in result.keys() if 'error' in k]
            if error_keys:
                print(f"\n⚠️  Errors/Warnings:")
                for error_key in error_keys:
                    print(f"   {error_key}: {result[error_key]}")
                    
        else:
            print("   ❌ No results returned (likely an error occurred)")
            
    except Exception as e:
        print(f"\n❌ Error during validator evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Complete validator test finished!")
    print("=" * 80)


def test_quick_hf_model():
    """Quick test with limited datasets for faster validation"""
    
    print("🧪 Quick validator test with HF model (limited datasets)")
    
    # Set environment variable to limit datasets for testing
    os.environ['VOICEBENCH_TEST_MODE'] = 'true'
    
    try:
        test_hf_miner_model()
    finally:
        # Clean up
        if 'VOICEBENCH_TEST_MODE' in os.environ:
            del os.environ['VOICEBENCH_TEST_MODE']


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test validator with HF miner model")
    parser.add_argument('--quick', action='store_true', help='Run quick test with limited datasets')
    
    args = parser.parse_args()
    
    if args.quick:
        test_quick_hf_model()
    else:
        test_hf_miner_model()