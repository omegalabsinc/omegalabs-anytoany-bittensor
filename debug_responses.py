#!/usr/bin/env python3
"""
Debug script to see actual responses from the miner model
"""

import sys
import time
sys.path.insert(0, '.')

from neurons.voicebench_adapter import run_voicebench_evaluation_miner
import bittensor as bt

def debug_responses():
    """Debug what responses the model is actually giving"""
    
    bt.logging.set_trace(True)
    bt.logging.set_debug(True)
    
    print("🔍 Debugging miner model responses")
    
    api_url = "http://localhost:8010/api/v1/v2t"
    
    try:
        # Test multiple datasets to see varied performance
        test_datasets = ['alpacaeval', 'commoneval', 'wildvoice', 'openbookqa', 'bbh']
        
        results = run_voicebench_evaluation_miner(
            api_url=api_url,
            datasets=test_datasets,
            splits=['test'],
            timeout=1000
        )
        
        # Print overall scores first
        voicebench_scores = results.get('voicebench_scores', {})  # Now LLM scores (primary)
        traditional_scores = results.get('traditional_scores', {})
        
        print(f"\n🤖 PRIMARY LLM-BASED SCORES (Quality-Based):")
        for dataset, score in voicebench_scores.items():
            print(f"   {dataset:<20}: {score:.3f}")
            
        print(f"\n📊 Traditional Scores (Success Rate-Based):")
        if traditional_scores:
            for dataset, score in traditional_scores.items():
                print(f"   {dataset:<20}: {score:.3f}")
        else:
            print("   (Using LLM scores as primary)")
        
        # Print detailed responses
        voicebench_results = results.get('voicebench_results', {})
        
        for dataset_key, dataset_result in voicebench_results.items():
            print(f"\n" + "="*100)
            print(f"📊 Dataset: {dataset_key}")
            print(f"="*100)
            
            if 'error' in dataset_result:
                print(f"❌ Error: {dataset_result['error']}")
                continue
                
            responses = dataset_result.get('responses', [])
            success_rate = dataset_result.get('success_rate', 0.0)
            print(f"Total responses: {len(responses)}")
            print(f"Success rate: {success_rate:.1%}")
            
            # Show first 2 responses for each dataset to keep output manageable
            for i, response in enumerate(responses[:2]):
                print(f"\n  Sample {i+1}:")
                print(f"    Prompt: {response.get('prompt', '')}")
                print(f"    Response: {response.get('response', '')}")
                print(f"    Response Length: {len(response.get('response', ''))}")
                print(f"    Reference: {response.get('reference', '')}")
                if 'llm_score' in response:
                    print(f"    LLM Score: {response['llm_score']:.3f}")
                if 'llm_raw_response' in response:
                    print(f"    LLM Raw: {response['llm_raw_response']}")
                if 'error' in response:
                    print(f"    Error: {response['error']}")
                if 'llm_error' in response:
                    print(f"    LLM Error: {response['llm_error']}")
                print("    " + "-"*80)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_responses()