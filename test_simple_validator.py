#!/usr/bin/env python3
"""
Simple test for complete validator workflow with HF model
"""

import sys
import subprocess
import time

def test_with_scoring_manager():
    """Test using scoring manager directly"""
    
    print("🚀 Testing Complete Validator with siddhantoon/miner_v2t")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "neurons.scoring_manager",
        "--hf_repo_id", "siddhantoon/miner_v2t",
        "--competition_id", "voicebench",
        "--hotkey", "5FTXYCvdLXS4tD6i87PD2dVFXSUJw4BdGomoecSfATr8hNCL", 
        "--block", "6155120",
        "--offline",
        "--wandb.off"
    ]
    
    print("🔧 Running command:")
    print(" ".join(cmd))
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Evaluation completed in {duration/60:.1f} minutes")
        print("=" * 80)
        print("📊 STDOUT:")
        print("=" * 80)
        print(result.stdout)
        
        if result.stderr:
            print("=" * 80)
            print("⚠️ STDERR:")
            print("=" * 80)
            print(result.stderr)
        
        print("=" * 80)
        print(f"🏁 Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
        else:
            print("❌ Test failed with errors")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 1 hour")
    except Exception as e:
        print(f"❌ Error running test: {e}")


def test_with_legacy_v2v():
    """Test with legacy v2v competition for comparison"""
    
    print("🧪 Testing with Legacy V2V Competition")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "neurons.scoring_manager",
        "--hf_repo_id", "siddhantoon/miner_v2t",
        "--competition_id", "v2v",
        "--hotkey", "5FTXYCvdLXS4tD6i87PD2dVFXSUJw4BdGomoecSfATr8hNCL", 
        "--block", "6155120",
        "--offline",
        "--wandb.off"
    ]
    
    print("🔧 Running command:")
    print(" ".join(cmd))
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Legacy evaluation completed in {duration/60:.1f} minutes")
        print("=" * 80)
        print("📊 STDOUT:")
        print("=" * 80)
        print(result.stdout)
        
        if result.stderr:
            print("=" * 80)
            print("⚠️ STDERR:")
            print("=" * 80)
            print(result.stderr)
        
        print("=" * 80)
        print(f"🏁 Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("❌ Legacy test timed out after 30 minutes")
    except Exception as e:
        print(f"❌ Error running legacy test: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test validator with HF model")
    parser.add_argument('--legacy', action='store_true', help='Test with legacy v2v competition')
    parser.add_argument('--both', action='store_true', help='Test both voicebench and legacy')
    
    args = parser.parse_args()
    
    if args.both:
        print("🔄 Running both VoiceBench and Legacy tests")
        test_with_scoring_manager()
        print("\n" + "="*100 + "\n")
        test_with_legacy_v2v()
    elif args.legacy:
        test_with_legacy_v2v()
    else:
        test_with_scoring_manager()