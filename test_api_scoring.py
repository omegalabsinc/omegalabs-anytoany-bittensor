#!/usr/bin/env python3
"""
Test the scoring API endpoints using Python requests
"""

import requests
import json
import time
from requests.auth import HTTPBasicAuth

def test_scoring_api():
    """Test the scoring API endpoints"""
    
    base_url = "http://localhost:8003"
    
    # Test credentials - replace with actual values
    validator_hotkey = "YOUR_VALIDATOR_HOTKEY"
    signature = "YOUR_SIGNATURE"  # This should be your actual signature
    
    # Test if API is running
    print("🔍 Testing API connection...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ API is running: {response.json()}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return
    
    # Test model scoring
    test_model = {
        "hf_repo_id": "microsoft/DialoGPT-medium",
        "competition_id": "v2v", 
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "block": 1000000
    }
    
    print(f"\n🚀 Starting scoring for: {test_model['hf_repo_id']}")
    
    # Start scoring
    try:
        response = requests.post(
            f"{base_url}/api/start_model_scoring",
            auth=HTTPBasicAuth(validator_hotkey, signature),
            json=test_model,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Scoring started: {result}")
            
            if result.get("success"):
                # Poll for results
                print("⏳ Waiting for scoring to complete...")
                
                while True:
                    time.sleep(10)
                    
                    status_response = requests.get(
                        f"{base_url}/api/check_scoring_status",
                        auth=HTTPBasicAuth(validator_hotkey, signature),
                        timeout=30
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"📊 Status: {status_data}")
                        
                        if status_data and status_data.get("status") != "scoring":
                            print("🏁 Scoring completed!")
                            break
                    else:
                        print(f"❌ Status check failed: {status_response.status_code}")
                        break
            else:
                print(f"❌ Failed to start scoring: {result}")
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error during API test: {e}")

if __name__ == "__main__":
    print("🧪 VoiceBench Scoring API Test")
    print("=" * 50)
    test_scoring_api()