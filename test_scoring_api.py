import sys
import json
import time
import requests
from requests.auth import HTTPBasicAuth
from neurons.validator import Validator
import bittensor as bt
from neurons.scoring_manager import ScoreModelInputs

config = Validator.config()
wallet = bt.wallet(config=config)
keypair = wallet.hotkey

def hit_api(url: str, data: dict):
    response = requests.post(url, json=data)
    return response.json()

if len(sys.argv) < 2:
    print("Please provide the API URL as a command line argument")
    sys.exit(1)

api_root = "http://localhost:8080"
post_url = f"{api_root}/api/start_model_scoring"
get_url = f"{api_root}/api/check_scoring_status"
hotkey = keypair.ss58_address
signature = f"0x{keypair.sign(hotkey).hex()}"

inputs = ScoreModelInputs(
    hf_repo_id="tezuesh/moshi_general",
    competition_id="v1",
    hotkey="5FeqmebkCWfepQPgSkrEHRwtpUmHGASF4BNERZDs9pvKFtcD",
    block=0,
)

inputs = ScoreModelInputs(
    hf_repo_id="tezuesh/IBLlama_v1",
    competition_id="o1",
    hotkey="5CVaXUhgrH3KvSpdu9Gh1k44ZcscRfAaS3yudXEZmwxsw52G",
    block=0,
)

status = "scoring"
print(requests.post(post_url, auth=HTTPBasicAuth(hotkey, signature), json=json.loads(inputs.model_dump_json())).json())
while status == "scoring":
    response = requests.get(get_url, auth=HTTPBasicAuth(hotkey, signature)).json()
    print(response)
    status = response["status"]
    time.sleep(5)

