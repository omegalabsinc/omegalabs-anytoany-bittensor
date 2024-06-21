# pip install streamlit datasets huggingface-hub ulid-py bittensor

import asyncio
import os
from datetime import datetime as dt
import typing
import json

from traceback import print_exception

import streamlit as st
import huggingface_hub
from datasets import load_dataset
import ulid

from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from neurons.model_scoring import pull_latest_omega_dataset, get_model_score

import bittensor as bt

NETWORK = None
NETUID = 21
VPERMIT_TAO_LIMIT = 1024
JSON_FILE = "leaderboard.json"

subtensor = bt.subtensor(network=NETWORK)
metagraph: bt.metagraph = subtensor.metagraph(NETUID)

model_tracker = ModelTracker()

# Setup a ModelMetadataStore
metadata_store = ChainModelMetadataStore(
    subtensor, NETUID, None
)

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    return True

def get_uids(
    exclude: typing.List[int] = None
) -> typing.List[int]:
    """Returns k available random uids from the metagraph.
    Args:
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (List[int]): Available uids.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(
            metagraph, uid, VPERMIT_TAO_LIMIT
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    
    return avail_uids

def get_uid_rank(metagraph, uids: typing.List[int] = None, uid_to_rank: int = None) -> int:
    incentives = [(uid, metagraph.I[uid].item()) for uid in uids]
    sorted_incentives = sorted(incentives, key=lambda x: x[1], reverse=True)
    
    rank = 1
    last_incentive = None
    uid_to_rank_map = {}
    
    for i, (uid, inc) in enumerate(sorted_incentives):
        if last_incentive is None or inc != last_incentive:
            rank = i + 1
        uid_to_rank_map[uid] = rank
        last_incentive = inc
    
    if uid_to_rank is not None:
        return uid_to_rank_map.get(uid_to_rank, None)
    
    return None

async def pull_and_cache_miner_info():
    uids = get_uids()
    
    model_info = []
    for uid_i in uids:
        hotkey = metagraph.hotkeys[uid_i]
        model_metadata = await metadata_store.retrieve_model_metadata(hotkey)
        #bt.logging.info(f"Model metadata for {uid_i} is {model_metadata}")
        if model_metadata is None:
            continue

        model_rank = get_uid_rank(metagraph, uids, uid_i)
        model_info.append({
            "uid": uid_i,
            "name": f"Miner UID {uid_i} (Rank: {model_rank})",
            "model_path": f"{model_metadata.id.namespace}/{model_metadata.id.name}",
            "incentive": metagraph.I[uid_i].item(),
            "rank": model_rank
        })

    with open(JSON_FILE, "w") as f:
        json.dump(model_info, f)
    
    return True

async def resync_model_info():
    while True:
        try:
            await pull_and_cache_miner_info()
        except Exception as err:
            print("Error during model info sync", str(err))
            print_exception(type(err), err, err.__traceback__)
        await asyncio.sleep(1800) # 30 minutes

async def main():
    # run initial pull and cache/creation of JSON
    #await pull_and_cache_miner_info()

    # Load models from JSON file
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
            models = {model['name']: model for model in data}
    else:
        models = {}

    st.set_page_config(layout="wide")  # Set the layout to wide
    st.title("OMEGA Any2Any Leaderboard")

    # Create a two-column layout
    col1, col2 = st.columns([0.65, 0.35])

    # Main column for model demo
    with col1:
        st.header("Model Demo")
        model_names = list(models.keys())
        selected_model = st.selectbox("Select a model", model_names)

        if selected_model:
            model_info = models[selected_model]
            st.write(f"**Model Path:** {model_info['model_path']}")
            st.write(f"**Incentive:** {model_info['incentive']}")
            st.write(f"**Rank:** {model_info['rank']}")

            mini_batch = pull_latest_omega_dataset()
            print(get_model_score(model_info['model_path'], mini_batch))

            input_text = st.text_area("Input Text")
            if st.button("Generate"):
                # Dummy output for the sake of example
                output_text = f"Generated text for model {selected_model}"
                st.text_area("Output Text", value=output_text, height=200)

    # Sidebar for leaderboard
    with col2:
        st.header("Leaderboard")
        for model_name, model_info in models.items():
            st.write(f"**{model_name}**")
            st.write(f"Model Path: [{model_info['model_path']}](https://huggingface.co/{model_info['model_path']})")
            st.write(f"Incentive: {model_info['incentive']}")
            st.write(f"Rank: {model_info['rank']}")
            st.write("---")
    asyncio.create_task(resync_model_info())

if __name__ == "__main__":
    asyncio.run(main())