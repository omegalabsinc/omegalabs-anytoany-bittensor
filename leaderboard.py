# pip install streamlit datasets huggingface-hub ulid-py bittensor

import asyncio
import os
import traceback
from datetime import datetime as dt
import time
import typing
import json

from traceback import print_exception

import streamlit as st
import huggingface_hub
from datasets import load_dataset
import pandas as pd
import ulid
import torch.nn as nn

from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from neurons.model_scoring import pull_latest_omega_dataset, get_model_score, get_model_score_cached

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
    await pull_and_cache_miner_info()

    # Load models from JSON file
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
            models = {model['name']: model for model in data}
    else:
        models = {}

    st.set_page_config(layout="wide")  # Set the layout to wide

    # Custom CSS for centering the title and table data
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
        }
        .centered-data td {
            text-align: center;
        }
        .logo {
            display: block; /* Use block to apply margin auto for centering */
            width: 75px; /* Set the width of the logo container */
            height: 75px; /* Set the height of the logo container */
            margin: 0 auto; /* Center the logo horizontally */
            margin-top: 0rem; /* Add space above the logo */
        }

        .logo svg {
            width: 100%; /* Make the SVG fill the container */
            height: 100%; /* Make the SVG fill the container */
        }
        </style>
    """, unsafe_allow_html=True)

    # Add the SVG logo above the title
    st.markdown("""
        <div class="logo">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 75 75">
                <!-- Define the drop shadow filter -->
                <defs>
                    <filter id="text-shadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
                        <feOffset in="blur" dx="2" dy="2" result="offsetBlur"/>
                        <feMerge>
                            <feMergeNode in="offsetBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                <text x="50%" y="70%" dominant-baseline="middle" text-anchor="middle" font-family="Roboto" font-size="100" fill="#068AC7" filter="url(#text-shadow)">Ω</text>
            </svg>
        </div>
    """, unsafe_allow_html=True)

    # Center the title
    st.markdown('<h1 class="centered-title">OMEGA Any2Any Leaderboard</h1>', unsafe_allow_html=True)

    # Create a two-column layout
    col1, col2, col3 = st.columns([0.6, 0.05, 0.3])

    # Main column for model demo
    with col1:
        st.header("Model Demo")
        model_names = ["- Select a model -"] + list(models.keys())
        selected_model = st.selectbox("Select a model", model_names)

        if selected_model and selected_model != "- Select a model -":
            model_info = models[selected_model]
            st.write(f"**Model Path:** {model_info['model_path']}")
            st.write(f"**Incentive:** {model_info['incentive']}")
            st.write(f"**Rank:** {model_info['rank']}")

            # Show a spinner and progress bar while loading the model
            with st.spinner('Loading model...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate loading time
                    progress_bar.progress(i + 1)

            try:
                print("Trying to get model score")
                mini_batch = pull_latest_omega_dataset()
                print(get_model_score_cached(model_info['model_path'], mini_batch))
            except Exception as e:
                print(e)
                traceback.print_exc()

            input_text = st.text_area("Input Text")
            if st.button("Generate"):
                # Dummy output for the sake of example
                output_text = f"Generated text for model {selected_model}"
                st.text_area("Output Text", value=output_text, height=200)

    # Sidebar for leaderboard
    with col3:
        st.header("Leaderboard")

        # Prepare data for the table
        table_data = []
        for model_name, model_info in models.items():
            table_data.append({
                "rank": model_info['rank'],
                "UID": model_info['uid'],
                "model_path": f'https://huggingface.co/{model_info["model_path"]}',
                "incentive": str(round(float(model_info['incentive']), 3))
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "model_path": st.column_config.LinkColumn()
            }
        )
    

    #asyncio.create_task(resync_model_info())

if __name__ == "__main__":
    asyncio.run(main())