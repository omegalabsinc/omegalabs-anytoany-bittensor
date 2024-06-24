# pip install streamlit datasets huggingface-hub ulid-py bittensor apscheduler

import asyncio
import os
import traceback
from datetime import datetime as dt
import time
import typing
import json
import random
import html
from tempfile import TemporaryDirectory

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
from neurons.model_scoring import get_caption_from_model

import bittensor as bt
from apscheduler.schedulers.background import BackgroundScheduler

NETWORK = None
NETUID = 21
VPERMIT_TAO_LIMIT = 1024
JSON_FILE = "leaderboard.json"
CACHE_FILE = "omega_dataset_examples.json"
HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 48 * 60 * 60  #48 hours
MAX_FILES = 1
MAX_METADATA = 25

subtensor = bt.subtensor(network=NETWORK)
metagraph: bt.metagraph = subtensor.metagraph(NETUID)

model_tracker = ModelTracker()

# Setup a ModelMetadataStore
metadata_store = ChainModelMetadataStore(
    subtensor, NETUID, None
)

def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

async def pull_and_cache_recent_descriptions() -> typing.List[str]:
    # Get the list of files in the dataset repository
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    
    # Filter files that match the DATA_FILES_PREFIX
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]
    
    # Randomly sample up to MAX_FILES from the matching files
    sampled_files = random.sample(recent_files, min(MAX_FILES, len(recent_files)))
    
    # Load the dataset using the sampled files
    video_metadata = []
    with TemporaryDirectory() as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=sampled_files, cache_dir=temp_dir)["train"]
        for entry in omega_dataset:
            if "description" in entry and "description_embed" in entry:
                video_metadata.append(entry)
            
            if len(video_metadata) >= MAX_METADATA:
                break
    
    # Cache the descriptions to a local file
    with open(CACHE_FILE, "w") as f:
        json.dump(video_metadata, f)
    
    return video_metadata

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

@st.cache_data(ttl=1800) # Cache for 30 minutes
def load_models():
    # Load models from JSON file
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
            models = {model['name']: model for model in data}
    else:
        models = {}

    return models

@st.cache_data(ttl=1800) # Cache for 30 minutes
def load_video_metadata():
    # Load video metadata from JSON file
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            video_metadata = json.load(f)
    else:
        video_metadata = []

    return video_metadata

# Background task scheduler
scheduler = BackgroundScheduler()

def scheduled_tasks():
    pull_and_cache_miner_info()
    pull_and_cache_recent_descriptions()

# Check if the job already exists before adding it
if not scheduler.get_job('scheduled_tasks'):
    scheduler.add_job(scheduled_tasks, 'interval', minutes=30, id='scheduled_tasks')
scheduler.start()

async def main():
    # run initial pull and cache/creation of JSON
    #await pull_and_cache_miner_info()
    #await pull_and_cache_recent_descriptions()

    st.set_page_config(
        layout="wide",
        page_title='OMEGA Any2Any Leaderboard',
        page_icon="\u03a9"
    )

    models = load_models()
    video_metadata = load_video_metadata()
    

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
        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Roboto', sans-serif;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        button {
            background-color: #068AC7;
            color: white;
            border: none;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        button:hover {
            background-color: #005f8a;
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

    # Create a three-column layout
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
            
            st.markdown('<h2 class="centered-title">Recent Video Metadata</h2>', unsafe_allow_html=True)
            st.write("-----------------------------------------------------")    
            
            # Iterate over the DataFrame rows and create a button for each row
            for index, row in enumerate(video_metadata):
                # Create a three-column layout
                ccol1, ccol2, ccol3 = st.columns([0.45, 0.05, 0.45])
                with ccol1:
                    st.write(f"**YouTube ID:** {row['youtube_id']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Start Time:** {row['start_time']}")
                    st.write(f"**End Time:** {row['end_time']}")
                    st.write(f"**Relevance Score:** {row['description_relevance_score']}")
                    if st.button(f"Generate Caption for Video {row['youtube_id']}", key=f"button_{index}"):
                        st.write(f"Processing video ID: {row['youtube_id']} with the LLM...")
                        try:
                            print("Trying to generate caption from model...")
                            generated_caption = get_caption_from_model(model_info['model_path'], row['video_embed'])
                            st.text_area(f"Generated Caption", value=generated_caption, height=200, disabled=True)

                        except Exception as e:
                            print(e)
                            traceback.print_exc()
                with ccol3:
                    youtube_url = f"https://www.youtube.com/embed/{row['youtube_id']}"
                    st.video(youtube_url, start_time=row['start_time'])
        
                st.write("-----------------------------------------------------")    
            

    # Sidebar for leaderboard
    with col3:
        st.header("Leaderboard")

        # Prepare data for the table
        table_data = []
        for model_name, model_info in models.items():
            table_data.append({
                "rank": model_info['rank'],
                "UID": model_info['uid'],
                "model path": f'https://huggingface.co/{model_info["model_path"]}',
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

if __name__ == "__main__":
    asyncio.run(main())