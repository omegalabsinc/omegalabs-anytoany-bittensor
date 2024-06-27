import asyncio
import os
import time
import typing
import json
import random
from tempfile import TemporaryDirectory

import huggingface_hub
from datasets import load_dataset
import ulid

from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore

import bittensor as bt

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
    
    await asyncio.sleep(1)
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
    
    await asyncio.sleep(1)
    return True

async def periodic_task(interval, *tasks):
    while True:
        try:
            await asyncio.gather(*[task() for task in tasks])
            print("Data synced successfully")
        except Exception as err:
            print("Error during syncing data", str(err))
        await asyncio.sleep(interval)

async def main():
    await periodic_task(1800, pull_and_cache_miner_info, pull_and_cache_recent_descriptions)

if __name__ == "__main__":
    asyncio.run(main())