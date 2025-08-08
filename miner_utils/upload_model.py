"""A script that pushes a model from disk to the subnet for evaluation.

Usage:
    python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. load_model_dir points to a directory containing a previously trained model, with relevant ckpt file named "checkpoint.pth".
   3. Your miner is registered
"""
import traceback
import asyncio
import os
import argparse
import constants
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore, get_required_files, check_config
from model.model_updater import ModelUpdater
import bittensor as bt
from utilities import utils
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from huggingface_hub import update_repo_settings
import time
import hashlib

from dotenv import load_dotenv

load_dotenv()

# enable bt.logging
import logging
logging.basicConfig(level=logging.INFO)

def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="The director of the model to load.",
    )

    parser.add_argument(
        "--epoch",
        type=str,
        default=0,
        help="The epoch number to load e.g. if you want to upload meta_model_0.pt, epoch should be 0",
    )

    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )

    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )

    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )
    parser.add_argument(
        "--subtensor.network",
        type=str,
        default="finney",
        help="The subtensor network flag. The likely choices are: finney (main network) test (testnet)",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)
    return config


def regenerate_hash(namespace, name, epoch, competition_id):
    s = " ".join([namespace, name, epoch, competition_id])
    hash_output = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return int(hash_output[:16], 16)  # Returns a 64-bit integer from the first 16 hexadecimal characters


def validate_repo(ckpt_dir, epoch, model_type):
    for filename in get_required_files(epoch, model_type):
        filepath = os.path.join(ckpt_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file {filepath} not found in {ckpt_dir}")
    if model_type == "o1":
        check_config(ckpt_dir)


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging.set_config(config=config.logging)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    bt.logging.debug("Subtensor network: ", subtensor.network)
    metagraph: bt.metagraph = subtensor.metagraph(config.netuid)

    # Make sure we're registered and have a HuggingFace token.
    utils.assert_registered(wallet, metagraph)

    # Get current model parameters
    parameters = ModelUpdater.get_competition_parameters(config.competition_id)
    if parameters is None:
        raise RuntimeError(
            f"Could not get competition parameters for block {config.competition_id}"
        )

    repo_namespace, repo_name = utils.validate_hf_repo_id(config.hf_repo_id)
    bt.logging.info(f"Repo namespace: {repo_namespace}, repo name: {repo_name}, competition id: {config.competition_id}")
    model_id = ModelId(
        namespace=repo_namespace,
        name=repo_name,
        epoch=config.epoch,
        competition_id=config.competition_id,
        commit="",
        hash="",
    )

    model = Model(id=model_id, local_repo_dir=config.model_dir)
   

    bt.logging.info(f"Validated repo for {config.model_dir}")

    remote_model_store = HuggingFaceModelStore()

    bt.logging.info(f"Uploading model to Hugging Face with id {model_id}")

    model_id_with_commit = await remote_model_store.upload_model(
        model=model,
        competition_parameters=parameters,
        hotkey=wallet.hotkey.ss58_address
    )

    model_hash = regenerate_hash(repo_namespace, repo_name, config.epoch, config.competition_id)
    model_id_with_hash = ModelId(
        namespace=repo_namespace,
        name=repo_name,
        epoch=config.epoch,
        hash=str(model_hash),
        commit=model_id_with_commit.commit,
        competition_id=config.competition_id,
    )
    
    bt.logging.info(
        f"Model uploaded to Hugging Face with commit {model_id_with_hash.commit} and hash {model_id_with_hash.hash}"
    )

    model_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=config.netuid
    )

    # We can only commit to the chain every n minutes, so run this in a loop, until successful.
    while True:
        try:
            await model_metadata_store.store_model_metadata(
                wallet.hotkey.ss58_address, model_id_with_hash
            )
            update_repo_settings(
                model_id.namespace + "/" + model_id.name,
                private=False,
                token=os.getenv("HF_ACCESS_TOKEN"),
            )
            bt.logging.success("Committed model to the chain.")
            break
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error("Retrying in 120 seconds...")
            traceback.print_exc()
            time.sleep(120)


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        bt.logging.info(constants.COMPETITION_SCHEDULE)
    else:
        bt.logging.info(config)
        asyncio.run(main(config))
