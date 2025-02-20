# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
os.environ["USE_TORCH"] = "1"
os.environ["BT_LOGGING_INFO"] = "1"

from utilities.logging_setup import warmup_logging; warmup_logging()

from collections import defaultdict
import datetime as dt
import math
import time
import torch
import numpy as np
import shutil
import asyncio
from aiohttp import ClientSession, BasicAuth
import requests
from requests.auth import HTTPBasicAuth
import argparse
import typing
import json
import constants

import bittensor as bt
import wandb
from scipy import optimize

from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.mysql_model_queue import init_database, ModelQueueManager
from model.storage.disk.utils import get_hf_download_path
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from datasets import disable_caching
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console
from neurons.scoring_manager import ScoreModelInputs

from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor
from utilities.temp_dir_cache import TempDirCache
from utilities.git_utils import is_git_latest

def iswin(score_i, score_j, block_i, block_j, hash_i=None, hash_j=None):
    """
    Determines the winner between two models based on score, block number, and model hash.
    Implements a strong winner-takes-all dynamic by heavily penalizing later models.

    Parameters:
        score_i (float): Score of uid i
        score_j (float): Score of uid j
        block_i (int): Block of uid i
        block_j (int): Block of uid j
        hash_i (str): Model hash of uid i
        hash_j (str): Model hash of uid j
    Returns:
        bool: True if score i is better, False otherwise.
    """
    # Adjust score based on timestamp and pretrain epsilon
    score_i = (1 - constants.timestamp_epsilon) * score_i if block_i > block_j else score_i
    score_j = (1 - constants.timestamp_epsilon) * score_j if block_j > block_i else score_j
    return score_i > score_j

def compute_wins(
    uids: typing.List[int],
    scores_per_uid: typing.Dict[int, float], 
    uid_to_block: typing.Dict[int, int],
    uid_to_hash: typing.Dict[int, str] = None,
):
    """
    Computes the wins and win rate for each model based on score comparison.
    Implements winner-takes-all by giving extra weight to early high-scoring models.
    Handles duplicate models by favoring the earlier submission.

    Parameters:
        uids (list): A list of uids to compare.
        scores_per_uid (dict): A dictionary of scores for each uid.
        uid_to_block (dict): A dictionary of blocks for each uid - higher means more recent.
        uid_to_hash (dict): A dictionary mapping uids to their model hashes.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    uid_to_hash = uid_to_hash or {}  # Default to empty dict if None

    # If there is only one score, then the winning uid is the one with the score.
    if len([score for score in scores_per_uid.values() if score != 0]) == 1:
        winning_uid = [uid for uid, score in scores_per_uid.items() if score != 0][0]
        wins[winning_uid] = 1
        win_rate[winning_uid] = 1.0
        return wins, win_rate

    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        hash_i = uid_to_hash[uid_i]
        # Skip if block is None or score is 0
        if block_i is None or scores_per_uid[uid_i] == 0:
            continue
        for j, uid_j in enumerate(uids):
            if i == j or uid_to_block[uid_j] is None or scores_per_uid[uid_j] == 0:
                continue
            block_j = uid_to_block[uid_j]
            hash_j = uid_to_hash[uid_j]
            score_i = scores_per_uid[uid_i]
            score_j = scores_per_uid[uid_j]
            # Pass model hashes to iswin function
            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j, hash_i, hash_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate

def best_uid(metagraph: bt.metagraph) -> int:
    """Returns the best performing UID in the metagraph."""
    return max(range(metagraph.n), key=lambda uid: metagraph.I[uid].item())

def nearest_tempo(start_block, tempo, block):
    start_num = start_block + tempo
    intervals = (block - start_num) // tempo
    nearest_num = start_num + intervals * tempo
    if nearest_num >= block:
        nearest_num -= tempo
    return nearest_num

class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device name.",
        )
        parser.add_argument(
            "--run_api",
            action="store_true",
            help="Validator code runs api tasks. Only used when running the API server.",
        )
        parser.add_argument(
            "--sample_total_models",
            type=int,
            default=8,
            help="Number of uids to eval each step.",
        )
        parser.add_argument(
            "--sample_top_models",
            type=int,
            default=2,
            help="Number of top-scoring uid models to persist for eval each step. Should be LESS than sample_min.",
        )
        parser.add_argument(
            "--sample_updated_models",
            type=int,
            default=2,
            help="Number of updated uid models to eval each step. Should be LESS than sample_min.",
        )
        parser.add_argument(
            "--cached_models",
            type=int,
            default=10,
            help="Number of model repos to cache on disk.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--immediate",
            action="store_true",
            help="Triggers setting weights immediately. NOT RECOMMENDED FOR PRODUCTION. This is used internally by SN21 devs for faster testing.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--model_dir",
            default=os.path.join(constants.ROOT_DIR, "model-store/"),
            help="Where to store downloaded models",
        )
        parser.add_argument(
            "--netuid", type=int, default=constants.SUBNET_UID, help="The subnet UID."
        )
        parser.add_argument(
            "--genesis",
            action="store_true",
            help="Don't sync to consensus, rather start evaluation from scratch",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            help="datatype to load model in, either bfloat16 or float16",
        )
        parser.add_argument(
            "--clean_period_minutes",
            type=int,
            default=1,
            help="How often to delete unused models",
        )
        parser.add_argument(
            "--update_delay_minutes",
            type=int,
            default=5,
            help="Period between checking for new models from each UID",
        )
        parser.add_argument(
            "--do_sample",
            action="store_true",
            help="Sample a response from each model (for leaderboard)",
        )
        parser.add_argument(
            "--auto_update",
            action="store_true",
            help="Quits and restarts the validator if it is out of date.",
            default=False,
        )
        parser.add_argument(
            "--wandb.off",
            action="store_true",
            help="Runs wandb in offline mode.",
            default=False,
        )
        parser.add_argument(
            "--scoring_api_url",
            type=str,
            default="http://localhost:8080",
            help="The scoring API node url to use.",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def state_path(self) -> str:
        """
        Constructs a file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                bt.logging.config().logging.logging_dir,
                self.wallet.name,
                self.wallet.hotkey_str,
                self.config.netuid,
                "vali-state",
            )
        )
    
    def new_wandb_run(self):
        # Shoutout SN13 for the wandb snippet!
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        now = dt.datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project="omega-sn21-validator-logs",
            entity="omega-labs",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "validator",
            },
            allow_val_change=True,
            anonymous="allow",
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def __init__(self):
        self.config = Validator.config()
        bt.logging.set_config(config=self.config.logging)
        bt.logging.info(f"Starting validator with config: {self.config}")
        disable_caching()

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        api_root = (
            "https://dev-sn21-api.omegatron.ai"
            if self.config.subtensor.network == "test"
            else "https://sn21-api.omegatron.ai"
        )
        bt.logging.info(f"Using SN21 API: {api_root}")
        self.get_model_endpoint = f"{api_root}/get-model-to-score"
        self.score_model_endpoint = f"{api_root}/score-model"
        self.get_all_model_scores_endpoint = f"{api_root}/get-all-model-scores"
        self.start_model_scoring_endpoint = f"{self.config.scoring_api_url}/api/start_model_scoring"
        self.check_scoring_status_endpoint = f"{self.config.scoring_api_url}/api/check_scoring_status"

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # === W&B ===
        self.wandb_run_start = None
        if not self.config.wandb.off and not self.config.offline:
            if os.getenv("WANDB_API_KEY"):
                self.new_wandb_run()
            else:
                bt.logging.exception("WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off, but it is strongly recommended to run with W&B enabled.")
        else:
            bt.logging.warning("Running with --wandb.off. It is strongly recommended to run with W&B enabled.")

        # === Model caching ===
        self.temp_dir_cache = TempDirCache(self.config.cached_models)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # === Running args ===
        self.weights = torch.zeros_like(torch.tensor(self.metagraph.S))
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        
        self.uids_to_eval: typing.Dict[str, typing.Set] = {}

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.all_uids: typing.Dict[str, typing.Set] = {}
        self.all_uids_lock = threading.RLock()

        # Initialize all_uids with empty sets for each competition.
        with self.all_uids_lock:
            for competition_params in constants.COMPETITION_SCHEDULE:
                self.all_uids[competition_params.competition_id] = set()

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.config.netuid, self.wallet
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # Touch all models, starting a timer for them to be deleted if not used
        self.model_tracker.touch_all_miner_models()
        
        self.stop_event = threading.Event()
        
        # == Initialize the update thread ==
        self.update_thread = threading.Thread(
            target=self.update_models,
            args=(self.config.update_delay_minutes,),
            daemon=True,
        )
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        self.clean_thread = threading.Thread(
            target=self.clean_models,
            args=(self.config.clean_period_minutes,),
            daemon=True,
        )
        self.clean_thread.start()

        # == Initialize the gathering scores, win-rate competition, and weight setting thread ==
        self.weight_thread = threading.Thread(
            target=self.try_set_scores_and_weights,
            args=(300,),
            daemon=True,
        )
        self.weight_thread.start()

        self.last_update_check = dt.datetime.now()
        self.update_check_interval = 1800  # 30 minutes
        if self.config.auto_update:
            bt.logging.info("Auto update enabled.")
        else:
            bt.logging.warning("Auto update disabled.")

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            if self.update_thread is not None:
                self.update_thread.join()
            self.clean_thread.join()
            if not self.config.dont_set_weights and not self.config.offline:
                self.weight_thread.join()

    def is_model_old_enough(self, model_metadata: ModelMetadata):
        """
        Determines if a model is old enough to be evaluated i.e. it must not have seen data from the OMEGA data subnet.

        Parameters:
            model_metadata (ModelMetadata): The metadata of the model to evaluate.

        Returns:
            bool: True if the model is old enough, False otherwise.
        """
        block_uploaded_at = model_metadata.block
        current_block = self.metagraph.block.item()
        model_age = (current_block - block_uploaded_at) * constants.BLOCK_DURATION
        is_old_enough = model_age > constants.MIN_AGE
        if not is_old_enough:
            bt.logging.debug(
                f"Model {model_metadata.id} is too new to evaluate. Age: {model_age} seconds"
            )
        return is_old_enough

    def update_models(self, update_delay_minutes):
        # Track how recently we updated each uid
        uid_last_checked = dict()

        queue_manager = None
        if self.config.run_api:
            # Initialize database at application startup
            init_database()
            queue_manager = ModelQueueManager()

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # Get the next uid to check
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't checked it in the last `update_delay_minutes` minutes.
                time_diff = (
                    dt.datetime.now() - uid_last_checked[next_uid]
                    if next_uid in uid_last_checked
                    else None
                )
                
                if time_diff and time_diff < dt.timedelta(minutes=update_delay_minutes):
                    # If we have seen it within `update_delay_minutes` minutes then sleep until it has been at least `update_delay_minutes` minutes.
                    time_to_sleep = (
                        dt.timedelta(minutes=update_delay_minutes) - time_diff
                    ).total_seconds()
                    bt.logging.debug(
                        f"Update loop has already processed all UIDs in the last {update_delay_minutes} minutes. Sleeping {time_to_sleep:.0f} seconds."
                    )
                    time.sleep(time_to_sleep)
                
                bt.logging.debug(f"Updating model for UID={next_uid}")

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))

                metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                if metadata is not None and self.is_model_old_enough(metadata):
                    bt.logging.info(f"Model is old enough for UID={next_uid} with hotkey {hotkey}, metadata: {metadata} , {self.config.run_api}")
                    if self.config.run_api:
                        queue_manager.store_updated_model(next_uid, hotkey, metadata, updated)
                    else:
                        with self.all_uids_lock:
                            self.all_uids[metadata.id.competition_id].add(next_uid)
                    
                    bt.logging.debug(f"Updated model for UID={next_uid}. Was new = {updated}")
                    if updated:
                        bt.logging.debug(f"Found a new model for UID={next_uid} for competition {metadata.id.competition_id}.")
                else:
                    bt.logging.debug(f"Unable to sync model for consensus UID {next_uid} with hotkey {hotkey}")

                uid_last_checked[next_uid] = dt.datetime.now()
                # Sleep for a bit to avoid spamming the API
                time.sleep(0.5)

            except Exception as e:
                bt.logging.error(
                    f"Error in update loop \n {e} \n {traceback.format_exc()}"
                )

        bt.logging.info("Exiting update models loop.")

    def clean_models(self, clean_period_minutes: int):
        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                old_models = self.model_tracker.get_and_clear_old_models()

                if len(old_models) > 0:
                    bt.logging.debug("Starting cleanup of stale models. Removing {}...".format(len(old_models)))

                for hotkey, model_metadata in old_models:
                    local_path = self.local_store.get_path(hotkey)
                    model_dir = get_hf_download_path(local_path, model_metadata.id)
                    shutil.rmtree(model_dir, ignore_errors=True)

                if len(old_models) > 0:
                    bt.logging.debug("Starting cleanup of stale models. Removing {}... Done!".format(len(old_models)))

            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}: {traceback.format_exc()}")

            time.sleep(dt.timedelta(minutes=clean_period_minutes).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    def _try_win_rate_competition(self):
        """
        Gather the scores from our API and run the win-rate competition to determine the best models for each competition.
        """
        # Get the most recent scores from the API
        try:
            all_model_scores = self.get_all_model_scores()
        except Exception as e:
            bt.logging.error(f"Failed to get all model scores: {e}: {traceback.format_exc()}")
            return
        
        if all_model_scores is None:
            bt.logging.error("No model scores returned from API.")
            return
        
        # Execute the win-rate competition
        #load_model_perf = PerfMonitor("Eval: Load model")
        #compute_loss_perf = PerfMonitor("Eval: Compute loss")

        new_weights = torch.zeros_like(self.metagraph.S)

        for competition_parameters in constants.COMPETITION_SCHEDULE:
            curr_comp_weights = torch.zeros_like(self.metagraph.S)
            uids = self.metagraph.uids.tolist()
            # convert uids to int
            uids = [int(uid) for uid in uids]
            
            # Initialize with default values (0 for scores, None for blocks)
            scores_per_uid = {uid: 0 for uid in uids}
            uid_to_block = {uid: None for uid in uids}
            uid_to_hash = {uid: None for uid in uids}
            sample_per_uid = {muid: None for muid in uids}

            # Iterate through each UID and its associated models
            curr_model_scores = {
                uid: models_data
                for uid, models_data in all_model_scores.items()
                if models_data[0]["competition_id"] == competition_parameters.competition_id
            }
            for uid, models_data in curr_model_scores.items():
                if not models_data:  # Skip if no models for this UID
                    continue
            
                # Convert UID to int
                uid = int(uid)
                # Take the first model's data (assuming one model per UID)
                model_data = models_data[0]
                
                # Extract score and block, defaulting to None if not present
                score = model_data.get('score', 0)
                block = model_data.get('block', 0)
                model_hash = model_data.get('model_hash', None)
                if model_hash == "":
                    model_hash = None
                
                if score is not None:
                    scores_per_uid[uid] = score
                if block is not None:
                    uid_to_block[uid] = block
                if model_hash is not None:
                    uid_to_hash[uid] = model_hash

            # Compute wins and win rates per uid.
            wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)

            # Loop through all models and check for duplicate model hashes. If found, score 0 for model with the newer block.
            uids_penalized_for_hash_duplication = set()
            for uid in uids:
                if uid_to_hash[uid] is None and uid_to_block[uid] is None:
                    continue
                for uid2 in uids:
                    if uid == uid2 or uid_to_hash[uid] is None or uid_to_block[uid2] is None:
                        continue
                    if uid != uid2 and uid_to_hash[uid] == uid_to_hash[uid2]:
                        if uid_to_block[uid] < uid_to_block[uid2]:
                            scores_per_uid[uid2] = 0
                            uids_penalized_for_hash_duplication.add(uid2)
                            bt.logging.warning(f"uid {uid2} at block {uid_to_block[uid2]} has duplicate model hash of uid {uid} at block {uid_to_block[uid]}. Penalizing uid {uid2} with score of 0.")
                        else:
                            scores_per_uid[uid] = 0
                            uids_penalized_for_hash_duplication.add(uid)
                            bt.logging.warning(f"uid {uid} at block {uid_to_block[uid]} has duplicate model hash of uid {uid2} at block {uid_to_block[uid2]}. Penalizing uid {uid} with score of 0.")

            # Compute softmaxed weights based on win rate.
            model_weights = torch.tensor(
                [win_rate[uid] for uid in uids], dtype=torch.float32
            )
            step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
            for i, uid_i in enumerate(uids):
                curr_comp_weights[uid_i] = step_weights[i]
            scale = competition_parameters.reward_percentage
            curr_comp_weights *= scale / curr_comp_weights.sum()
            new_weights += curr_comp_weights

            self.log_step(
                competition_parameters.competition_id,
                uids,
                uid_to_block,
                wins,
                win_rate,
                scores_per_uid,
                sample_per_uid,
            )

        # Do EMA with existing (self.weights) and new (new_weights) weights.
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = (
            constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        )
        self.weights = self.weights.nan_to_num(0.0)

        # Log the performance of the eval loop.
        #bt.logging.debug(load_model_perf.summary_str())
        #bt.logging.debug(compute_loss_perf.summary_str())

    @staticmethod
    def adjust_for_vtrust(weights: np.ndarray, consensus: np.ndarray, vtrust_min: float = 0.5):
        """
        Interpolate between the current weight and the normalized consensus weights so that the
        vtrust does not fall below vturst_min, assuming the consensus does not change.
        """
        vtrust_loss_desired = 1 - vtrust_min

        # If the predicted vtrust is already above vtrust_min, then just return the current weights.
        orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
        if orig_vtrust_loss <= vtrust_loss_desired:
            bt.logging.info("Weights already satisfy vtrust_min. {} >= {}.".format(1 - orig_vtrust_loss, vtrust_min))
            return weights

        # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
        # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
        vtrust_loss_min = 1 - np.sum(consensus)
        if vtrust_loss_min > vtrust_loss_desired:
            bt.logging.info(
                "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                    1 - vtrust_loss_min, vtrust_min
                )
            )
            vtrust_loss_desired = 1.05 * vtrust_loss_min

        # We could solve this with a LP, but just do rootfinding with scipy.
        consensus_normalized = consensus / np.sum(consensus)

        def fn(lam: float):
            new_weights = (1 - lam) * weights + lam * consensus_normalized
            vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
            return vtrust_loss - vtrust_loss_desired

        sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
        lam_opt = sol.root

        new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
        vtrust_pred = np.minimum(weights, consensus).sum()
        bt.logging.info(
            "Interpolated weights to satisfy vtrust_min. {} -> {}.".format(1 - orig_vtrust_loss, vtrust_pred)
        )
        return new_weights

    def try_set_scores_and_weights(self, ttl: int):
        def _try_set_weights():
            try:
                # Fetch latest metagraph
                metagraph = self.subtensor.metagraph(self.config.netuid)
                consensus = metagraph.C.cpu().numpy()
                cpu_weights = self.weights.cpu().numpy()
                adjusted_weights = self.adjust_for_vtrust(cpu_weights, consensus)
                self.weights = torch.tensor(adjusted_weights, dtype=torch.float32)
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=adjusted_weights,
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
                weights_report = {"weights": {}}
                for uid, score in enumerate(self.weights):
                    weights_report["weights"][uid] = score
                bt.logging.debug(weights_report)
            except Exception as e:
                bt.logging.error(f"failed to set weights {e}: {traceback.format_exc()}")
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        # Continually loop and set weights at the 20-minute mark
        while not self.stop_event.is_set():
            current_time = dt.datetime.utcnow()
            minutes = current_time.minute
            
            # Check if we're at a 20-minute mark for setting weights
            if minutes % 20 == 0 or self.config.immediate:
                try:
                    bt.logging.debug("Gathering scores and running win-rate competition.")
                    self._try_win_rate_competition()
                    bt.logging.debug("Finished running win-rate competition.")

                    if not self.config.dont_set_weights and not self.config.offline:
                        bt.logging.debug("Setting weights.")
                        _try_set_weights()
                        bt.logging.debug("Finished setting weights.")
                        if self.config.immediate:
                            time.sleep(3600)

                except asyncio.TimeoutError:
                    bt.logging.error(f"Failed to set weights after {ttl} seconds")

                except Exception as e:
                    bt.logging.error(f"Failed to set weights: {e}\n{traceback.format_exc()}")

            else:
                bt.logging.debug(f"Skipping setting weights. Only set weights at 20-minute marks.")

            # sleep for 1 minute before checking again
            time.sleep(60)

    def get_basic_auth(self) -> HTTPBasicAuth:
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        return HTTPBasicAuth(hotkey, signature)

    async def try_sync_metagraph(self, ttl: int):
        self.metagraph = bt.subtensor(self.subtensor.chain_endpoint).metagraph(self.config.netuid)
        bt.logging.info("Synced metagraph")
        self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.info("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.info("Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (dt.datetime.now() - self.last_update_check).seconds < self.update_check_interval:
            return False

        self.last_update_check = dt.datetime.now()

        return not is_git_latest()

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation (top sample_min from last run + newly updated models).
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """
        # Update self.metagraph
        await self.try_sync_metagraph(ttl=60 * 5)
        competition_parameters = constants.COMPETITION_SCHEDULE[
            self.global_step % len(constants.COMPETITION_SCHEDULE)
        ]
        
        uids = []
        # Query API for next model to score.
        bt.logging.info(f"Getting model to score...")
        uid = await self.get_model_to_score(competition_parameters.competition_id)

        if uid is not None:
            uids = [uid]

        if len(uids) == 0:
            bt.logging.debug(
                f"API returned no uid to eval for competition {competition_parameters.competition_id}."
            )
            return
        
        for uid in uids:
            hotkey = self.metagraph.hotkeys[uid]
            try:
                await self.model_updater.sync_model(hotkey)
                if (
                    self.model_tracker.get_model_metadata_for_miner_hotkey(
                        hotkey
                    )
                    is None
                ):
                    bt.logging.warning(
                        f"Unable to get metadata for UID {uid} with hotkey {hotkey}"
                    )
            except Exception as e:
                bt.logging.warning(
                    f"Unable to sync model for UID {uid} with hotkey {hotkey}: {traceback.format_exc()}"
                )

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        uid_to_block = defaultdict(lambda: math.inf)

        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {uids} for competition {competition_parameters.competition_id}"
        )
        scores_per_uid = {muid: None for muid in uids}

        self.model_tracker.release_all()
        uid_to_hotkey_and_model_metadata: typing.Dict[
            int, typing.Tuple[str, typing.Optional[ModelMetadata]]
        ] = {}
        for uid_i in uids:
            # Check that the model is in the tracker.
            hotkey = self.metagraph.hotkeys[uid_i]
            model_i_metadata = self.model_tracker.take_model_metadata_for_miner_hotkey(
                hotkey
            )
            bt.logging.info(f"Model metadata for {uid_i} is {model_i_metadata}")
            if model_i_metadata is not None:
                for other_uid, (
                    other_hotkey,
                    other_metadata,
                ) in uid_to_hotkey_and_model_metadata.items():
                    if (
                        other_metadata
                        and model_i_metadata.id.hash == other_metadata.id.hash
                    ):
                        if model_i_metadata.block < other_metadata.block:
                            bt.logging.info(
                                f"Perferring duplicate of {other_uid} with {uid_i} since it is older"
                            )
                            # Release the other model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(other_hotkey, other_metadata)
                            uid_to_hotkey_and_model_metadata[other_uid] = (
                                other_hotkey,
                                None,
                            )
                        else:
                            bt.logging.info(
                                f"Perferring duplicate of {uid_i} with {other_uid} since it is newer"
                            )
                            # Release own model since it is not in use.
                            self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)
                            model_i_metadata = None
                        break

            uid_to_hotkey_and_model_metadata[uid_i] = (hotkey, model_i_metadata)
            
        #bt.logging.info("Looking at model metadata", uid_to_hotkey_and_model_metadata)

        print(uid_to_hotkey_and_model_metadata)

        for uid_i, (
            hotkey,
            model_i_metadata,
        ) in uid_to_hotkey_and_model_metadata.items():
            score = None
            if model_i_metadata is not None:
                if (
                    model_i_metadata.id.competition_id
                    == competition_parameters.competition_id
                ):
                    self.model_tracker.touch_miner_model(hotkey)

                    try:
                        # Update the block this uid last updated their model.
                        uid_to_block[uid_i] = model_i_metadata.block
                        hf_repo_id = model_i_metadata.id.namespace + "/" + model_i_metadata.id.name
                        start_time = time.time()
                        scoring_inputs = ScoreModelInputs(
                            hf_repo_id=hf_repo_id,
                            competition_id=competition_parameters.competition_id,
                            hotkey=hotkey,
                            block=model_i_metadata.block
                        )
                        start_response = requests.post(
                            self.start_model_scoring_endpoint,
                            auth=self.get_basic_auth(),
                            json=json.loads(scoring_inputs.model_dump_json()),
                            timeout=30,
                        )
                        start_response.raise_for_status()
                        start_response_json = start_response.json()
                        if not start_response_json.get("success", False):
                            bt.logging.error(f"Failed to start scoring for {model_i_metadata}: {start_response_json.get('message', 'Unknown error')}")
                            return
                        is_scoring = True
                        while is_scoring:
                            time.sleep(15)
                            scoring_response = requests.get(
                                self.check_scoring_status_endpoint,
                                auth=self.get_basic_auth(),
                                timeout=30,
                            )
                            scoring_response.raise_for_status()
                            scoring_response_json = scoring_response.json()
                            is_scoring = scoring_response_json.get("status") == "scoring"
                            score = scoring_response_json["score"]
                        bt.logging.info(f"Score for {model_i_metadata} is {score}, took {time.time() - start_time} seconds")
                    except Exception as e:
                        bt.logging.error(
                            f"Error in eval loop: {e}. Setting score for uid: {uid_i} to 0. \n {traceback.format_exc()}"
                        )
                    finally:
                        # After we are done with the model, release it.
                        self.model_tracker.release_model_metadata_for_miner_hotkey(hotkey, model_i_metadata)
                else:
                    bt.logging.debug(
                        f"Skipping {uid_i}, submission is for a different competition ({model_i_metadata.id.competition_id}). Setting loss to 0."
                    )
            else:
                bt.logging.debug(
                    f"Unable to load the model for {uid_i} (perhaps a duplicate?). Setting loss to 0."
                )
            if score is None:
                bt.logging.error(f"Failed to get score for uid: {uid_i}: {model_i_metadata}")
                score = 0

            scores_per_uid[uid_i] = score

            bt.logging.debug(
                f"Computed model score for uid: {uid_i}: {score}"
            )
            bt.logging.debug(f"Computed model losses for uid: {uid_i}: {score}")

        # Post model scores to the API
        for uid, score in scores_per_uid.items():
            hotkey, model_metadata = uid_to_hotkey_and_model_metadata[uid]
            if model_metadata is not None:
                try:
                    # Check if the model hash is in the tracker
                    model_hash = ""
                    if hotkey in self.model_tracker.miner_hotkey_to_model_hash_dict:
                        model_hash = self.model_tracker.miner_hotkey_to_model_hash_dict[hotkey]
                    if not self.config.offline:
                        await self.post_model_score(hotkey, uid, model_metadata, model_hash, score)
                except Exception as e:
                    bt.logging.error(f"Failed to post model score for uid: {uid}: {model_metadata} {e}")
                    bt.logging.error(traceback.format_exc())

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def log_step(
        self,
        competition_id,
        uids,
        uid_to_block,
        wins,
        win_rate,
        scores_per_uid,
        sample_per_uid,
    ):
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "competition_id": competition_id,
            "uids": uids,
            "uid_data": {},
        }
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_block[uid],
                # "average_loss": (
                #     sum(losses_per_uid[uid]) / len(losses_per_uid[uid])
                #     if len(losses_per_uid[uid]) > 0
                #     else math.inf
                # ),
                "score": scores_per_uid[uid],
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
                "sample_prompt": (
                    sample_per_uid[uid][0] if sample_per_uid[uid] is not None else None
                ),
                "sample_response": (
                    sample_per_uid[uid][1] if sample_per_uid[uid] is not None else None
                ),
                "sample_truth": (
                    sample_per_uid[uid][2] if sample_per_uid[uid] is not None else None
                ),
            }
        table = Table(title=f"Step ({competition_id})")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["score"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)
        
        table = Table(title="Top 25 Miners by Win Rate")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        # Collect and sort data
        miner_data = []
        for uid in uids:
            str_uid = str(uid)
            try:
                if str_uid not in step_log["uid_data"]:
                    continue
                uid_data = step_log["uid_data"][str_uid]
                # if score is 0 from penalty, set win rate to 0
                if uid_data.get('score', 0.0) == 0.0:
                    uid_data['win_rate'] = 0.0
                    uid_data['win_total'] = 0
                miner_data.append({
                    'uid': uid,
                    'score': round(uid_data.get('score', 0.0), 4),
                    'win_rate': round(uid_data.get('win_rate', 0.0), 4),
                    'win_total': uid_data.get('win_total', 0),
                    'weight': round(float(self.weights[uid].item()), 4),
                    'block': uid_data.get('block', 'N/A')
                })
            except Exception as e:
                bt.logging.warning(f"Error processing UID {uid}: {str(e)}: {traceback.format_exc()}")
                continue

        # Sort by win_rate (descending) and take top 25
        sorted_miners = sorted(
            miner_data, 
            key=lambda x: x['win_rate'], 
            reverse=True
        )[:25]
        # Add rows to table
        for rank, miner in enumerate(sorted_miners, 1):
            table.add_row(
                str(miner['uid']),
                f"{miner['score']:.4f}",
                f"{miner['win_rate']:.4f}",
                str(miner['win_total']),
                f"{miner['weight']:.4f}",
                str(miner['block'])
            )
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        #bt.logging.info(f"Step results: {step_log}")

    async def get_model_to_score(self, competition_id: str) -> str:
        """
        Queries the SN21 API for the next model to score.

        Returns:
        - float: The reward value for the miner.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        try:
            async with ClientSession() as session:
                async with session.post(
                    self.get_model_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json={"competition_id": competition_id},
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    if "success" in response_json and not response_json["success"]:
                        bt.logging.warning(response_json["message"])
                        return None
                    elif "success" in response_json and response_json["success"]:
                        uid = int(response_json["miner_uid"])
                        bt.logging.info(f"Retrieved model to score, miner uid: {uid}")
                        return uid
                
        except Exception as e:
            traceback.print_exc()
            bt.logging.debug(f"Error retrieving model to score from API: {e}: {traceback.format_exc()}")
            return None
        
    async def post_model_score(self, miner_hotkey, miner_uid, model_metadata, model_hash, model_score) -> bool:
        """
        Posts the score of a model to the SN21 API.

        Returns:
        - bool: True if the score was successfully posted, False otherwise.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        try:
            async with ClientSession() as session:
                async with session.post(
                    self.score_model_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json={
                        "miner_hotkey": miner_hotkey,
                        "miner_uid": miner_uid,
                        "model_metadata": {
                            "id": model_metadata.id.to_compressed_str(),
                            "block": model_metadata.block
                        },
                        "model_hash": model_hash,
                        "model_score": model_score,
                    },
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    if "success" in response_json and not response_json["success"]:
                        bt.logging.warning(response_json["message"])
                    elif "success" in response_json and response_json["success"]:
                        bt.logging.info(f"Successfully posted model score for {miner_uid} with score {model_score}")
                    return True
            
        except Exception as e:
            bt.logging.error(f"Error posting model score to API: {e}: {traceback.format_exc()}")
            return None

    def get_all_model_scores(self) -> typing.Optional[typing.Dict[str, float]]:
        """
        Gets all the current scores of models from the SN21 API. Can't be async because it's called from a multiprocessing process.
        Will retry up to 3 times with a 5 second pause between retries if the request fails.

        Returns:
        - Dict[str, float]: A dictionary of model IDs to their scores, or None if all retries fail.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 5
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.get_all_model_scores_endpoint,
                    auth=self.get_basic_auth(),
                    timeout=30
                )
                response.raise_for_status()
                response_json = response.json()
                if "success" in response_json and not response_json["success"]:
                    bt.logging.warning(response_json["message"])
                    return None
                elif "success" in response_json and response_json["success"]:
                    model_scores = response_json["model_scores"]
                    bt.logging.info(f"Retrieved model scores from API")
                    return model_scores
                if attempt > 0:
                    bt.logging.debug(f"Successfully retrieved model scores after {attempt + 1} attempts")
            
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:  # Don't wait after the last attempt
                    bt.logging.debug(f"Get all model scores request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                    bt.logging.debug(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    bt.logging.debug(f"Final attempt failed. Request error: {str(e)}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:  # Don't wait after the last attempt
                    bt.logging.debug(f"Get all model scores unexpected error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                    bt.logging.debug(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    bt.logging.error(f"Final attempt failed. Unexpected error: {str(e)}: {traceback.format_exc()}")
        
        return None

    async def run(self):
        while True:
            try:
                
                if self.config.run_api:
                    # Update self.metagraph
                    await self.try_sync_metagraph(ttl=60 * 5)
                    # Sleep for 5 minutes before resycing metagraph
                    await asyncio.sleep(60 * 5)
                else:
                    await self.try_run_step(ttl=constants.MODEL_EVAL_TIMEOUT)
                
                self.global_step += 1

                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

                # Check if we should start a new wandb run.
                if not self.config.wandb.off and not self.config.offline and self.wandb_run_start != None:
                    if (dt.datetime.now() - self.wandb_run_start) >= dt.timedelta(
                        days=1
                    ):
                        bt.logging.info(
                            "Current wandb run is more than 1 day old. Starting a new run."
                        )
                        self.wandb_run.finish()
                        self.new_wandb_run()

                if self.config.auto_update and self.should_restart():
                    bt.logging.info(f'Validator is out of date, quitting to restart.')
                    raise KeyboardInterrupt

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught"
                )
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
