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

from collections import defaultdict
import datetime as dt
import os
import math
import time
import torch
import numpy as np
import shutil
from subprocess import Popen, PIPE
import asyncio
import argparse
import typing
import random
import constants

import bittensor as bt
import wandb
from scipy import optimize

from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.disk.utils import get_hf_download_path
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from datasets import disable_caching
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor
from utilities.temp_dir_cache import TempDirCache
from neurons.model_scoring import (
    get_model_score,
    pull_latest_omega_dataset,
    MIN_AGE,
    cleanup_gpu_memory,
    log_gpu_memory,
    get_gpu_memory
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MINS_TO_SLEEP = 10
GPU_MEM_GB_REQD = 39

def iswin(score_i, score_j, block_i, block_j):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust score based on timestamp and pretrain epsilon
    score_i = (1 - constants.timestamp_epsilon) * score_i if block_i > block_j else score_i
    score_j = (1 - constants.timestamp_epsilon) * score_j if block_j > block_i else score_j
    return score_i > score_j

def compute_wins(
    uids: typing.List[int],
    scores_per_uid: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        scores_per_uid (dict): A dictionary of losses for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = uid_to_block[uid_j]
            score_i = scores_per_uid[uid_i]
            score_j = scores_per_uid[uid_j]
            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j) else 0
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
            "--blocks_per_epoch",
            type=int,
            default=100,
            help="Number of blocks to wait before setting weights.",
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
            "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
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
        bt.logging(config=self.config)

        bt.logging.info(f"Starting validator with config: {self.config}")
        disable_caching()

        try:
            # early exit if GPU memory insufficient
            total_gb, used_gb, avail_gb = get_gpu_memory()
            if avail_gb < GPU_MEM_GB_REQD:
                m = f"Insufficient GPU Memory available: {avail_gb:.2f} GB available, out of total {total_gb:.2f} GB"
                bt.logging.error(m)
                raise RuntimeError(m)
        except Exception as e:
            bt.logging.error(f"Failed to get GPU memory: {e}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

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
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        
        self.uids_to_eval: typing.Dict[str, typing.Set] = {}

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.all_uids: typing.Dict[str, typing.Set] = {}
        self.all_uids_lock = threading.RLock()
        self.updated_uids_to_eval_lock = threading.RLock()
        self.updated_uids_to_eval: typing.Dict[str, typing.Set] = {}
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval: typing.Dict[str, typing.Set] = {}

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

        # Sync to consensus
        if not self.config.genesis:
            competition_ids: typing.Dict[int, typing.Optional[str]] = {}
            for uid, hotkey in enumerate(list(self.metagraph.hotkeys)):
                try:
                    metadata: typing.Optional[ModelMetadata] = asyncio.run(
                        self.metadata_store.retrieve_model_metadata(hotkey)
                    )
                    competition_ids[uid] = (
                        (
                            metadata.id.competition_id
                            if metadata.id.competition_id is not None
                            else constants.ORIGINAL_COMPETITION_ID
                        )
                        if metadata is not None and self.is_model_old_enough(metadata)
                        else None
                    )
                except Exception as e:
                    bt.logging.error(
                        f"Unable to get metadata for consensus UID {uid} with hotkey {hotkey}"
                    )
                    bt.logging.error(e)
                    competition_ids[uid] = None

            self.weights.copy_(self.metagraph.C)

            for competition in constants.COMPETITION_SCHEDULE:
                bt.logging.debug(
                    f"Building consensus state for competition {competition.competition_id}"
                )
                
                # Get all models for initial storage and first competition
                consensus = [
                    x[0]
                    for x in sorted(
                        [
                            (i, val.nan_to_num(0).item())
                            for (i, val) in enumerate(list(self.metagraph.consensus))
                            if competition_ids[i] == competition.competition_id
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]

                self.all_uids[competition.competition_id] = set(consensus)
                # Get the sample_total_models best models for for first competition
                consensus = consensus[: self.config.sample_total_models]
                self.uids_to_eval[competition.competition_id] = set(consensus)
                self.updated_uids_to_eval[competition.competition_id] = set()
                self.pending_uids_to_eval[competition.competition_id] = set()

                consensus_map = {uid: self.weights[uid].item() for uid in consensus}
                bt.logging.info(
                    f"Consensus for competition {competition.competition_id}: {consensus_map}"
                )

                for uid in consensus:
                    hotkey = self.metagraph.hotkeys[uid]
                    try:
                        asyncio.run(self.model_updater.sync_model(hotkey))
                        if (
                            self.model_tracker.get_model_metadata_for_miner_hotkey(
                                hotkey
                            )
                            is None
                        ):
                            bt.logging.warning(
                                f"Unable to get metadata for consensus UID {uid} with hotkey {hotkey}"
                            )
                    except Exception as e:
                        bt.logging.warning(
                            f"Unable to sync model for consensus UID {uid} with hotkey {hotkey}"
                        )

        # Touch all models, starting a timer for them to be deleted if not used
        self.model_tracker.touch_all_miner_models()
        
        # == Initialize the update thread ==
        self.stop_event = threading.Event()
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

        # == Initialize the weight setting thread ==
        if not self.config.dont_set_weights and not self.config.offline:
            self.weight_thread = threading.Thread(
                target=self.try_set_weights,
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
        is_old_enough = model_age > MIN_AGE
        if not is_old_enough:
            bt.logging.debug(
                f"Model {model_metadata.id} is too new to evaluate. Age: {model_age} seconds"
            )
        return is_old_enough

    def update_models(self, update_delay_minutes):
        # Track how recently we updated each uid
        uid_last_checked = dict()

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

                uid_last_checked[next_uid] = dt.datetime.now()
                bt.logging.debug(f"Updating model for UID={next_uid}")

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))
                
                # Ensure we eval the new model on the next loop.
                metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                if metadata is not None and self.is_model_old_enough(metadata):
                    with self.all_uids_lock:
                        self.all_uids[metadata.id.competition_id].add(next_uid)
                    
                    bt.logging.trace(f"Updated model for UID={next_uid}. Was new = {updated}")
                    if updated:
                        with self.updated_uids_to_eval_lock:
                            self.updated_uids_to_eval[metadata.id.competition_id].add(next_uid)
                            bt.logging.debug(f"Found a new model for UID={next_uid} for competition {metadata.id.competition_id}. It will be evaluated on the next loop.")
                    else:
                        with self.pending_uids_to_eval_lock:
                            self.pending_uids_to_eval[metadata.id.competition_id].add(next_uid)
                else:
                    bt.logging.debug(f"Unable to sync model for consensus UID {next_uid} with hotkey {hotkey}")

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
                bt.logging.error(f"Error in clean loop: {e}")
                print(traceback.format_exc())

            time.sleep(dt.timedelta(minutes=clean_period_minutes).total_seconds())

        bt.logging.info("Exiting clean models loop.")

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

    def try_set_weights(self, ttl: int):
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
                bt.logging.error(f"failed to set weights {e}")
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
                    bt.logging.debug("Setting weights.")
                    _try_set_weights()
                    bt.logging.debug("Finished setting weights.")
                    if self.config.immediate:
                        time.sleep(3600)
                except asyncio.TimeoutError:
                    bt.logging.error(f"Failed to set weights after {ttl} seconds")
            else:
                bt.logging.debug(f"Skipping setting weights. Only set weights at 20-minute marks.")

            # sleep for 1 minute before checking again
            time.sleep(60)


    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            # Update self.metagraph
            self.metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            self.metagraph.save()
            bt.logging.info(f"Metagraph synced and saved")

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        self.metagraph.load()
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

    def is_git_latest(self) -> bool:
        p = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        current_commit = out.decode().strip()
        p = Popen(['git', 'ls-remote', 'origin', 'HEAD'], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        latest_commit = out.decode().split()[0]
        bt.logging.info(f'Current commit: {current_commit}, Latest commit: {latest_commit}')
        return current_commit == latest_commit

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (dt.datetime.now() - self.last_update_check).seconds < self.update_check_interval:
            return False

        self.last_update_check = dt.datetime.now()

        return not self.is_git_latest()

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
        cleanup_gpu_memory()
        log_gpu_memory('at start of run_step')

        # Update self.metagraph
        await self.try_sync_metagraph(ttl=60 * 5)
        competition_parameters = constants.COMPETITION_SCHEDULE[
            self.global_step % len(constants.COMPETITION_SCHEDULE)
        ]

        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            # Get the current UIDs to evaluate
            current_uids = self.uids_to_eval[competition_parameters.competition_id]
            
            with self.updated_uids_to_eval_lock:
                if len(current_uids) < self.config.sample_total_models:
                    updated_uids = list(self.updated_uids_to_eval[competition_parameters.competition_id])
                    selected_updated_uids = set(random.sample(
                        updated_uids,
                        min(self.config.sample_updated_models, len(updated_uids))
                    ))
                    current_uids |= selected_updated_uids
                    self.updated_uids_to_eval[competition_parameters.competition_id] -= selected_updated_uids

            with self.pending_uids_to_eval_lock:
                if len(current_uids) < self.config.sample_total_models:
                    pending_uids = list(self.pending_uids_to_eval[competition_parameters.competition_id])
                    selected_pending_uids = set(random.sample(
                        pending_uids,
                        min(self.config.sample_total_models - len(current_uids), len(pending_uids))
                    ))
                    current_uids |= selected_pending_uids
                    self.pending_uids_to_eval[competition_parameters.competition_id] -= selected_pending_uids

            # Fallback mechanism to ensure `current_uids` is filled up to `sample_total_models`
            if len(current_uids) < self.config.sample_total_models:
                all_uids = self.all_uids[competition_parameters.competition_id]
                remaining_needed = self.config.sample_total_models - len(current_uids)
                # grab random sample of uids not already in current_uids
                additional_uids = set(random.sample(
                    all_uids - current_uids,
                    min(remaining_needed, len(all_uids - current_uids))
                ))
                current_uids |= additional_uids

            # Update `uids_to_eval` with the final set of UIDs
            self.uids_to_eval[competition_parameters.competition_id] = current_uids

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval[competition_parameters.competition_id])

        if not uids:
            if self.config.genesis:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}. Waiting 5 minutes to download some models."
                )
                time.sleep(300)
            else:
                bt.logging.debug(
                    f"No uids to eval for competition {competition_parameters.competition_id}."
                )
            return

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        uid_to_block = defaultdict(lambda: math.inf)

        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {uids} for competition {competition_parameters.competition_id}"
        )
        scores_per_uid = {muid: None for muid in uids}
        sample_per_uid = {muid: None for muid in uids}

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

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
            
        bt.logging.info("Looking at model metadata", uid_to_hotkey_and_model_metadata)

        eval_data = pull_latest_omega_dataset()
        log_gpu_memory('after pulling dataset')
        if eval_data is None:
            bt.logging.warning(
                f"No data is currently available to evalute miner models on, sleeping for {MINS_TO_SLEEP} minutes."
            )
            time.sleep(MINS_TO_SLEEP * 60)

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
                        score = get_model_score(
                            hf_repo_id,
                            mini_batch=eval_data,
                            local_dir=self.temp_dir_cache.get_temp_dir(hf_repo_id),
                            hotkey=hotkey,
                            block=model_i_metadata.block,
                            model_tracker=self.model_tracker
                        )
                        bt.logging.info(f"Score for {model_i_metadata} is {score}")
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
            if not score:
                bt.logging.error(f"Failed to get score for uid: {uid_i}: {model_i_metadata}")
                score = 0

            scores_per_uid[uid_i] = score

            bt.logging.debug(
                f"Computed model score for uid: {uid_i}: {score}"
            )
            bt.logging.debug(f"Computed model losses for uid: {uid_i}: {score}")

        # let's loop through the final scores and make sure all model's are not copycats
        for uid, score in scores_per_uid.items():
            if score > 0:
                hotkey, model_metadata = uid_to_hotkey_and_model_metadata[uid]
                for other_uid, other_score in scores_per_uid.items():
                    if other_score > 0 and other_uid != uid:
                        other_hotkey, other_model_metadata = uid_to_hotkey_and_model_metadata[other_uid]
                        if hotkey in self.model_tracker.miner_hotkey_to_last_touched_dict \
                            and other_hotkey in self.model_tracker.miner_hotkey_to_model_hash_dict \
                            and self.model_tracker.miner_hotkey_to_model_hash_dict[hotkey] == self.model_tracker.miner_hotkey_to_model_hash_dict[other_hotkey]:

                            if model_metadata.block > other_model_metadata.block:
                                bt.logging.info(f"*** Model from {hotkey} on block {model_metadata.block} is a copycat of {other_hotkey} on block {other_model_metadata.block} and will be scored 0. ***")
                                scores_per_uid[uid] = 0

        # Compute wins and win rates per uid.
        wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block)
        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.metagraph.S)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        scale = (
            len(constants.COMPETITION_SCHEDULE)
            * competition_parameters.reward_percentage
        )
        new_weights *= scale / new_weights.sum()
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

        # Filter based on win rate removing all but the sample_top_models best models for evaluation.
        self.uids_to_eval[competition_parameters.competition_id] = set(
            sorted(win_rate, key=win_rate.get, reverse=True)[: self.config.sample_top_models]
        )

        # Log the performance of the eval loop.
        bt.logging.debug(load_model_perf.summary_str())
        bt.logging.debug(compute_loss_perf.summary_str())

        # Log to screen.
        self.log_step(
            competition_parameters.competition_id,
            uids,
            uid_to_block,
            wins,
            win_rate,
            scores_per_uid,
            sample_per_uid,
        )

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
        table = Table(title="Step")
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
        bt.logging.info(f"Step results: {step_log}")

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 50)
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
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
