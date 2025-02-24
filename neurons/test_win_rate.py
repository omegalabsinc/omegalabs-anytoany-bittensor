import os
os.environ["USE_TORCH"] = "1"
os.environ["BT_LOGGING_INFO"] = "1"

import datetime as dt
import math
import time
import torch
import numpy as np
import traceback
import bittensor as bt
import requests
import json
import constants
from typing import Dict, List, Optional, Tuple
from rich.table import Table
from rich.console import Console
from scipy import optimize
import argparse
import pandas as pd
import matplotlib.pyplot as plt


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
    uids: List[int],
    scores_per_uid: Dict[int, float], 
    uid_to_block: Dict[int, int],
    uid_to_hash: Dict[int, str] = None,
) -> Tuple[Dict[int, int], Dict[int, float]]:
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



def iswin_old(score_i, score_j, block_i, block_j):
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

def compute_wins_old(
    uids: List[int],
    scores_per_uid: Dict[int, float],
    uid_to_block: Dict[int, int],
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        scores_per_uid (dict): A dictionary of losses for each uid by batch.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}

    # If there is only one score, then the winning uid is the one with the score.
    if len([score for score in scores_per_uid.values() if score != 0]) == 1:
        winning_uid = [uid for uid, score in scores_per_uid.items() if score != 0][0]
        wins[winning_uid] = 1
        win_rate[winning_uid] = 1.0
        return wins, win_rate

    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        # Skip if block is None or score is 0
        if block_i is None or scores_per_uid[uid_i] == 0:
            continue
        for j, uid_j in enumerate(uids):
            if i == j or uid_to_block[uid_j] is None or scores_per_uid[uid_j] == 0:
                continue
            block_j = uid_to_block[uid_j]
            score_i = scores_per_uid[uid_i]
            score_j = scores_per_uid[uid_j]
            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


class TestWinRate:
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

    def __init__(self):
        self.config = TestWinRate.config()
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.weights = torch.zeros_like(self.metagraph.S)
        
        # Set up API endpoints - use localhost for testing
        self.get_all_model_scores_endpoint = "http://localhost:8000/get-all-model-scores"
        bt.logging.info(f"Using API endpoint: {self.get_all_model_scores_endpoint}")



    def _try_win_rate_competition(self, temperature=0.0001):
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
            
            for uid_str, models_data in curr_model_scores.items():
                if not models_data:  # Skip if no models for this UID
                    continue
                
                # Convert UID to int
                uid = int(uid_str)
                # Take the first model's data (assuming one model per UID)
                model_data = models_data[0]
                
                # Extract score, block and hash, defaulting to appropriate values if not present
                score = model_data.get('score', 0)
                block = model_data.get('block', None)
                model_hash = model_data.get('model_hash', None)
                
                # Skip if score is None
                if score is None:
                    continue
                    
                # Only update values if they are valid
                scores_per_uid[uid] = float(score)
                if block is not None:
                    uid_to_block[uid] = int(block)
                if model_hash is not None and model_hash != "":
                    uid_to_hash[uid] = model_hash

            # Compute wins and win rates per uid
            wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)

            # Loop through all models and check for duplicate model hashes
            uids_penalized_for_hash_duplication = set()
            for uid in uids:
                if uid_to_hash[uid] is None or uid_to_block[uid] is None:
                    continue
                for uid2 in uids:
                    if uid == uid2 or uid_to_hash[uid] is None or uid_to_block[uid2] is None:
                        continue
                    if uid != uid2 and uid_to_hash[uid] == uid_to_hash[uid2]:
                        if uid_to_block[uid] < uid_to_block[uid2]:
                            scores_per_uid[uid2] = 0
                            win_rate[uid2] = 0
                            wins[uid2] = 0
                            uids_penalized_for_hash_duplication.add(uid2)
                            bt.logging.warning(f"uid {uid2} at block {uid_to_block[uid2]} has duplicate model hash of uid {uid} at block {uid_to_block[uid]}. Penalizing uid {uid2} with score of 0.")
                        else:
                            scores_per_uid[uid] = 0
                            win_rate[uid] = 0
                            wins[uid] = 0
                            uids_penalized_for_hash_duplication.add(uid)
                            bt.logging.warning(f"uid {uid} at block {uid_to_block[uid]} has duplicate model hash of uid {uid2} at block {uid_to_block[uid2]}. Penalizing uid {uid} with score of 0.")

            # Compute softmaxed weights based on win rate
            model_weights = torch.tensor(
                [win_rate[uid] for uid in uids], dtype=torch.float32
            )
            
            step_weights = torch.softmax(model_weights / temperature, dim=0)
            
            for i, uid_i in enumerate(uids):
                curr_comp_weights[uid_i] = step_weights[i]
            scale = competition_parameters.reward_percentage
            curr_comp_weights *= scale / curr_comp_weights.sum()
            new_weights += curr_comp_weights

        # Do EMA with existing (self.weights) and new (new_weights) weights
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

        # Now log after weights have been updated
        for competition_parameters in constants.COMPETITION_SCHEDULE:
            curr_model_scores = {
                uid: models_data
                for uid, models_data in all_model_scores.items()
                if models_data[0]["competition_id"] == competition_parameters.competition_id
            }
            
            uids = self.metagraph.uids.tolist()
            uids = [int(uid) for uid in uids]
            
            scores_per_uid = {uid: 0 for uid in uids}
            uid_to_block = {uid: None for uid in uids}
            uid_to_hash = {uid: None for uid in uids}
            sample_per_uid = {muid: None for muid in uids}

            for uid_str, models_data in curr_model_scores.items():
                if not models_data:
                    continue
                uid = int(uid_str)
                model_data = models_data[0]
                score = float(model_data.get('score', 0))
                block = model_data.get('block', None)
                model_hash = model_data.get('model_hash', None)
                
                if score is not None:
                    scores_per_uid[uid] = score
                if block is not None:
                    uid_to_block[uid] = int(block)
                if model_hash is not None and model_hash != "":
                    uid_to_hash[uid] = model_hash

            wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)
            
            # Check for duplicates and update win rates
            for uid in uids:
                if uid_to_hash.get(uid) is not None:
                    for uid2 in uids:
                        if (uid != uid2 and 
                            uid_to_hash.get(uid2) is not None and 
                            uid_to_hash[uid] == uid_to_hash[uid2]):
                            if uid_to_block[uid] < uid_to_block[uid2]:
                                win_rate[uid2] = 0.0
                                wins[uid2] = 0
                            else:
                                win_rate[uid] = 0.0
                                wins[uid] = 0
            
            self.log_step(
                competition_parameters.competition_id,
                uids,
                uid_to_block,
                wins,
                win_rate,
                scores_per_uid,
                sample_per_uid,
                temperature,
            )

    def _try_win_rate_new(self, temperature=0.0001):
        """
        Enhanced version of win-rate competition that processes model scores and determines weights.
        This version includes improved error handling, cleaner code organization, and better logging.
        
        Args:
            temperature (float): Temperature parameter for softmax calculation
        """
        try:
            # Get model scores from API
            all_model_scores = self.get_all_model_scores()
            if all_model_scores is None:
                bt.logging.error("No model scores returned from API.")
                return

            # Initialize weights tensor
            new_weights = torch.zeros_like(self.metagraph.S)
            
            # Process each competition in the schedule
            for competition_parameters in constants.COMPETITION_SCHEDULE:
                # Initialize competition weights
                curr_comp_weights = torch.zeros_like(self.metagraph.S)
                
                # Get UIDs and convert to int
                uids = [int(uid) for uid in self.metagraph.uids.tolist()]
                
                # Initialize score tracking dictionaries
                scores_per_uid = {uid: 0 for uid in uids}
                uid_to_block = {uid: None for uid in uids}
                uid_to_hash = {uid: None for uid in uids}
                sample_per_uid = {uid: None for uid in uids}
                uid_to_non_zero_scores = {uid: 0 for uid in uids}  # Track non-zero scores per UID

                # Get scores for current competition
                curr_model_scores = {
                    uid: models_data
                    for uid, models_data in all_model_scores.items()
                    if models_data[0]["competition_id"] == competition_parameters.competition_id
                }

                # Process each model's scores
                for uid_str, models_data in curr_model_scores.items():
                    if not models_data:
                        continue
                    
                    uid = int(uid_str)
                    model_data = models_data[0]
                    
                    # Extract score data
                    score = model_data.get('score', 0)
                    block = model_data.get('block', None)
                    model_hash = model_data.get('model_hash', None)
                    score_details = model_data.get('score_details', [])

                    # Count non-zero scores from score_details
                    if score_details:
                        non_zero_scores = sum(1 for detail in score_details if detail.get('score', 0) > 0)
                        uid_to_non_zero_scores[uid] = non_zero_scores
                    
                    if uid == 58:
                        bt.logging.info(f"uid {uid} has {uid_to_non_zero_scores[uid]} non-zero scores")
                    
                    if model_hash == "":
                        model_hash = None
                    
                    # Store valid data
                    if score is not None:
                        # Only assign score if minimum non-zero scores requirement is met
                        if uid_to_non_zero_scores[uid] >= constants.MIN_NON_ZERO_SCORES:
                            scores_per_uid[uid] = float(score)
                    if block is not None:
                        uid_to_block[uid] = int(block)
                    if model_hash is not None:
                        uid_to_hash[uid] = model_hash

                # Handle duplicate model hashes
                uids_penalized_for_hash_duplication = set()
                for uid in uids:
                    if uid_to_hash.get(uid) is None or uid_to_block.get(uid) is None:
                        continue
                    
                    for uid2 in uids:
                        if (uid != uid2 and 
                            uid_to_hash.get(uid2) is not None and 
                            uid_to_block.get(uid2) is not None and 
                            uid_to_hash[uid] == uid_to_hash[uid2]):
                            
                            if uid_to_block[uid] < uid_to_block[uid2]:
                                scores_per_uid[uid2] = 0
                                uids_penalized_for_hash_duplication.add(uid2)
                                bt.logging.warning(
                                    f"uid {uid2} at block {uid_to_block[uid2]} has duplicate "
                                    f"model hash of uid {uid} at block {uid_to_block[uid]}. "
                                    f"Penalizing uid {uid2}."
                                )
                            else:
                                scores_per_uid[uid] = 0
                                uids_penalized_for_hash_duplication.add(uid)
                                bt.logging.warning(
                                    f"uid {uid} at block {uid_to_block[uid]} has duplicate "
                                    f"model hash of uid {uid2} at block {uid_to_block[uid2]}. "
                                    f"Penalizing uid {uid}."
                                )

                # Compute wins and win rates
                wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)

                # Calculate weights using softmax
                model_weights = torch.tensor([win_rate[uid] for uid in uids], dtype=torch.float32)
                step_weights = torch.softmax(model_weights / temperature, dim=0)
                
                # Apply weights to competition
                for i, uid_i in enumerate(uids):
                    curr_comp_weights[uid_i] = step_weights[i]
                
                # Scale weights by reward percentage
                curr_comp_weights *= competition_parameters.reward_percentage / curr_comp_weights.sum()
                new_weights += curr_comp_weights

            # Update weights using EMA
            if new_weights.shape[0] < self.weights.shape[0]:
                self.weights = self.weights[:new_weights.shape[0]]
            elif new_weights.shape[0] > self.weights.shape[0]:
                self.weights = torch.cat([
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0])
                ])

            self.weights = (
                constants.alpha * self.weights + 
                (1 - constants.alpha) * new_weights
            )
            self.weights = self.weights.nan_to_num(0.0)

            # Now log after weights have been updated
            for competition_parameters in constants.COMPETITION_SCHEDULE:
                curr_model_scores = {
                    uid: models_data
                    for uid, models_data in all_model_scores.items()
                    if models_data[0]["competition_id"] == competition_parameters.competition_id
                }
                
                uids = [int(uid) for uid in self.metagraph.uids.tolist()]
                
                scores_per_uid = {uid: 0 for uid in uids}
                uid_to_block = {uid: None for uid in uids}
                uid_to_hash = {uid: None for uid in uids}
                sample_per_uid = {uid: None for uid in uids}

                for uid_str, models_data in curr_model_scores.items():
                    if not models_data:
                        continue
                    uid = int(uid_str)
                    model_data = models_data[0]
                    score = model_data.get('score', 0)
                    block = model_data.get('block', None)
                    model_hash = model_data.get('model_hash', None)
                    
                    if score is not None:
                        scores_per_uid[uid] = float(score)
                    if block is not None:
                        uid_to_block[uid] = int(block)
                    if model_hash is not None and model_hash != "":
                        uid_to_hash[uid] = model_hash

                wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)
                
                # Check for duplicates and update win rates
                for uid in uids:
                    if uid_to_hash.get(uid) is not None:
                        for uid2 in uids:
                            if (uid != uid2 and 
                                uid_to_hash.get(uid2) is not None and 
                                uid_to_hash[uid] == uid_to_hash[uid2]):
                                if uid_to_block[uid] < uid_to_block[uid2]:
                                    win_rate[uid2] = 0.0
                                    wins[uid2] = 0
                                else:
                                    win_rate[uid] = 0.0
                                    wins[uid] = 0
                
                self.log_step(
                    competition_parameters.competition_id,
                    uids,
                    uid_to_block,
                    wins,
                    win_rate,
                    scores_per_uid,
                    sample_per_uid,
                    temperature
                )

        except Exception as e:
            bt.logging.error(f"Error in win rate competition: {str(e)}\n{traceback.format_exc()}")

    def get_basic_auth(self):
        """Get basic auth credentials for API requests"""
        # Use the hardcoded test credentials that match the API
        dummy_hotkey = "5G77777777777777777777777777777777777777777777777777777777777777"
        return requests.auth.HTTPBasicAuth(dummy_hotkey, "signature")

    def get_all_model_scores(self) -> Optional[Dict[str, float]]:
        """
        Gets all the current scores of models from the SN21 API.
        Will retry up to 3 times with a 5 second pause between retries if the request fails.

        Returns:
            Dict[str, float]: A dictionary of model IDs to their scores, or None if all retries fail.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 5
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.get_all_model_scores_endpoint,
                    auth=self.get_basic_auth(),
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                response_json = response.json()
                
                bt.logging.debug(f"API Response: {response_json}")
                
                if "success" in response_json and not response_json["success"]:
                    bt.logging.warning(f"API returned failure: {response_json.get('message', 'No error message')}")
                    return None
                elif "success" in response_json and response_json["success"]:
                    model_scores = response_json.get("model_scores")
                    if not model_scores:
                        bt.logging.warning("API returned success but no model scores")
                        return None
                    bt.logging.info(f"Retrieved {len(model_scores)} model scores from API")
                    return model_scores
                else:
                    bt.logging.warning("API response missing success field")
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:  # Don't wait after the last attempt
                    bt.logging.debug(f"Get all model scores request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                    bt.logging.debug(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    bt.logging.error(f"Final attempt failed. Request error: {str(e)}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:  # Don't wait after the last attempt
                    bt.logging.debug(f"Get all model scores unexpected error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                    bt.logging.debug(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    bt.logging.error(f"Final attempt failed. Unexpected error: {str(e)}: {traceback.format_exc()}")
        
        return None
    
    def parse_get_all_model_scores(self):
        model_scores = self.get_all_model_scores()
        if not model_scores:
            bt.logging.warning("API returned success but no model scores")
            return None
        # print(type(model_scores))
        # print(model_scores.keys())
        # print(model_scores.get('1')[0].keys())
        # exit()
            
        # Convert model scores to a list of dictionaries
        records = []
        for uid, models in model_scores.items():
            for model in models:
                record = {
                    'uid': uid,
                    'competition_id': model.get('competition_id'),
                    'score': model.get('score'),
                    'block': model.get('block'),
                    'model_hash': model.get('model_hash')
                }
                records.append(record)
                
        # Create pandas DataFrame
        df = pd.DataFrame(records)
        
        # Split into v1 and o1 dataframes
        v1_df = df[df['competition_id'] == constants.V1_MODEL_ID]
        o1_df = df[df['competition_id'] == constants.O1_MODEL_ID]
        
        # Save to separate CSV files
        v1_df.to_csv('v1_model_scores.csv', index=False)
        o1_df.to_csv('o1_model_scores.csv', index=False)
        
        bt.logging.info(f"Parsed {len(v1_df)} V1 model scores and {len(o1_df)} O1 model scores into separate CSVs")
        return df

    def log_step(
        self,
        competition_id,
        uids,
        uid_to_block,
        wins,
        win_rate,
        scores_per_uid,
        sample_per_uid,
        temperature,
    ):
        """
        Logs the step results including scores, wins, and win rates for each model.
        
        Parameters:
            competition_id (str): The ID of the current competition
            uids (List[int]): List of UIDs being evaluated
            uid_to_block (Dict[int, int]): Mapping of UIDs to their block numbers
            wins (Dict[int, int]): Number of wins per UID
            win_rate (Dict[int, float]): Win rate per UID
            scores_per_uid (Dict[int, float]): Scores per UID
            sample_per_uid (Dict[int, Tuple]): Sample data per UID
        """
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
        # console = Console()
        # console.print(table)
        
        table = Table(title="Top 10 Miners by Win Rate Competition: " + competition_id + " Temperature: " + str(temperature))
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
        )[:10]
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

    def plot_weight_comparison(self, temperatures=[0.00001, 0.0001, 0.001, 0.01, 0.1]):
        """
        Plot weight distributions for different temperatures on the same graph.
        
        Args:
            temperatures (list): List of temperature values to compare
        """
        try:
            all_model_scores = self.get_all_model_scores()
            if all_model_scores is None:
                bt.logging.error("No model scores returned from API.")
                return
            
            plt.figure(figsize=(15, 10))
            
            for temp in temperatures:
                new_weights = torch.zeros_like(self.metagraph.S)
                
                for competition_parameters in constants.COMPETITION_SCHEDULE:
                    curr_comp_weights = torch.zeros_like(self.metagraph.S)
                    uids = self.metagraph.uids.tolist()
                    uids = [int(uid) for uid in uids]
                    
                    scores_per_uid = {uid: 0 for uid in uids}
                    uid_to_block = {uid: None for uid in uids}
                    uid_to_hash = {uid: None for uid in uids}
                    
                    curr_model_scores = {
                        uid: models_data
                        for uid, models_data in all_model_scores.items()
                        if models_data[0]["competition_id"] == competition_parameters.competition_id
                    }
                    
                    for uid_str, models_data in curr_model_scores.items():
                        if not models_data:
                            continue
                        
                        uid = int(uid_str)
                        model_data = models_data[0]
                        
                        score = model_data.get('score', 0)
                        block = model_data.get('block', None)
                        model_hash = model_data.get('model_hash', None)
                        
                        if score is not None:
                            scores_per_uid[uid] = float(score)
                        if block is not None:
                            uid_to_block[uid] = int(block)
                        if model_hash is not None and model_hash != "":
                            uid_to_hash[uid] = model_hash
                    
                    wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block, uid_to_hash)
                    
                    # Check for duplicates and update win rates
                    for uid in uids:
                        if uid_to_hash.get(uid) is not None:
                            for uid2 in uids:
                                if (uid != uid2 and 
                                    uid_to_hash.get(uid2) is not None and 
                                    uid_to_hash[uid] == uid_to_hash[uid2]):
                                    if uid_to_block[uid] < uid_to_block[uid2]:
                                        win_rate[uid2] = 0.0
                                    else:
                                        win_rate[uid] = 0.0
                    
                    model_weights = torch.tensor([win_rate[uid] for uid in uids], dtype=torch.float32)
                    step_weights = torch.softmax(model_weights / temp, dim=0)
                    
                    for i, uid_i in enumerate(uids):
                        curr_comp_weights[uid_i] = step_weights[i]
                    
                    scale = competition_parameters.reward_percentage
                    curr_comp_weights *= scale / curr_comp_weights.sum()
                    new_weights += curr_comp_weights
                
                # Sort weights for plotting
                sorted_weights, _ = torch.sort(new_weights, descending=True)
                plt.plot(range(len(sorted_weights)), 
                        sorted_weights.numpy(), 
                        label=f'Temperature={temp}',
                        alpha=0.7)
            
            plt.xlabel('Rank')
            plt.ylabel('Weight')
            plt.title('Weight Distribution for Different Temperatures')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Use log scale for better visualization
            plt.savefig('weight_distribution.png')
            plt.close()
            
            bt.logging.info(f"Weight distribution plot saved as 'weight_distribution.png'")
            
        except Exception as e:
            bt.logging.error(f"Error plotting weight comparison: {str(e)}: {traceback.format_exc()}")

if __name__ == "__main__":
    test_win_rate = TestWinRate()
    print("Starting win rate test...")
    print("Config:", test_win_rate.config)
    print("Metagraph shape:", test_win_rate.metagraph.S.shape)
    print("----------------------------------------------------------------")
    
    # Plot weight distributions for different temperatures
    # test_win_rate.plot_weight_comparison(temperatures=[0.005, 0.001, 0.08, 0.1, 0.01])
    
    # Run the original competition with very low temperature

    # test_win_rate._try_win_rate_competition(temperature=constants.temperature)

    print("###############################"*5)
    
    # Run the new competition with very low temperature
    test_win_rate._try_win_rate_new(temperature=constants.temperature)

    
