from typing import Annotated, List, Optional
from traceback import print_exception
from datetime import datetime, timezone
import threading
import time
import os
from collections import OrderedDict

import bittensor
import uvicorn
import asyncio
import logging
import json
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair
from pydantic import BaseModel

from model.storage.mysql_model_queue import init_database, ModelQueueManager
from model.storage.eval_leaderboard import init_database as init_eval_database, EvalLeaderboardManager
from model.storage.reputation_store import init_database as init_rep_database, ReputationStore
from vali_api.config import NETWORK, NETUID, IS_PROD, SENTRY_DSN
from constants import MODEL_EVAL_TIMEOUT, MIN_NON_ZERO_SCORES, penalty_score, deviation_percent
from utilities.compare_block_and_model import compare_block_and_model

import sentry_sdk
print("SENTRY_DSN:", SENTRY_DSN)
sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

security = HTTPBasic()

# Cache for block and model comparisons with size limit
MAX_CACHE_SIZE = 1000  # Limit the number of cached entries
_block_model_cache = OrderedDict()  # OrderedDict maintains insertion order for simple LRU
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}

# File-based cache for metagraph data to avoid issues with resyncing during API calls
METAGRAPH_CACHE_FILE = "temp/meta.json"
_metagraph_file_lock = threading.Lock()

def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )

def authenticate_with_bittensor(hotkey, metagraph):
    if hotkey not in metagraph.hotkeys:
        print(f"Hotkey not found in metagraph.")
        return False

    uid = metagraph.hotkeys.index(hotkey)
    if not metagraph.validator_permit[uid] and NETWORK != "test":
        print("Bittensor validator permit required")
        return False

    if metagraph.S[uid] < 20000 and NETWORK != "test":
        print(f"Bittensor validator requires 20,000+ staked TAO, but validator={hotkey} only has {metagraph.S[uid]} staked")
        return False

    return True


def filter_scores(scores, deviation_percent=deviation_percent):
    """
    Filter scores using median-based approach:
    Calculate the median score
    Define acceptable range as median ± (deviation_percent * median)
    Only keep scores within that range and any penalty scores (0)
    If too few scores remain, return original list
    """
    if len(scores) < 3:
        return scores # Not enough scores to filter
    

    
    # Separate penalty scores (0) from regular scores
    penalty_scores = [s for s in scores if s['score'] == penalty_score]
    regular_scores = [s for s in scores if s['score'] > penalty_score]
    
    
    if len(regular_scores) < 3:
        return scores # Not enough non-penalty scores to filter
    
    # Calculate median score from non-penalty scores
    score_values = [s['score'] for s in regular_scores]
    median_score = sorted(score_values)[len(score_values) // 2]
    
    # Define acceptable range around median
    acceptable_deviation = deviation_percent * median_score
    lower_bound = median_score - acceptable_deviation
    upper_bound = median_score + acceptable_deviation
    
    # Filter regular scores
    filtered_scores = [s for s in regular_scores if lower_bound <= s['score'] <= upper_bound]
    
    # If we filtered too much, fall back to original scores
    if len(filtered_scores) < 2:
        return scores
    
    # Create a new list with filtered regular scores and original penalty scores
    result = filtered_scores.copy()
    result.extend(penalty_scores)
    return result


def get_cached_metagraph_file_only():
    """
    Get cached metagraph data from temp/meta.json file ONLY.
    Does not fall back to live metagraph to avoid network connections.
    Returns None if file is not available.
    """
    
    with _metagraph_file_lock:
        try:
            # Try to read cached data from file
            if os.path.exists(METAGRAPH_CACHE_FILE):
                with open(METAGRAPH_CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                print(f"Loaded metagraph cache from {METAGRAPH_CACHE_FILE} with {len(cached_data.get('hotkeys', []))} hotkeys")
                return {
                    'hotkeys': cached_data.get('hotkeys', []),
                    'stakes': cached_data.get('stakes', {})
                }
            else:
                print(f"Metagraph cache file {METAGRAPH_CACHE_FILE} not found")
                return None
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error reading metagraph cache file: {e}")
            return None


def get_cached_metagraph(current_metagraph):
    """
    Get cached metagraph data from temp/meta.json file.
    Returns cached data if file exists, otherwise falls back to current metagraph.
    """
    
    with _metagraph_file_lock:
        try:
            # Try to read cached data from file
            if os.path.exists(METAGRAPH_CACHE_FILE):
                with open(METAGRAPH_CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                print(f"Loaded metagraph cache from {METAGRAPH_CACHE_FILE} with {len(cached_data.get('hotkeys', []))} hotkeys")
                return {
                    'hotkeys': cached_data.get('hotkeys', []),
                    'stakes': cached_data.get('stakes', {})
                }
            else:
                print(f"Metagraph cache file {METAGRAPH_CACHE_FILE} not found, using fallback")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error reading metagraph cache file: {e}, using fallback")
        
        # Fallback to current metagraph data if file doesn't exist or is corrupted
        try:
            fallback_data = {
                'hotkeys': list(current_metagraph.hotkeys),
                'stakes': {hotkey: float(current_metagraph.S[i].item()) 
                          for i, hotkey in enumerate(current_metagraph.hotkeys)}
            }
            print("Using fallback metagraph data (file cache not available)")
            return fallback_data
        except Exception as e:
            print(f"Error creating fallback metagraph data: {e}")
            # Return empty data structure as last resort
            return {'hotkeys': [], 'stakes': {}}


def calculate_stake_weighted_scores_cached(recent_model_scores, cached_metagraph_data, max_scores=10):
    """
    Calculate stake-weighted scores using cached metagraph data to avoid issues with resyncing.
    This is identical to calculate_stake_weighted_scores but uses cached metagraph data.
    """
    # If no cached metagraph data available, return empty results
    if cached_metagraph_data is None:
        print("No cached metagraph data available, returning empty scores")
        return {}
    
    weighted_scores = {}
    
    for uid, models in recent_model_scores.items():
        weighted_scores[uid] = {}
        
        for model_key, scores in models.items():
            if not scores or scores[0]['score'] is None:
                weighted_scores[uid][model_key] = {
                    'score': None,
                    'hotkey': scores[0]['hotkey'],
                    'num_scores': 0,
                    'score_details': []
                }
                continue
            
            # Get the model_hash of the most recent score
            latest_model_hash = scores[0]['model_hash']

            # Process scores only for the latest model version
            processed_scores = []
            for score in scores[:max_scores]:  # Still limit to max_scores
                try:
                    scorer_hotkey = score['scorer_hotkey']
                    # Use cached metagraph data instead of live metagraph
                    if scorer_hotkey not in cached_metagraph_data['stakes']:
                        logging.warning(f"Scorer hotkey {scorer_hotkey} not found in cached metagraph")
                        continue
                    
                    stake = cached_metagraph_data['stakes'][scorer_hotkey]
                    
                    processed_scores.append({
                        'score': score['score'],
                        'stake': stake,
                        'timestamp': score['scored_at'],
                        'scorer_hotkey': scorer_hotkey,
                        'model_hash': score['model_hash']
                    })
                except (KeyError, ValueError) as e:
                    logging.warning(f"Error processing score for hotkey {scorer_hotkey}: {e}")
                    continue
            
            if not processed_scores:
                weighted_scores[uid][model_key] = {
                    'score': None,
                    'hotkey': scores[0]['hotkey'],
                    'num_scores': 0,
                    'score_details': []
                }
                continue

            # Analyze the zero score pattern
            total_scores = len(processed_scores)
            
            # Case 1: All scores are zero
            if all(s['score'] == 0 for s in processed_scores):
                logging.warning(f"All scores are zero for model {scores[0]['hotkey']}. Model may be non-functional.")
                final_scores = processed_scores
                
            # Case 2: Latest scores (last 3) are all zero
            elif len(processed_scores) >= 3 and all(s['score'] == 0 for s in processed_scores[:3]):
                logging.warning(f"Latest 3 scores are zero for model {scores[0]['hotkey']}. Recent model issue likely.")
                final_scores = processed_scores
                
            # Case 3: Remove zeros from calculation
            else:
                final_scores = [s for s in processed_scores if s['score'] > 0]
                if len(final_scores) == 0:
                    final_scores = processed_scores  # Fallback if all scores would be removed
            
            num_scores = len(final_scores)
            final_scores = filter_scores(final_scores)

            # Calculate weighted average
            total_stake = sum(score['stake'] for score in final_scores)
            weighted_sum = sum(score['score'] * score['stake'] for score in final_scores)
            
            # print(f"total_stake: {total_stake}")
            # print(f"weighted_sum: {weighted_sum}")

            if total_stake > 0:
                avg_score = weighted_sum / total_stake
            else:
                print(f"All scores are zero for model {scores[0]['hotkey']}. Model may be non-functional.")
                avg_score = None
                
            unique_validators = len(set(score['scorer_hotkey'] for score in final_scores))

            weighted_scores[uid][model_key] = {
                'score': avg_score,
                'scored_at': scores[0]['scored_at'],
                'hotkey': scores[0]['hotkey'],
                'competition_id': scores[0]['competition_id'],
                'block': scores[0]['block'],
                'model_metadata': scores[0]['model_metadata'],
                'model_hash': latest_model_hash,
                'num_scores': num_scores,
                'unique_validators': unique_validators,
                'score_pattern': {
                    'total_scores': total_scores,
                    'zero_scores': len([s for s in processed_scores if s['score'] == 0]),
                    'non_zero_scores': len([s for s in processed_scores if s['score'] > 0]),
                    'latest_three_zeros': all(s['score'] == 0 for s in processed_scores[:3]) if len(processed_scores) >= 3 else False
                },
                'score_details': [{
                    'hotkey': score['scorer_hotkey'],
                    'stake': score['stake'],
                    'score': score['score'],
                    'timestamp': score['timestamp'],
                    'weight': score['stake'] / total_stake if total_stake > 0 else 0,
                    'model_hash': score['model_hash']
                } for score in final_scores]
            }
    
    return weighted_scores


def calculate_stake_weighted_scores(recent_model_scores, metagraph, max_scores=10):
    """
    Calculate stake-weighted scores handling zero scores:
    1. If all scores are 0 - keep them (model is likely broken)
    2. If latest scores are 0 - keep them (model likely recently broke)
    3. Otherwise - remove zeros from calculation
    """
    weighted_scores = {}
    
    for uid, models in recent_model_scores.items():
        weighted_scores[uid] = {}
        
        for model_key, scores in models.items():
            if not scores or scores[0]['score'] is None:
                weighted_scores[uid][model_key] = {
                    'score': None,
                    'hotkey': scores[0]['hotkey'],
                    'num_scores': 0,
                    'score_details': []
                }
                continue
            
            # Get the model_hash of the most recent score
            latest_model_hash = scores[0]['model_hash']
            # if not latest_model_hash:
            #     logging.warning(f"No model hash for latest score of model {scores[0]['hotkey']} (UID {uid})")
            #     continue

            # Process scores only for the latest model version
            processed_scores = []
            for score in scores[:max_scores]:  # Still limit to max_scores
                # if score['model_hash'] != latest_model_hash:
                #     # Skip scores from different model versions
                #     logging.debug(
                #         f"Skipping score with different model hash for {scores[0]['hotkey']} "
                #         f"(Latest: {latest_model_hash}, Found: {score['model_hash']})"
                #     )
                #     continue
                    
                try:
                    scorer_hotkey = score['scorer_hotkey']
                    vali_uid = metagraph.hotkeys.index(scorer_hotkey)
                    stake = float(metagraph.S[vali_uid].item())
                    
                    processed_scores.append({
                        'score': score['score'],
                        'stake': stake,
                        'timestamp': score['scored_at'],
                        'scorer_hotkey': scorer_hotkey,
                        'model_hash': score['model_hash']
                    })
                except ValueError:
                    logging.warning(f"Scorer hotkey {scorer_hotkey} not found in metagraph")
                    continue
            
            if not processed_scores:
                weighted_scores[uid][model_key] = {
                    'score': None,
                    'hotkey': scores[0]['hotkey'],
                    'num_scores': 0,
                    'score_details': []
                }
                continue

            # Analyze the zero score pattern
            total_scores = len(processed_scores)
            
            # Case 1: All scores are zero
            if all(s['score'] == 0 for s in processed_scores):
                logging.warning(f"All scores are zero for model {scores[0]['hotkey']}. Model may be non-functional.")
                final_scores = processed_scores
                
            # Case 2: Latest scores (last 3) are all zero
            elif len(processed_scores) >= 3 and all(s['score'] == 0 for s in processed_scores[:3]):
                logging.warning(f"Latest 3 scores are zero for model {scores[0]['hotkey']}. Recent model issue likely.")
                final_scores = processed_scores
                
            # Case 3: Remove zeros from calculation
            else:
                final_scores = [s for s in processed_scores if s['score'] > 0]
                if len(final_scores) == 0:
                    final_scores = processed_scores  # Fallback if all scores would be removed
            
            num_scores = len(final_scores)
            final_scores = filter_scores(final_scores)

            # Calculate weighted average
            total_stake = sum(score['stake'] for score in final_scores)
            weighted_sum = sum(score['score'] * score['stake'] for score in final_scores)
            
            if total_stake > 0:
                avg_score = weighted_sum / total_stake
            else:
                avg_score = None
                
            unique_validators = len(set(score['scorer_hotkey'] for score in final_scores))

            weighted_scores[uid][model_key] = {
                'score': avg_score,
                'scored_at': scores[0]['scored_at'],
                'hotkey': scores[0]['hotkey'],
                'competition_id': scores[0]['competition_id'],
                'block': scores[0]['block'],
                'model_metadata': scores[0]['model_metadata'],
                'model_hash': latest_model_hash,
                'num_scores': num_scores,
                'unique_validators': unique_validators,
                'score_pattern': {
                    'total_scores': total_scores,
                    'zero_scores': len([s for s in processed_scores if s['score'] == 0]),
                    'non_zero_scores': len([s for s in processed_scores if s['score'] > 0]),
                    'latest_three_zeros': all(s['score'] == 0 for s in processed_scores[:3]) if len(processed_scores) >= 3 else False
                },
                'score_details': [{
                    'hotkey': score['scorer_hotkey'],
                    'stake': score['stake'],
                    'score': score['score'],
                    'timestamp': score['timestamp'],
                    'weight': score['stake'] / total_stake if total_stake > 0 else 0,
                    'model_hash': score['model_hash']
                } for score in final_scores]
            }
    
    return weighted_scores

class ModelScoreResponse(BaseModel):
    miner_hotkey: str
    miner_uid: int
    model_metadata: dict
    model_hash: Optional[str] = None
    model_score: float
    metric_scores: Optional[dict] = None


async def main():
    app = FastAPI()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    port = 8000 if IS_PROD else 8003
    
    # Initialize database at application startup
    init_database()
    init_eval_database()
    init_rep_database()
    queue_manager = ModelQueueManager()
    eval_manager = EvalLeaderboardManager()
    reputation_store = ReputationStore()

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)

                # Save metagraph data to file after successful sync
                metagraph_data = {
                    'hotkeys': list(metagraph.hotkeys),
                    'stakes': {hotkey: float(metagraph.S[i].item()) 
                              for i, hotkey in enumerate(metagraph.hotkeys)},
                    'timestamp': time.time(),
                    'last_updated': datetime.utcnow().isoformat()
                }
                
                # Ensure temp directory exists
                os.makedirs(os.path.dirname(METAGRAPH_CACHE_FILE), exist_ok=True)
                
                with _metagraph_file_lock:
                    with open(METAGRAPH_CACHE_FILE, 'w') as f:
                        json.dump(metagraph_data, f, indent=2)
                    print(f"Saved metagraph cache to {METAGRAPH_CACHE_FILE} with {len(metagraph_data['hotkeys'])} hotkeys")

                # Create pairs of (hotkey, uid) from metagraph
                registered_pairs = [
                    (hotkey, str(metagraph.hotkeys.index(hotkey))) 
                    for hotkey in metagraph.hotkeys
                ]
                queue_manager.archive_scores_for_deregistered_models(registered_pairs)

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during metagraph sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(90)

    async def check_stale_scoring_tasks():
        while True:
            # check and reset any stale scoring tasks
            reset_count = queue_manager.reset_stale_scoring_tasks(max_scoring_time_minutes=MODEL_EVAL_TIMEOUT/60)
            print(f"Reset {reset_count} stale scoring tasks")

            # Check every 1 minute to see if we have any newly flagged stale scoring tasks
            await asyncio.sleep(60)

    @app.get("/")
    def healthcheck():
        return {"status": "ok", "message": datetime.utcnow()}
    
    @app.get("/sentry-debug")
    async def trigger_error():
        division_by_zero = 1 / 0
    
    class GetModelToScoreRequest(BaseModel):
        competition_id: str = "o1"

    @app.post("/get-model-to-score")
    async def get_model_to_score(
        request: GetModelToScoreRequest = Body(default=GetModelToScoreRequest()),
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey)

        try:
            next_model = queue_manager.get_next_model_to_score(request.competition_id)
            if next_model:
                success = queue_manager.mark_model_as_being_scored(next_model['hotkey'], next_model['uid'], hotkey)
                print(f"Next model to score: {next_model['hotkey'], next_model['uid']} for validator uid {uid}, hotkey {hotkey}")
                if success:
                    return {
                        "success": True,
                        "miner_uid": next_model['uid']
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to mark model as being scored"
                    }
            
            else:
                return {
                    "success": False,
                    "message": "No model available to score. This should be a rare occurrence."
                }
                
        except Exception as e:
            logging.error(f"Error getting model to score: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    
    @app.post("/score-model")
    async def post_model_score(
        model_score_results: ModelScoreResponse,
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey)

        """
        {
            "miner_hotkey": miner_hotkey,
            "miner_uid": miner_uid,
            "model_metadata": model_metadata,
            "model_hash": model_hash,
            "model_score": model_score,
        }
        """

        try:
            print(f"Submitting score for model: {model_score_results.miner_hotkey, model_score_results.miner_uid, model_score_results.model_score}")
            success = queue_manager.submit_score(
                model_score_results.miner_hotkey,
                model_score_results.miner_uid,
                hotkey,
                model_score_results.model_hash,
                model_score_results.model_score,
                model_score_results.metric_scores
                #model_score_results.model_metadata,
            )
            if success:
                return {
                    "success": True,
                    "message": "Model score results successfully updated from validator "
                    + str(uid)
                }
            else:
                # Error in submitting score, perhaps model is not being scored or by a different validator
                return {
                    "success": False,
                    "message": "Failed to update model score results from validator "
                    + str(uid)
                }
        except Exception as e:
            logging.error(f"Error posting post_model_score: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    

    @app.get("/get-eval-metrics")
    async def get_eval_metrics():
        try:
            data = eval_manager.get_metrics_timeseries()
            return {
                "success": True,
                "data": data
            }
        except Exception as e:
            logging.error(f"Error getting leaderboard data: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        

    def filter_scores_by_today(recent_model_scores):
        """
        Filter recent_model_scores to only include scores from today.
        Returns a new dictionary with the same structure but only today's scores.
        """
        # Get current date in UTC
        today = datetime.now(timezone.utc).date()
        
        filtered_scores = {}
        
        for uid, models in recent_model_scores.items():
            filtered_models = {}
            
            for model_key, scores in models.items():
                filtered_score_list = []
                
                for score in scores:
                    # Parse the scored_at timestamp
                    try:
                        scored_at = score.get('scored_at')
                        
                        # Skip if scored_at is None or empty
                        if not scored_at:
                            print(f"Skipping score with missing scored_at: {score}")
                            continue
                        
                        # Handle different types of scored_at values
                        if isinstance(scored_at, str):
                            # Handle the timestamp format '2025-05-25T15:23:22'
                            scored_date = datetime.fromisoformat(scored_at).date()
                        elif isinstance(scored_at, datetime):
                            # Already a datetime object
                            scored_date = scored_at.date()
                        else:
                            # Try to convert to string first
                            scored_date = datetime.fromisoformat(str(scored_at)).date()
                        
                        # Only include scores from today
                        if scored_date == today:
                            filtered_score_list.append(score)
                            
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Error parsing scored_at timestamp: {score.get('scored_at', 'N/A')} (type: {type(score.get('scored_at'))}) - {e}")
                        continue
                
                # Only include the model if it has scores from today
                if filtered_score_list:
                    filtered_models[model_key] = filtered_score_list
            
            # Only include the UID if it has models with today's scores
            if filtered_models:
                filtered_scores[uid] = filtered_models
        
        return filtered_scores

    def filter_scores_except_today(recent_model_scores):
        """
        Filter recent_model_scores to exclude scores from today.
        Returns a new dictionary with the same structure but without today's scores.
        """
        # Get current date in UTC
        today = datetime.now(timezone.utc).date()
        
        filtered_scores = {}
        
        for uid, models in recent_model_scores.items():
            filtered_models = {}
            
            for model_key, scores in models.items():
                filtered_score_list = []
                
                for score in scores:
                    # Parse the scored_at timestamp
                    try:
                        scored_at = score.get('scored_at')
                        
                        # Skip if scored_at is None or empty
                        if not scored_at:
                            print(f"Skipping score with missing scored_at: {score}")
                            continue
                        
                        # Handle different types of scored_at values
                        if isinstance(scored_at, str):
                            # Handle the timestamp format '2025-05-25T15:23:22'
                            scored_date = datetime.fromisoformat(scored_at).date()
                        elif isinstance(scored_at, datetime):
                            # Already a datetime object
                            scored_date = scored_at.date()
                        else:
                            # Try to convert to string first
                            scored_date = datetime.fromisoformat(str(scored_at)).date()
                        
                        # Only include scores that are NOT from today (historical scores)
                        if scored_date != today:
                            filtered_score_list.append(score)
                            
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Error parsing scored_at timestamp: {score.get('scored_at', 'N/A')} (type: {type(score.get('scored_at'))}) - {e}")
                        continue
                
                # Only include the model if it has historical scores
                if filtered_score_list:
                    filtered_models[model_key] = filtered_score_list
            
            # Only include the UID if it has models with historical scores
            if filtered_models:
                filtered_scores[uid] = filtered_models
        
        return filtered_scores

    @app.get("/get-top-model")
    async def get_top_model():
        # if not authenticate_with_bittensor(hotkey, metagraph):
        #     print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
        #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Valid hotkey required.")
        # # get uid of bittensor validator
        # uid = metagraph.hotkeys.index(hotkey)

        try:
            # Get cached metagraph data (file-only, no network calls)
            cached_metagraph_data = get_cached_metagraph_file_only()
            if cached_metagraph_data is None:
                raise HTTPException(
                    status_code=503, 
                    detail="Metagraph cache not available. Please try again in a few moments."
                )
            
            recent_model_scores = queue_manager.get_recent_model_scores(scores_per_model=MIN_NON_ZERO_SCORES)
            
            # Filter out today's scores - we want historical data only
            historical_scores = filter_scores_except_today(recent_model_scores)
            
            # Calculate stake-weighted averages for each model using historical scores
            weighted_scores = calculate_stake_weighted_scores_cached(historical_scores, cached_metagraph_data)
            
            # Group models by competition_id and find top 2 for each
            competition_top_models = {}
            
            # Collect all valid models with scores
            all_models = []
            for uid, models in weighted_scores.items():
                for model_key, data in models.items():
                    if (data['score'] is not None and 
                        float(data['score']) > 0 and 
                        data['num_scores'] >= MIN_NON_ZERO_SCORES and
                        data.get('competition_id')):
                        
                        try:
                            model_metadata = json.loads(data['model_metadata'])["id"]
                            model_name = f"{model_metadata['namespace']}/{model_metadata['name']}"
                            block_is_earlier = cached_compare_block_and_model_file_only(data['block'], model_name)
                            
                            # Only include if block is earlier (valid model)
                            if block_is_earlier:
                                all_models.append({
                                    'uid': uid,
                                    'hotkey': data['hotkey'],
                                    'competition_id': data['competition_id'],
                                    'model_name': model_name,
                                    'score': data['score'],
                                    'scored_at': data['scored_at'],
                                    'block': data['block'],
                                    'model_hash': data['model_hash'],
                                    'num_scores': data['num_scores'],
                                    'unique_validators': data['unique_validators'],
                                    'score_pattern': data['score_pattern'],
                                    'score_details': data['score_details']
                                })
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing model metadata for UID {uid}: {e}")
                            continue
            
            # Group by competition_id and get top 2 for each
            for model in all_models:
                comp_id = model['competition_id']
                if comp_id not in competition_top_models:
                    competition_top_models[comp_id] = []
                competition_top_models[comp_id].append(model)
            
            # Sort each competition's models by score (descending) and take top 2
            for comp_id in competition_top_models:
                competition_top_models[comp_id] = sorted(
                    competition_top_models[comp_id], 
                    key=lambda x: x['score'], 
                    reverse=True
                )[:1]  # Take top 2
            
            print("Top 2 models per competition (excluding today's scores):")
            trimmed_top_models = { "o1": [], "v1": [] }
            for comp_id, models in competition_top_models.items():
                print(f"\nCompetition {comp_id}:")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. UID {model['uid']}: {model['model_name']} - Score: {model['score']:.4f}")
                    trimmed_top_models[comp_id].append({
                        "uid": model['uid'],
                        "score": model['score']
                    })
            return trimmed_top_models
            
        except HTTPException:
            # Re-raise HTTP exceptions (like 503 for cache not available)
            raise
        except Exception as e:
            logging.error(f"Error getting top model: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.post("/get-all-model-scores")
    async def get_all_model_scores(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey) 

        try:
            all_model_scores = dict()
            
            recent_model_scores = queue_manager.get_recent_model_scores(scores_per_model=MIN_NON_ZERO_SCORES)
            
            # Calculate stake-weighted averages for each model
            
            weighted_scores = calculate_stake_weighted_scores(recent_model_scores, metagraph)
   
            # Example of accessing results
            for uid, models in weighted_scores.items():
                for model_key, data in models.items():
                    if data['score'] is not None and float(data['score']) > 0 and data['num_scores'] >= MIN_NON_ZERO_SCORES:
                        print(f"\nUID: {uid}, Hotkey: {data['hotkey']}")
                        print(f"Weighted Average Score: {data['score']:.4f}")
                        print(f"Most Recent Score: {data['score_details'][0]['score']:.4f}")
                        print(f"Total Scores Used: {data['num_scores']}")
                        print(f"Unique Validators: {data['unique_validators']}")
                        print(f"Score Pattern: {data['score_pattern']}")
                    
                        model_metadata = json.loads(data['model_metadata'])["id"]
                        model_name = f"{model_metadata['namespace']}/{model_metadata['name']}"
                        block_is_earlier = cached_compare_block_and_model_file_only(data['block'], model_name)

                        
                        all_model_scores[uid] = [{
                            'hotkey': data['hotkey'],
                            'competition_id': data['competition_id'],
                            'model_name': f"{model_metadata['namespace']}/{model_metadata['name']}",
                            'score': data['score'] if block_is_earlier else penalty_score,
                            'scored_at': data['scored_at'],
                            'block': data['block'],
                            'model_hash': data['model_hash'],
                            'score_details': data['score_details']
                        }]
                    else:
                        all_model_scores[uid] = [{
                            'hotkey': data['hotkey'],
                            'competition_id': None,
                            'model_name': None,
                            'score': None,
                            'scored_at': None,
                            'block': None,
                            'model_hash': None,
                            'score_details': None
                        }]
                    
                    # Print details of each score used in the average
                    """
                    for score in data['score_details']:
                        print(f"  Scorer: {score['hotkey']}")
                        print(f"    Stake: {score['stake']:.2f}")
                        print(f"    Score: {score['score']:.4f}")
                        print(f"    Weight in Average: {score['weight']:.4f}")
                        print(f"    Time: {score['timestamp']}")
                    """

            
            if all_model_scores:
                return {
                    "success": True,
                    "model_scores": all_model_scores
                }
            elif not all_model_scores or len(all_model_scores) == 0:
                return {
                    "success": False,
                    "message": "No model scores available. This should be a rare occurrence."
                }
        except Exception as e:
            logging.error(f"Error getting all model scores: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    

    # Clear cache periodically to prevent memory growth
    async def clear_block_model_cache():
        while True:
            await asyncio.sleep(3600*12)  # Clear cache every 12 hours
            with _cache_lock:
                print(f"Clearing block-model comparison cache. Stats before clear: {_cache_stats}")
                _block_model_cache.clear()

    @app.get("/baseline-score/{competition_id}", summary="Get the latest baseline score for a competition.")
    async def get_baseline(
        competition_id: str, 
        hotkey: Annotated[str, Depends(get_hotkey)] = None
    ):
        """
        Returns the most recent baseline score for the specified competition_id.
        """
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey)
        print(f"UID:{uid} hit the get_baseline endpoint")

        try:
            latest_baseline = reputation_store.get_latest_baseline_score(competition_id)
        except Exception as e:
            logging.error(f"Error getting baseline score: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")
        
        if not latest_baseline:
            raise HTTPException(status_code=404, detail=f"No baseline score found for competition_id: {competition_id}")
        return latest_baseline

    @app.get("/reputations", summary="Get reputations for all miners.")
    async def get_all_reputations(
        hotkey: Annotated[str, Depends(get_hotkey)] = None
    ):
        """
        Returns a mapping of hotkey to its current reputation for all miners.
        """
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey) 
        print(f"UID:{uid} hit the get_reputations endpoint")
        try:
            reputation_map = reputation_store.get_all_reputations()
            return reputation_map
        except Exception as e:
            logging.error(f"Error getting all miner reputations: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    await asyncio.gather(
        resync_metagraph(),
        asyncio.to_thread(
            uvicorn.run,
            app,
            host="0.0.0.0",
            port=port
        ),
        check_stale_scoring_tasks(),
        clear_block_model_cache(),
    )


def cached_compare_block_and_model_file_only(block: int, model_name: str) -> bool:
    """
    File-only cached version of compare_block_and_model function.
    Returns True if block is earlier than model's creation date.
    Does NOT make network calls - only uses existing cache.
    Returns True (allow model) if not in cache to avoid blocking valid models.
    """
    global _cache_stats
    cache_key = f"{block}:{model_name}"
    
    # Update total requests counter
    with _cache_lock:
        _cache_stats["total_requests"] += 1
    
    # Check if key exists in cache
    if cache_key in _block_model_cache:
        # Move to end (most recently used)
        with _cache_lock:
            value = _block_model_cache.pop(cache_key)
            _block_model_cache[cache_key] = value
            _cache_stats["hits"] += 1
            
            # Log cache stats occasionally
            if _cache_stats["total_requests"] % 100 == 0:
                hit_rate = (_cache_stats["hits"] / _cache_stats["total_requests"]) * 100
                print(f"Cache stats: {_cache_stats}, Hit rate: {hit_rate:.2f}%, Size: {len(_block_model_cache)}")
        return value
    
    # If not in cache, return True (allow model) to avoid network calls
    # This prevents blocking valid models when cache doesn't have the data
    with _cache_lock:
        _cache_stats["misses"] += 1
    print(f"Block comparison not in cache for {model_name}, allowing model to avoid network calls")
    return True


def cached_compare_block_and_model(block: int, model_name: str) -> bool:
    """
    Thread-safe dictionary-cached version of compare_block_and_model function.
    Returns True if block is earlier than model's creation date.
    Uses LRU caching with size limit to prevent memory issues.
    """
    global _cache_stats
    cache_key = f"{block}:{model_name}"
    
    # Update total requests counter
    with _cache_lock:
        _cache_stats["total_requests"] += 1
    
    # First check if key exists in cache without acquiring lock
    if cache_key in _block_model_cache:
        # Move to end (most recently used)
        with _cache_lock:
            value = _block_model_cache.pop(cache_key)
            _block_model_cache[cache_key] = value
            _cache_stats["hits"] += 1
            
            # Log cache stats occasionally
            if _cache_stats["total_requests"] % 100 == 0:
                hit_rate = (_cache_stats["hits"] / _cache_stats["total_requests"]) * 100
                print(f"Cache stats: {_cache_stats}, Hit rate: {hit_rate:.2f}%, Size: {len(_block_model_cache)}")
        return value
    
    # If not in cache, acquire lock and check again (double-checked locking pattern)
    with _cache_lock:
        if cache_key in _block_model_cache:
            # Another thread might have added it while we were waiting for the lock
            value = _block_model_cache.pop(cache_key)
            _block_model_cache[cache_key] = value
            _cache_stats["hits"] += 1
            return value
            
        # Expensive operation done under lock
        _cache_stats["misses"] += 1
        result = compare_block_and_model(block, model_name)
        
        # If cache is full, remove oldest item (first in OrderedDict)
        if len(_block_model_cache) >= MAX_CACHE_SIZE:
            _block_model_cache.popitem(last=False)
            
        _block_model_cache[cache_key] = result
        return result


if __name__ == "__main__":
    asyncio.run(main())
