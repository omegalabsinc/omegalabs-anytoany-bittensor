import argparse
import bittensor as bt
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Import database configuration and initialization functions
# Assuming the script is run from the root of the project.
from vali_api.config import DBHOST, DBNAME, DBUSER, DBPASS, IS_PROD, NETWORK, NETUID
from vali_api.app import calculate_stake_weighted_scores, cached_compare_block_and_model_file_only
from model.storage.mysql_model_queue import init_database, get_session, ModelQueueManager
from model.storage.reputation_store import init_database as init_rep_database, ReputationStore
from model.storage.reputation_store import BaselineScore, MinerReputation, ReputationHistory
from constants import MIN_NON_ZERO_SCORES, penalty_score

ALPHA = 0.005

# --- Core Reputation Logic ---

def get_latest_baseline_score(session: Session, competition_id: str) -> float:
    """Fetches the most recent baseline score for a given competition."""
    latest_baseline = (
        session.query(BaselineScore)
        .filter(BaselineScore.competition_id == competition_id)
        .order_by(desc(BaselineScore.created_at))
        .first()
    )
    if not latest_baseline:
        logging.warning(f"No baseline score found for competition_id: {competition_id}. Defaulting to 0.")
        return 0.0
    return latest_baseline.score

def update_reputations(session: Session, metagraph: bt.metagraph, all_model_scores:dict,  competition_id: str):
    """
    Updates miner reputations based on their performance against the baseline.
    """
    logging.info("Starting reputation update process...")

    # 1. Get active miners from the metagraph #TODO: Check if this is the same logic used when pushing models to queue.
    active_hotkeys = {neuron.hotkey for neuron in metagraph.neurons}
    logging.info(f"Found {len(active_hotkeys)} active miners in the metagraph.")

    # 2. Get existing reputations from the DB
    existing_reputations = {rep.hotkey: rep for rep in session.query(MinerReputation).all()}
    logging.info(f"Found {len(existing_reputations)} miners with reputation in the database.")

    # 3. Handle new and deregistered miners
    new_miners = active_hotkeys - set(existing_reputations.keys())
    for hotkey in new_miners:
        new_rep = MinerReputation(hotkey=hotkey, reputation=0.5)
        session.add(new_rep)
        logging.info(f"New miner detected: {hotkey}. Initializing reputation to 0.5.")
    
    deregistered_miners = set(existing_reputations.keys()) - active_hotkeys
    for hotkey in deregistered_miners:
        #TODO: Only delete after a certain period of time. last_updated is more than 30 days old. delete it.
        session.query(MinerReputation).filter_by(hotkey=hotkey).delete()
        session.query(ReputationHistory).filter_by(hotkey=hotkey).delete()
        logging.info(f"Deregistered miner detected: {hotkey}. Deleting history records.")


    session.commit()

    # 4. Update reputations for active miners
    baseline_score = get_latest_baseline_score(session, competition_id)
    logging.info(f"Using baseline score of {baseline_score} for competition '{competition_id}'.")

    updated_reputations = session.query(MinerReputation).filter(MinerReputation.hotkey.in_(active_hotkeys)).all()
    print(f"{len(updated_reputations)=}")
    # Get scores for each UID
    uids = [int(uid) for uid in metagraph.uids.tolist()]
    scores_per_uid = {uid: 0.0 for uid in uids}
    hotkey_score_mapping = {}
    for uid in uids:
        models_data = all_model_scores.get(str(uid), [])
        for model_data in models_data:
            if model_data.get("competition_id") == "v2":
                score = model_data.get("score", 0.0)
                if score is not None:
                    scores_per_uid[uid] = score
                    hotkey_score_mapping[model_data["hotkey"]] = score
                break
    
    for rep in updated_reputations:
        miner_score = hotkey_score_mapping.get(rep.hotkey)

        if miner_score is None:
            # Miner is active but has no score yet. Treat as not beating baseline.
            #TODO: Edge case: the miner is present. But the vali_api get_all_scores does not return it's score.
            # Either the model of this hotkey is not scored.
            # Or not following some conditions in vali api. 
            # but anyways same thing is going to happen during set weights in validator.
            performance = 0
            continue
        else:
            performance = 1 if miner_score > baseline_score else 0
        
        old_rep = rep.reputation
        # EMA calculation
        new_rep = (1 - ALPHA) * old_rep + ALPHA * performance
        old_rep = new_rep
        rep.reputation = new_rep

        logging.debug(f"Updating reputation for {rep.hotkey}: old={old_rep:.4f}, new={new_rep:.4f}, performance={performance}")

        # Log to history #TODO: add a logic to not store every update -> but something like after every 5 cycles or perday updates.
        history_record = ReputationHistory(hotkey=rep.hotkey, timestamp=datetime.utcnow(), reputation=new_rep)
        session.add(history_record)

    session.commit()
    logging.info("Reputation update process completed.")

def get_all_model_scores(queue_manager, metagraph):
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
        raise


def main():
    parser = argparse.ArgumentParser(description="Update miner reputations based on performance.")
    parser.add_argument("--netuid", type=int, default=NETUID, help="The chain netuid.")
    parser.add_argument("--network", type=str, default=NETWORK, help="The chain network.")
    parser.add_argument("--competition_id", type=str, required=True, help="The competition ID to update reputations for.")
    
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    
    config = bt.config(parser)

    # Setup bittensor objects
    bt.logging(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    # Initialize database
    init_database()
    session = get_session()
    queue_manager = ModelQueueManager()

    model_scores = None
    response_json = get_all_model_scores(queue_manager, metagraph)

    if "success" in response_json and response_json["success"]:
        model_scores = response_json["model_scores"]
        bt.logging.info(f"Retrieved model scores from API")
    
    try:
        update_reputations(session, metagraph, model_scores, config.competition_id)
    finally:
        session.close()

if __name__ == "__main__":
    main() 