from typing import Annotated, List, Optional
from traceback import print_exception
from datetime import datetime

import bittensor
import uvicorn
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair
from pydantic import BaseModel

from model.storage.mysql_model_queue import init_database, ModelQueueManager
from vali_api.config import NETWORK, NETUID, IS_PROD


app = FastAPI()
security = HTTPBasic()


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

    if metagraph.S[uid] < 1000 and NETWORK != "test":
        print("Bittensor validator requires 1000+ staked TAO")
        return False

    return True

class ModelScoreResponse(BaseModel):
    miner_hotkey: str
    miner_uid: int
    model_metadata: dict
    model_hash: str
    model_score: float


async def main():
    app = FastAPI()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)
    
    # Initialize database at application startup
    init_database()
    queue_manager = ModelQueueManager()

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)

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
            reset_count = queue_manager.reset_stale_scoring_tasks()
            print(f"Reset {reset_count} stale scoring tasks")

            # Check every 1 minute to see if we have any newly flagged stale scoring tasks
            await asyncio.sleep(60)

    @app.get("/")
    def healthcheck():
        return {"status": "ok", "message": datetime.utcnow()}
    
    @app.post("/get-model-to-score")
    async def get_model_to_score(
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
            next_model = queue_manager.get_next_model_to_score()
            if next_model:
                success = queue_manager.mark_model_as_being_scored(next_model['hotkey'], next_model['uid'], hotkey)
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
            # In your API endpoint for submitting scores:
            print(f"Submitting score for model: {model_score_results.miner_hotkey, model_score_results.miner_uid, model_score_results.model_score}")
            success = queue_manager.submit_score(
                model_score_results.miner_hotkey,
                model_score_results.miner_uid,
                hotkey,
                model_score_results.model_hash,
                model_score_results.model_score
                #model_score_results.model_metadata
            )
            if success:
                # Score submitted successfully
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
            all_model_scores = queue_manager.get_all_model_scores()
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
    

    await asyncio.gather(
        resync_metagraph(),
        asyncio.to_thread(
            uvicorn.run,
            app,
            host="0.0.0.0",
            #port=443,
            #ssl_certfile="/root/origin-cert.pem",
            #ssl_keyfile="/root/origin-key.key",
        ),
        check_stale_scoring_tasks(),
    )


if __name__ == "__main__":
    asyncio.run(main())
