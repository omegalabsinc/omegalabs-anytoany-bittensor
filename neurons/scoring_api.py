from contextlib import asynccontextmanager
from typing import Annotated, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair
import uvicorn
import argparse
import asyncio
from datetime import datetime
from neurons.scoring_manager import ScoringManager, ModelScoreTaskData, ScoreModelInputs, ModelScoreStatus
import bittensor as bt
from constants import SUBNET_UID
from dotenv import load_dotenv; load_dotenv("vali.env")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vali_hotkey", type=str)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wandb.off", action="store_true")
    parser.add_argument("--auto_update", action="store_true")
    parser.add_argument("--netuid", type=int, default=SUBNET_UID, help="The subnet UID.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)
    assert config.vali_hotkey, "vali_hotkey is required"

    scoring_manager = ScoringManager(config)
    security = HTTPBasic()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await scoring_manager.cleanup()

    app = FastAPI(lifespan=lifespan)

    def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
        """Authenticate requests using validator hotkey"""
        keypair = Keypair(ss58_address=credentials.username)

        if not keypair.verify(credentials.username, credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature mismatch",
            )

        # Only allow configured validator hotkey
        if credentials.username != config.vali_hotkey:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized validator",
            )

        return credentials.username

    @app.get("/")
    async def root():
        return { "now": datetime.now().isoformat() }

    @app.post("/api/start_model_scoring")
    async def start_model_scoring(
        request: ScoreModelInputs,
        background_tasks: BackgroundTasks,
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        current_task = scoring_manager.get_current_task()
        if current_task and current_task.status == ModelScoreStatus.SCORING:
            return {
                "success": False,
                "message": f"Model {current_task.hf_repo_id} is already being scored"
            }

        background_tasks.add_task(scoring_manager.start_scoring, request)

        return {
            "success": True,
            "message": f"Added {request.hf_repo_id} to scoring queue"
        }

    @app.get("/api/check_scoring_status")
    async def check_scoring_status(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ) -> Optional[ModelScoreTaskData]:
        return scoring_manager.get_current_task()

    async def check_for_updates():
        if config.auto_update:
            while True:
                if scoring_manager.should_restart():
                    bt.logging.info(f'Validator is out of date, quitting to restart.')
                    raise KeyboardInterrupt
                await asyncio.sleep(scoring_manager.update_check_interval + 10)

    async def restart_wandb_session():
        if not config.wandb.off:
            while True:
                scoring_manager.check_wandb_run()
                await asyncio.sleep(scoring_manager.update_check_interval)

    # Handle shutdown gracefully
    try:
        await asyncio.gather(
            asyncio.to_thread(
                uvicorn.run,
                app,
                host="0.0.0.0",
                port=config.port,
            ),
            check_for_updates(),
            restart_wandb_session(),
        )
    except KeyboardInterrupt:
        await scoring_manager.cleanup()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
