from utilities.logging_setup import warmup_logging; warmup_logging()
from contextlib import asynccontextmanager
from typing import Annotated, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair
import uvicorn
import asyncio
from datetime import datetime, timedelta
from neurons.scoring_manager import (
    ScoringManager, ModelScoreTaskData, ScoreModelInputs, ModelScoreStatus, get_scoring_config
)
import bittensor as bt
from dotenv import load_dotenv; load_dotenv("vali.env")

async def main():
    config = get_scoring_config()
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
                "message": f"Model {current_task.inputs.hf_repo_id} is already being scored"
            }

        bt.logging.info(f"Starting scoring for {request}")
        background_tasks.add_task(scoring_manager.start_scoring, request)

        return {
            "success": True,
            "message": f"Added {request.hf_repo_id} to scoring queue"
        }

    @app.get("/api/check_scoring_status")
    async def check_scoring_status(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ) -> Optional[ModelScoreTaskData]:
        current_task = scoring_manager.get_current_task()
        bt.logging.debug(f"Current task: {current_task}")
        return current_task

    async def check_for_updates(server):
        if config.auto_update:
            while True:
                if scoring_manager.should_restart():
                    bt.logging.info(f'Validator is out of date, quitting to restart.')
                    await scoring_manager.cleanup()
                    server.should_exit = True
                    break
                await asyncio.sleep(scoring_manager.update_check_interval + 10)

    async def restart_wandb_session(server):
        if not config.wandb.off:
            while True:
                if server.should_exit:
                    bt.logging.info(f'Terminating restart_wandb_session thread')
                    break

                # Check if we need to restart wandb session
                scoring_manager.check_wandb_run()

                # only sleep for 10 seconds, in order to respect the server shutdown
                await asyncio.sleep(10)

    # Handle shutdown gracefully
    try:
        server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.port))
        await asyncio.gather(
            server.serve(),
            check_for_updates(server),
            restart_wandb_session(server),
        )
    except KeyboardInterrupt:
        await scoring_manager.cleanup()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
