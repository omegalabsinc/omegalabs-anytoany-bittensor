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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vali_hotkey", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    scoring_manager = ScoringManager()
    app = FastAPI()
    security = HTTPBasic()

    def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
        """Authenticate requests using validator hotkey"""
        keypair = Keypair(ss58_address=credentials.username)

        if not keypair.verify(credentials.username, credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature mismatch",
            )

        # Only allow configured validator hotkey
        if credentials.username != args.vali_hotkey:
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

    await asyncio.gather(
        asyncio.to_thread(
            uvicorn.run,
            app,
            host="0.0.0.0", 
            port=args.port
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
