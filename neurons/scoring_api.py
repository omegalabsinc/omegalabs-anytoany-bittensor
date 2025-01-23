from neurons.validator import Validator
from typing import Annotated, Optional
from fastapi import FastAPI, HTTPException, Depends, Path
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair
import bittensor as bt
import uvicorn
import argparse
import asyncio
from datetime import datetime

async def main():
    config = Validator.config()
    wallet = bt.wallet(config=config)
    VALI_HOTKEY = wallet.hotkey.ss58_address

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
        if credentials.username != VALI_HOTKEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized validator",
            )

        return credentials.username

    @app.get("/")
    async def root():
        return { "now": datetime.now().isoformat() }

    @app.post("/api/add_to_scoring_queue")
    async def add_to_scoring_queue(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        """Add model to scoring queue"""
        # TODO: Implement queue logic
        return {
            "success": True,
            "message": f"Added to scoring queue"
        }

    @app.get("/api/check_scoring_status/{model_id}")
    async def check_scoring_status(
        model_id: str = Path(...),
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        """Check scoring status for a model"""
        # TODO: Implement status check logic
        return {
            "success": True,
            "model_id": model_id,
            "status": "pending"
        }

    await asyncio.gather(
        asyncio.to_thread(
            uvicorn.run,
            app,
            host="0.0.0.0", 
            port=8000
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
