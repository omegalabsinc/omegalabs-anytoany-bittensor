import os
import argparse
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import traceback
import multiprocessing
import time
import docker

os.environ["USE_TORCH"] = "1"
os.environ["BT_LOGGING_INFO"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
GPU_MEM_GB_REQD = 3 # TODO: change to 39 for prod

import bittensor as bt
from pydantic import BaseModel, computed_field, Field
import wandb

from model.model_tracker import ModelTracker
from constants import MODEL_EVAL_TIMEOUT, NUM_CACHED_MODELS, SUBNET_UID
from neurons.docker_inference_v2v import run_v2v_scoring
from neurons.docker_model_scoring import run_o1_scoring
from neurons.docker_inference_voicebench import run_voicebench_scoring
from utilities.temp_dir_cache import TempDirCache
from utilities.gpu import get_gpu_memory
from utilities.git_utils import is_git_latest

class ModelScoreStatus(Enum):
    SCORING = "scoring"
    COMPLETED = "completed" 
    FAILED = "failed"
    TIMEOUT = "timeout"

class ScoreModelInputs(BaseModel):
    hf_repo_id: str
    competition_id: str
    hotkey: str
    block: int

class ModelScoreTaskData(BaseModel):
    inputs: ScoreModelInputs
    score: Optional[float] = None
    metric_scores: Optional[dict] = None
    status: ModelScoreStatus = ModelScoreStatus.SCORING
    started_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

    @computed_field
    @property
    def runtime_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vali_hotkey", type=str)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wandb.off", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Don't connect to metagraph")
    parser.add_argument("--auto_update", action="store_true")
    parser.add_argument("--netuid", type=int, default=SUBNET_UID, help="The subnet UID.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    return parser

def get_scoring_config():
    parser = get_argparse()
    return bt.config(parser)

def format_voicebench_output(scores: dict, status: dict, metadata: dict) -> dict:
    """
    Format VoiceBench output according to the required structure.
    
    Args:
        scores: Dictionary with dataset scores and 'overall' key
        status: Dictionary with evaluation status for each dataset
        metadata: Dictionary with evaluation metadata
        
    Returns:
        Formatted output dictionary
    """
    return {
        'raw_scores': {k: v for k, v in scores.items() if k != 'overall'},
        'combined_score': scores.get('overall', 0.0),
        'evaluation_status': status,
        'metadata': metadata
    }


class ScoringManager:
    def __init__(self, config):
        self.config = config
        self.current_task: Optional[ModelScoreTaskData] = None
        self.model_tracker = ModelTracker(thread_safe=False)
        self.current_process: Optional[multiprocessing.Process] = None

        self.temp_dir_cache = TempDirCache(NUM_CACHED_MODELS)

        try:
            # early exit if GPU memory insufficient
            total_gb, used_gb, avail_gb = get_gpu_memory()
            if avail_gb < GPU_MEM_GB_REQD:
                m = f"Insufficient GPU Memory available: {avail_gb:.2f} GB available, out of total {total_gb:.2f} GB"
                bt.logging.error(m)
                raise RuntimeError(m)
        except Exception as e:
            bt.logging.error(f"Failed to get GPU memory: {e}: {traceback.format_exc()}")
        
        # auto-update
        self.last_update_check = datetime.now()
        self.update_check_interval = 1800  # 30 minutes

        # get UID
        if self.config.offline:
            self.uid = 0
        else:
            subtensor = bt.subtensor(network=self.config.subtensor.network)
            metagraph: bt.metagraph = subtensor.metagraph(self.config.netuid)
            # self.uid = metagraph.hotkeys.index(self.config.vali_hotkey)
            self.uid = 96 # TODO: change to metagraph.hotkeys.index(hotkey) for prod

        # init wandb
        self.wandb_run_start = None
        if not self.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.new_wandb_run()
            else:
                bt.logging.exception("WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off, but it is strongly recommended to run with W&B enabled.")
        else:
            bt.logging.warning("Running with --wandb.off. It is strongly recommended to run with W&B enabled.")

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (datetime.now() - self.last_update_check).seconds < self.update_check_interval:
            return False

        self.last_update_check = datetime.now()

        return not is_git_latest()

    def check_wandb_run(self):
        # Check if we should start a new wandb run.
        if not self.config.wandb.off and self.wandb_run_start != None:
            if (datetime.now() - self.wandb_run_start) >= timedelta(
                days=1
            ):
                bt.logging.info(
                    "Current wandb run is more than 1 day old. Starting a new run."
                )
                self.wandb_run.finish()
                self.new_wandb_run()

    def new_wandb_run(self):
        # Shoutout SN13 for the wandb snippet!
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        now = datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = "scoring_api-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project="omega-sn21-validator-logs",
            entity="omega-labs",
            config={
                "uid": self.uid,
                "hotkey": self.config.vali_hotkey,
                "run_name": run_id,
                "type": "scoring_api",
            },
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

    def _score_model(self, inputs: ScoreModelInputs):
        """ Actual model scoring logic """
        start_time = time.time()
        
        # Route to appropriate scoring function based on competition
        if inputs.competition_id == "o1":
            fn_to_call = run_o1_scoring
        elif inputs.competition_id == "voicebench" or inputs.competition_id == "v2v":
            # Use VoiceBench evaluation for both voicebench and v2v competitions
            fn_to_call = run_voicebench_scoring
        else:
            # Default to VoiceBench for unknown competition types
            bt.logging.warning(f"Unknown competition_id: {inputs.competition_id}, defaulting to VoiceBench")
            fn_to_call = run_voicebench_scoring
        
        result_dict = fn_to_call(
            hf_repo_id=inputs.hf_repo_id,
            hotkey=inputs.hotkey,
            block=inputs.block,
            model_tracker=self.model_tracker,
            local_dir=self.temp_dir_cache.get_temp_dir(inputs.hf_repo_id),
        )
        
        total_time_seconds = time.time() - start_time
        
        result_dict['total_time_seconds'] = total_time_seconds
        
        print(f"RESULT dict: \n{result_dict}")
        # The result_dict should already contain only native Python types
        # based on the scoring flow, but let's validate to be safe
        serializable_result = result_dict
                
        bt.logging.info(f"Score for {inputs} is completed.")
        
        return serializable_result


    def kill_docker_images(self):
        """
        Kill and clean up docker images that have 'none' as image name or 'miner_' in the name
        """
        try:
            client = docker.from_env()
            images = client.images.list()
            
            for image in images:
                # Check for None/<none> tags or miner_ prefix
                if not image.tags or '<none>' in str(image.tags).lower() or any('miner_' in tag.lower() for tag in image.tags):
                    bt.logging.info(f"Removing docker image: {image.id}")
                    try:
                        client.images.remove(image.id, force=True)
                    except Exception as e:
                        bt.logging.warning(f"Failed to remove image {image.id}: {str(e)}")
                        
        except Exception as e:
            bt.logging.error(f"Failed to kill docker images: {str(e)}")
            raise


    def _score_model_wrapped(self, inputs: ScoreModelInputs, result_queue: multiprocessing.Queue):
        """ Wraps the scoring process in a queue to get the result """
        try:
            self.kill_docker_images()
            bt.logging.info(f"Starting scoring for model: {inputs.hf_repo_id}")
            result_dict = self._score_model(inputs)
            bt.logging.info(f"Completed scoring for model: {inputs.hf_repo_id}")
            result_queue.put(('success', result_dict))
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            bt.logging.error(f"Failed to score model {inputs.hf_repo_id}: {error_msg}")
            result_queue.put(('error', error_msg))
        finally:
            bt.logging.info(f"Scoring process completed for model: {inputs.hf_repo_id}")

    def start_scoring(self, inputs: ScoreModelInputs):
        """ Starts the scoring process """
        if self.current_process and self.current_process.is_alive():
            self.current_process.terminate()
            self.current_process.join(timeout=5)

        self.current_task = ModelScoreTaskData(inputs=inputs)

        # Create a queue for getting the result
        result_queue = multiprocessing.Queue()

        # Create and start the scoring process
        ctx = multiprocessing.get_context('spawn')
        self.current_process = process = ctx.Process(
            target=self._score_model_wrapped,
            args=(inputs, result_queue)
        )
        process.start()

        # Wait for either completion or timeout
        process.join(timeout=MODEL_EVAL_TIMEOUT)

        if process.is_alive():
            # If process is still running after timeout, terminate it
            process.terminate()
            process.join()
            self.current_task.status = ModelScoreStatus.TIMEOUT
            self.current_task.score = None
            bt.logging.error(f"Model evaluation timed out after {MODEL_EVAL_TIMEOUT} seconds for {inputs.hf_repo_id}")
            return

        # Get result if process completed
        if not result_queue.empty():
            status, result_dict = result_queue.get()
            if status == 'error':
                self.current_task.status = ModelScoreStatus.FAILED
                self.current_task.score = None
                self.current_task.error = result_dict # won't be a dict in case of error.
                return

            self.current_task.status = ModelScoreStatus.COMPLETED if result_dict is not None else ModelScoreStatus.FAILED
            self.current_task.score = result_dict['combined_score']
            self.current_task.metric_scores = result_dict

        else:
            bt.logging.error(f"Process terminated without returning a result for {inputs.hf_repo_id}")
            self.current_task.status = ModelScoreStatus.FAILED
            self.current_task.score = None

    def get_current_task(self):
        return self.current_task

    async def cleanup(self):
        """Cleanup method to be called during shutdown"""
        if self.current_process and self.current_process.is_alive():
            bt.logging.info("Terminating running scoring process...")
            self.current_process.terminate()
            self.current_process.join(timeout=5)
            if self.current_process.is_alive():
                bt.logging.warning("Force killing scoring process...")
                self.current_process.kill()
                self.current_process.join(timeout=1)


if __name__ == "__main__":
    parser = get_argparse()
    parser.add_argument("--hf_repo_id", type=str)
    parser.add_argument("--competition_id", type=str)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--hotkey", type=str)
    config = bt.config(parser)
    scoring_manager = ScoringManager(config)
    scoring_inputs = ScoreModelInputs(
        hf_repo_id=config.hf_repo_id,
        competition_id=config.competition_id,
        hotkey=config.hotkey,
        block=config.block
    )
    scoring_manager.current_task = ModelScoreTaskData(inputs=scoring_inputs)
    result = scoring_manager._score_model(scoring_inputs)
    current_task = scoring_manager.get_current_task()
    if result:
        current_task.score = result.get('combined_score')
        current_task.metric_scores = result
    current_task.status = ModelScoreStatus.COMPLETED if result else ModelScoreStatus.FAILED
    print(current_task)
    with open("scoring_task_result.json", "w") as f:
        f.write(current_task.model_dump_json(indent=4))
    
    # Also save the formatted result separately for inspection
    if result:
        import json
        with open("output/voicebench_formatted_result.json", "w") as f:
            json.dump(result, f, indent=4)
