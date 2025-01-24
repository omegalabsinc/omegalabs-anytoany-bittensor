import os
from typing import Optional
from enum import Enum
import traceback
from multiprocessing import Process, Queue
import time

os.environ["USE_TORCH"] = "1"
os.environ["BT_LOGGING_INFO"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
GPU_MEM_GB_REQD = 39
MINS_TO_SLEEP = 10

import bittensor as bt
from pydantic import BaseModel

from model.model_tracker import ModelTracker
from constants import MODEL_EVAL_TIMEOUT
from neurons.docker_inference_v2v import run_v2v_scoring
from neurons.docker_model_scoring import run_o1_scoring
from utilities.temp_dir_cache import TempDirCache
from utilities.gpu import get_gpu_memory

# from neurons.model_scoring import (
#     get_model_score,
#     pull_latest_omega_dataset,
#     cleanup_gpu_memory,
#     log_gpu_memory,
#     get_gpu_memory
# )
# from neurons.v2v_scoring import (
#     compute_s2s_metrics,
#     pull_latest_diarization_dataset,
# )

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
    status: ModelScoreStatus = ModelScoreStatus.SCORING

class ScoringManager:
    def __init__(self):
        self.current_task: Optional[ModelScoreTaskData] = None
        self.model_tracker = ModelTracker()
        bt.logging.error("Note to self: ModelTracker is not implemented yet")

        self.temp_dir_cache = TempDirCache(self.config.cached_models)

        try:
            # early exit if GPU memory insufficient
            total_gb, used_gb, avail_gb = get_gpu_memory()
            if avail_gb < GPU_MEM_GB_REQD:
                m = f"Insufficient GPU Memory available: {avail_gb:.2f} GB available, out of total {total_gb:.2f} GB"
                bt.logging.error(m)
                raise RuntimeError(m)
        except Exception as e:
            bt.logging.error(f"Failed to get GPU memory: {e}: {traceback.format_exc()}")
        
        # TODO: add a background WandB reloading task

    def _score_model(self, inputs: ScoreModelInputs):
        """ Actual model scoring logic """
        start_time = time.time()
        fn_to_call = run_o1_scoring if inputs.competition_id == "o1" else run_v2v_scoring
        score = fn_to_call(
            hf_repo_id=inputs.hf_repo_id,
            hotkey=inputs.hotkey,
            block=inputs.block,
            model_tracker=self.model_tracker,
            local_dir=self.temp_dir_cache.get_temp_dir(inputs.hf_repo_id),
        )
        bt.logging.info(f"Score for {inputs} is {score}, took {time.time() - start_time} seconds")
        return score

    def _score_model_wrapped(self, inputs: ScoreModelInputs, result_queue: Queue):
        """ Wraps the scoring process in a queue to get the result """
        try:
            score = self._score_model(inputs)
            result_queue.put(('success', score))
        except Exception as e:
            bt.logging.error(f"Failed to score model {inputs.hf_repo_id}: {str(e)}\n{traceback.format_exc()}")
            result_queue.put(('error', str(e)))

    def start_scoring(self, inputs: ScoreModelInputs):
        """ Starts the scoring process """
        self.current_task = ModelScoreTaskData(inputs=inputs)

        # Create a queue for getting the result
        result_queue = Queue()

        # Create and start the scoring process
        process = Process(
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
            status, result = result_queue.get()
            if status == 'error':
                self.current_task.status = ModelScoreStatus.FAILED
                self.current_task.score = None
                return

            self.current_task.status = ModelScoreStatus.COMPLETED if result is not None else ModelScoreStatus.FAILED
            self.current_task.score = result

        else:
            bt.logging.error(f"Process terminated without returning a result for {inputs.hf_repo_id}")
            self.current_task.status = ModelScoreStatus.FAILED
            self.current_task.score = None

    def get_current_task(self):
        return self.current_task
