from pathlib import Path
from dataclasses import dataclass
from typing import Type, Optional, Any, List, Tuple
import math


@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str


# ---------------------------------
# Project Constants.
# ---------------------------------

# The uid for this subnet.
SUBNET_UID = 21
# The start block of this subnet
SUBNET_START_BLOCK = 2635801
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 18 * 1024 * 1024 * 1024
O1_MODEL_ID = "o1"
V1_MODEL_ID = "v1"
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        reward_percentage=1.0,
        competition_id="v4",
    ),
]
ORIGINAL_COMPETITION_ID = O1_MODEL_ID
BLOCK_DURATION = 12  # 12 seconds


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 1

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.01
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
penalty_score = 0.001
deviation_percent = 0.1
# ---------------------------------
# Model scoring parameters.
# ---------------------------------

MIN_AGE = 0  # 4 hours
MODEL_EVAL_TIMEOUT = 60 * 60 * 2  # 2 hours
MIN_NON_ZERO_SCORES = 1  # Minimum number of non-zero scores required for weight assignment
NUM_CACHED_MODELS = 1
MAX_DS_FILES = 8
PERCENT_IMPROVEMENT = 10 # Minimum percentage improvement required for a model to be considered better than the previous one
PLAYERS_IN_ABOVE_BASELINE = 3
PLAYERS_IN_BELOW_BASELINE = 10

V2T_TIMEOUT = 80  # seconds
V2V_TIMEOUT = 200  # seconds

VOICEBENCH_WEIGHT = 0.7  # Weight of VoiceBench score in final model score
VOICE_MOS_WEIGHT = 0.3  # Weight of VoiceMOS score in final model score
# ---------------------------------
# Weight distribution parameters.
# ---------------------------------
# UID that receives the majority of weight (used as a burn sink).
BURN_UID = 111

#VoiceBench
# to run on full dataset, set VOICEBENCH_MAX_SAMPLES to 100000, and make the dict empty.
VOICEBENCH_MAX_SAMPLES=100000
SAMPLES_PER_DATASET = {
   'commoneval': 50, # 200
   'wildvoice': 100, # 1000
   'advbench': 100, # 520
   'ifeval': 50 # 345
}

# All available VoiceBench datasets with their appropriate splits. which datasets to run.
VOICEBENCH_DATASETS = {
    # 'alpacaeval': ['test'],
    # 'alpacaeval_full': ['test'], 
    'commoneval': ['test'],
    'wildvoice': ['test'],
    # 'openbookqa': ['test'],
    # 'mmsu': ['physics'],  # Use one specific domain instead of 'test'
    # 'sd-qa': ['usa'],  # Using USA dataset only
    # 'mtbench': ['test'],
    'ifeval': ['test'],
    # 'bbh': ['test'],
    'advbench': ['test']
}

CONTAINER_TIMEOUT = 300  # seconds -> to prevent hanging containers