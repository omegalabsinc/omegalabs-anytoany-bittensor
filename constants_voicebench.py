"""
VoiceBench Integration Constants and Configuration

This module provides configuration constants for VoiceBench integration,
allowing fine-tuned control over evaluation behavior.
"""

# VoiceBench evaluation configuration
VOICEBENCH_ENABLED = True
VOICEBENCH_PATH = "/home/salman/anmol/VoiceBench"

# Evaluation weights for hybrid scoring
DEFAULT_VOICEBENCH_WEIGHT = 1.0
DEFAULT_S2S_WEIGHT = 0.0

# VoiceBench dataset configuration
# Set to None to use all datasets, or specify list of datasets to use
VOICEBENCH_DATASETS = None  # Use all available datasets

# Performance and resource limits
MAX_VOICEBENCH_SAMPLES_PER_DATASET = None  # None = no limit
VOICEBENCH_TIMEOUT_SECONDS = 1800  # 30 minutes per dataset
VOICEBENCH_MEMORY_LIMIT_GB = 40

# Evaluation modalities
VOICEBENCH_DEFAULT_MODALITY = 'audio'
VOICEBENCH_SUPPORTED_MODALITIES = ['audio', 'text', 'ttft']

# GPT evaluation settings
ENABLE_GPT_EVALUATION = True  # For datasets that support GPT-4 evaluation
GPT_EVALUATION_TIMEOUT = 300  # 5 minutes

# Error handling
VOICEBENCH_MAX_RETRIES = 2
VOICEBENCH_RETRY_DELAY = 30  # seconds

# Logging configuration
VOICEBENCH_LOG_LEVEL = "INFO"
VOICEBENCH_DETAILED_LOGGING = True

# Legacy S2S compatibility
ENABLE_S2S_FALLBACK = False  # Whether to fall back to S2S if VoiceBench fails
S2S_FALLBACK_WEIGHT = 0.5    # Weight for S2S score in case of fallback