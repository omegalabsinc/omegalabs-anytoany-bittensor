scoring_manager.py entry point (if __name__ == "__main__")
  │
  ├── Parse arguments (lines 308-318)
  │
  ├── ScoringManager.__init__(config) (lines 319-323)
  │   ├── ModelTracker.__init__()
  │   ├── TempDirCache.__init__()
  │   └── get_gpu_memory()
  │
  ├── ScoreModelInputs(competition_id="voicebench", ...) (lines 324-329)
  │
  └── scoring_manager._score_model(scoring_inputs) (lines 182-190)
      │
      ├── [Routing Logic] Check competition_id:
      │   ├── If "voicebench" or "v2v": run_voicebench_scoring()
      │   ├── If "o1": run_o1_scoring()
      │   └── Else: default to run_voicebench_scoring()
      │
      └── [Line 192] run_voicebench_scoring()  [docker_inference_voicebench.py]
          │
          ├── [Lines 140-142] compare_block_and_model(block, hf_repo_id)
          │   └── If block older than model: return penalty_score
          │
          ├── [Lines 152-155] Sanitize repository name
          │   └── Replace '/' with '__' for container naming
          │
          ├── [Line 157] Generate container UID with timestamp
          │
          ├── [Lines 158-165] DockerManager.__init__(gpu_id, container_uid)
          │
          ├── docker_manager.start_container(hf_repo_id, hf_token)
          │   ├── Download model from HuggingFace
          │   ├── Build Docker image if needed
          │   ├── Run Docker container with GPU assignment
          │   └── If failure: return penalty_score + error
          │
          ├── [Lines 167-170] Verify hotkey (if provided)
          │   └── If mismatch: return penalty_score + error
          │
          └── run_voicebench_evaluation()  [voicebench_adapter.py]
              │
              ├── VoiceBenchEvaluator.__init__(max_samples_per_dataset)
              │
              ├── [Lines 493-506] Define VOICEBENCH_DATASETS:
              │   ├── 'commoneval': ['test']
              │   ├── 'mmsu': ['physics']  
              │   ├── 'ifeval': ['test']
              │   └── 'bbh': ['test']
              │
              ├── evaluator.inference_model() (lines 246-344)
              │   │
              │   ├── DockerModelAdapter.__init__(container_url, docker_manager)
              │   │
              │   └── For each dataset in VOICEBENCH_DATASETS:
              │       │
              │       ├── load_dataset('hlt-lab/voicebench', dataset_name)
              │       │
              │       └── _inference_dataset()
              │           │
              │           └── For each sample (up to max_samples):
              │               │
              │               ├── model_adapter.generate_audio(audio_data)
              │               │   │
              │               │   └── docker_manager.inference_v2t()
              │               │       └── HTTP POST to container for voice-to-text
              │               │
              │               └── Store response in results[dataset_name]
              │
              └── calculate_voicebench_scores_with_status(results) (lines 83-176)
                  │
                  └── For each dataset:
                      │
                      ├── [Lines 102-104] Parse dataset name and get evaluator type:
                      │   ├── Extract base name (e.g., 'bbh' from 'bbh_test')
                      │   └── Map to evaluator via DATASET_EVALUATOR_MAP
                      │
                      └── evaluate_dataset_with_proper_evaluator()
                          │
                          ├── If 'open' evaluator AND dataset in NEEDS_LLM_JUDGE:
                          │   ├── evaluate_responses_with_llm()  [llm_judge.py]
                          │   │   └── Call OpenAI/Chutes API for scoring
                          │   ├── Extract LLM scores from responses
                          │   ├── Format scores as "[[score]]" strings
                          │   └── OpenEvaluator.evaluate() with formatted scores
                          │
                          ├── If 'open' evaluator (no LLM needed):
                          │   └── OpenEvaluator.evaluate() directly
                          │
                          ├── If 'mcq' evaluator:
                          │   └── MCQEvaluator.evaluate()
                          │
                          ├── If 'ifeval' evaluator:
                          │   └── IFEvaluator.evaluate()
                          │
                          ├── If 'bbh' evaluator:
                          │   └── BBHEvaluator.evaluate()
                          │
                          ├── If 'harm' evaluator:
                          │   └── HarmEvaluator.evaluate()
                          │
                          └── ERROR HANDLING:
                              └── If evaluator fails: score = 0.0, status = 'error'

RETURN FLOW:
============

run_voicebench_evaluation() returns:
{
    'voicebench_scores': {
        'dataset_name': float,  # Score for each dataset
        ...
    },
    'evaluation_status': {
        'overall': {
            'status': 'success' | 'partial_success' | 'error',
            'evaluators_used': {...},
            'errors': {...}
        }
    }
}

run_voicebench_scoring() returns (lines 199-241):
├── SUCCESS: voicebench_scores from evaluation
├── FAILURE: {"combined_score": penalty_score, "error": reason}
└── ALWAYS: Cleanup in finally block
    ├── Stop Docker container
    ├── Clean Docker resources
    └── Clear GPU memory

_score_model() returns (lines 233-241):
{
    "raw_scores": {
        "voicebench": voicebench_scores dict | None,
        "o1": o1_scores dict | None
    },
    "combined_score": float,  # Weighted average or penalty
    "evaluation_status": {...},  # From evaluators
    "metadata": {
        "competition_id": str,
        "model_id": str,
        "hotkey": str | None,
        ...
    }
}

ERROR HANDLING PATHS:
====================

1. Block comparison failure:
   └── Return penalty_score immediately

2. Container startup failure:
   └── Return penalty_score + error message

3. Hotkey verification failure:
   └── Return penalty_score + "hotkey_mismatch" error

4. VoiceBench evaluation failure:
   ├── Partial results if some datasets succeeded
   └── Error details in evaluation_status

5. Individual evaluator failures:
   ├── Dataset gets 0.0 score
   └── Error logged in status for that dataset

CLEANUP (always executed):
==========================
Finally block (lines 224-239):
├── docker_manager.stop_container()
├── docker_manager.cleanup()
└── torch.cuda.empty_cache() if GPU available

KEY CONFIGURATIONS:
==================

DATASET_EVALUATOR_MAP = {
    'commoneval': 'open',
    'reasoning': 'open', 
    'knowledge': 'open',
    'safety': 'harm',
    'mmsu': 'mcq',
    'ifeval': 'ifeval',
    'bbh': 'bbh'
}

NEEDS_LLM_JUDGE = [
    'commoneval', 
    'reasoning',
    'knowledge'
]

MAX_SAMPLES_PER_DATASET = 100 (configurable)
PENALTY_SCORE = 0.0