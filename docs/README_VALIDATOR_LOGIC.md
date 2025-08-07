# Validator Logic Documentation

## Overview

The validator system operates with a **dual-mode architecture**:
1. **API Mode** (`--run_api` flag): Distributed validation using separate scoring API
2. **Standalone Mode**: Direct model evaluation within the validator

## Architecture Components

### Main Threads

The validator runs **4 concurrent threads**:

1. **Main Thread** (`run()` loop)
   - Handles model evaluation requests
   - Syncs metagraph every 5 minutes
   - Routes to either API mode or standalone evaluation

2. **Weight Thread** (`try_set_scores_and_weights`)
   - Runs every 20 minutes (at :00, :20, :40)
   - Pulls scores from vali_api database
   - Runs baseline incentive mechanism
   - Sets weights on Bittensor chain

3. **Update Thread** (`update_models`)
   - Syncs new model metadata from chain
   - Updates model tracker with latest submissions

4. **Clean Thread** (`clean_models`)
   - Removes outdated models from local storage
   - Runs periodically based on config

## Two-System Operation

### System 1: API Mode (Distributed)

**Your understanding is correct!** Here's the flow:

#### Scoring Thread (Main Loop)
```
1. get_model_to_score() → Pulls next model from vali_api queue
2. Assigns to scoring_api (POST /score_model)
3. Waits for scoring completion
4. post_model_score() → Pushes score + metadata back to vali_api
```

#### Weight Setting Thread
```
1. run_baseline_incentive() → Pulls all scores from vali_api
2. Calculates two-pool incentive system
3. Sets weights on Bittensor chain
```

### System 2: Standalone Mode

All evaluation happens within the validator process:
```
1. Selects models for evaluation
2. Runs scoring locally (no external API)
3. Calculates weights
4. Sets weights on chain
```

## Detailed Flow - API Mode

### Model Scoring Pipeline

```python
# Main thread - run_step()
1. Query for next model:
   uid = await get_model_to_score(competition_id)
   
2. Sync model metadata:
   await model_updater.sync_model(hotkey)
   
3. Send to scoring API:
   POST http://localhost:8003/score_model
   {
     "hf_repo_id": "...",
     "competition_id": "voicebench",
     "hotkey": "...",
     "block": 123456
   }
   
4. Scoring API processes:
   - Starts Docker container
   - Runs VoiceBench evaluation
   - Returns score
   
5. Post results back:
   await post_model_score(hotkey, uid, metadata, hash, score, metrics)
   → Stores in vali_api database
```

### Weight Setting Pipeline

```python
# Weight thread - every 20 minutes
1. Fetch all scores:
   response = requests.get(f"{vali_api}/get_all_model_scores")
   
2. Calculate baseline:
   - Group models by competition
   - Apply two-pool system
   - Above/below baseline allocation
   
3. Set weights on chain:
   subtensor.set_weights(
     netuid=21,
     weights=calculated_weights
   )
```

## Database Flow

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Validator  │──────► Scoring API  │──────►   Docker    │
│   (CPU)     │◄──────│    (GPU)     │◄──────│ Containers │
└─────────────┘      └──────────────┘      └─────────────┘
      │                      │
      ▼                      ▼
┌─────────────────────────────────────────┐
│            Vali API Database            │
│  - Model queue (pending scores)         │
│  - Completed scores + metrics           │
│  - Competition baselines                │
└─────────────────────────────────────────┘
      ▲
      │
┌─────────────┐
│Weight Thread│ (pulls scores every 20 min)
└─────────────┘
```

## Key Functions

### Model Scoring
- `get_model_to_score()`: Fetches next model from queue
- `post_model_score()`: Submits score to database

### Incentive Mechanism
- `run_baseline_incentive()`: New two-pool weight calculation
- `try_set_scores_and_weights()`: Weight setting coordinator

### Competition Management
- Rotates through competitions (voicebench, v2v, o1)
- Each competition has different scoring logic
- Weights distributed based on competition percentages

## Configuration Flags

### API Mode
```bash
--run_api                    # Enable API mode
--scoring_api_url <url>      # Scoring API endpoint
--immediate                  # Skip wait times (testing)
--offline                    # Don't connect to chain
```

### Weight Setting
```bash
--dont_set_weights          # Disable weight setting
--alpha 0.9                 # EMA factor for weights
```

## Error Handling

1. **Scoring Failures**: Model gets score = 0
2. **API Timeouts**: Retry with exponential backoff
3. **Chain Errors**: Log and continue
4. **Docker Failures**: Cleanup and penalize model

## Summary

Your understanding is **correct**:
- **Thread 1** (Main): Pulls models from queue → Sends to scoring API → Posts results back
- **Thread 2** (Weight): Pulls all scores from database → Calculates incentives → Sets chain weights

The system ensures:
- Distributed GPU utilization (scoring on GPU nodes)
- Centralized score management (vali_api database)
- Regular weight updates (every 20 minutes)
- Fault tolerance (continues on individual failures)