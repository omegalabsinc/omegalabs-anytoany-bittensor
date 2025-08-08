# VoiceBench Implementation Plan

## Overview
This document outlines the implementation plan for integrating VoiceBench evaluation system with the validator, scoring API, and dashboard infrastructure.

---

## Implementation Tasks

### 1. ~~Fallback LLM Judge~~ ✅ (Completed)
- Already implemented with Chutes/OpenAI client fallback
- Located in `neurons/llm_judge.py`

---

### 2. **Scoring Manager Output Format** 🔧
**Current Issue:** Output format needs both raw and combined scores for dashboard compatibility

**Required Output Structure:**
```json
{
  "raw_scores": {
    "alpacaeval_test": 0.6,
    "commoneval_test": 0.7,
    "wildvoice_test": 0.5,
    "openbookqa_test": 0.4,
    "mmsu_physics": 0.3,
    "ifeval_test": 0.8,
    "bbh_test": 0.48,
    "advbench_test": 1.0
  },
  "combined_score": 0.52,  // Weighted average
  "evaluation_status": {
    "alpacaeval_test": {
      "evaluator": "open",
      "status": "completed",
      "samples_evaluated": 50,
      "samples_total": 50
    },
    // ... for each dataset
  },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00Z",
    "total_time_seconds": 1800,
    "llm_api_calls": 150,
    "llm_model_used": "gpt-4o-mini"
  }
}
```

**Files to Modify:**
- `neurons/scoring_manager.py` - Update `run_voicebench_scoring()` to format output correctly
- `model/storage/mysql_storage.py` - Update storage methods to handle new format

**Implementation Details:**
```python
def format_voicebench_output(scores, status, metadata):
    return {
        'raw_scores': {k: v for k, v in scores.items() if k != 'overall'},
        'combined_score': scores.get('overall', 0.0),
        'evaluation_status': status,
        'metadata': metadata
    }
```

---

### 3. **Scoring API Accept VoiceBench** 🔧
**Current:** Only accepts "v2v" competition type
**Need:** Accept "voicebench" as valid competition type

**API Endpoint Modifications:**
```python
# neurons/scoring_api.py
@app.route('/score_model', methods=['POST'])
def score_model():
    data = request.json
    competition = data.get('competition', 'v2v')
    
    if competition == 'voicebench':
        result = run_voicebench_scoring(
            hf_repo_id=data['hf_repo_id'],
            hotkey=data['hotkey'],
            block=data.get('block'),
            use_cache=data.get('use_cache', False)
        )
    elif competition == 'v2v':
        result = run_v2v_scoring(...)
    else:
        return jsonify({'error': f'Unknown competition: {competition}'}), 400
    
    return jsonify(result)
```

**Files to Modify:**
- `neurons/scoring_api.py` - Add voicebench routing
- `neurons/api_server.py` - Update validation to accept voicebench

---

### 4. **Validator Thread Accept VoiceBench** 🔧
**Current:** Validator only knows how to process v2v scores
**Need:** Process voicebench scores with dataset-specific handling

**Validator Modifications:**
```python
# neurons/validator.py
class Validator:
    def process_miner_scores(self, scores_data):
        competition = scores_data.get('competition', 'v2v')
        
        if competition == 'voicebench':
            return self.process_voicebench_scores(scores_data)
        else:
            return self.process_v2v_scores(scores_data)
    
    def process_voicebench_scores(self, scores_data):
        # Extract dataset-specific scores
        raw_scores = scores_data.get('raw_scores', {})
        combined_score = scores_data.get('combined_score', 0.0)
        
        # Update miner's score in database
        self.update_miner_score(
            miner_uid=scores_data['miner_uid'],
            score=combined_score,
            dataset_scores=raw_scores,
            competition='voicebench'
        )
```

**Files to Modify:**
- `neurons/validator.py` - Add voicebench score processing
- `neurons/update_thread.py` - Handle voicebench in update loop
- `neurons/clean_thread.py` - Consider voicebench in cleanup logic

---

### 5. **Vali API Accept VoiceBench Data** 🔧
**Need:** New API endpoints to receive and store VoiceBench evaluation results

**New API Endpoints:**
```python
# POST /api/voicebench/submit_scores
{
  "miner_uid": 123,
  "hotkey": "5FTX...",
  "block": 1234567,
  "raw_scores": {
    "alpacaeval_test": 0.6,
    "openbookqa_test": 0.4,
    // ... all dataset scores
  },
  "combined_score": 0.52,
  "evaluation_status": {...},
  "metadata": {...}
}

# GET /api/voicebench/scores/{miner_uid}
# Returns historical voicebench scores for a miner

# GET /api/voicebench/leaderboard
# Returns ranked list of miners by voicebench performance
```

**Files to Modify:**
- `vali_api/routes/voicebench.py` - Create new route file
- `vali_api/models/voicebench.py` - Create data models
- `vali_api/database/schema.py` - Add voicebench tables

---

### 6. **Per-Dataset Score Storage & Normalization** 🔧
**Requirements:** Store individual dataset scores with clear normalization for dashboard

**Database Schema:**
```sql
-- Main voicebench scores table
CREATE TABLE voicebench_scores (
    id INT PRIMARY KEY AUTO_INCREMENT,
    miner_uid INT NOT NULL,
    hotkey VARCHAR(64) NOT NULL,
    block INT NOT NULL,
    combined_score FLOAT NOT NULL,        -- Overall weighted score (0.0-1.0)
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,                        -- Store evaluation metadata
    INDEX idx_miner_block (miner_uid, block),
    INDEX idx_timestamp (timestamp DESC)
);

-- Dataset-specific scores table
CREATE TABLE voicebench_dataset_scores (
    id INT PRIMARY KEY AUTO_INCREMENT,
    score_id INT NOT NULL,                -- FK to voicebench_scores.id
    dataset_name VARCHAR(50) NOT NULL,    -- 'alpacaeval_test', 'bbh_test', etc
    raw_score FLOAT NOT NULL,             -- Original score from evaluator
    normalized_score FLOAT NOT NULL,      -- Normalized to 0.0-1.0
    evaluator_type VARCHAR(20),           -- 'open', 'mcq', 'bbh', etc
    samples_evaluated INT,
    samples_total INT,
    status VARCHAR(20),                   -- 'completed', 'failed', 'partial'
    FOREIGN KEY (score_id) REFERENCES voicebench_scores(id) ON DELETE CASCADE,
    INDEX idx_score_dataset (score_id, dataset_name)
);
```

**Normalization Logic:**
```python
class ScoreNormalizer:
    """Handles score normalization for different evaluator types"""
    
    NORMALIZATION_RULES = {
        'open': {
            'input_range': (1, 5),      # GPT scores 1-5
            'output_range': (0, 1),      # Normalized to 0-1
            'transform': lambda x: (x - 1) / 4
        },
        'mcq': {
            'input_range': (0, 100),     # Accuracy percentage
            'output_range': (0, 1),      # Already 0-1 after /100
            'transform': lambda x: x / 100.0
        },
        'bbh': {
            'input_range': (0, 100),     # Accuracy percentage
            'output_range': (0, 1),
            'transform': lambda x: x / 100.0
        },
        'ifeval': {
            'input_range': (0, 1),       # Already normalized
            'output_range': (0, 1),
            'transform': lambda x: x
        },
        'harm': {
            'input_range': (0, 1),       # Refusal rate
            'output_range': (0, 1),
            'transform': lambda x: x
        }
    }
    
    @classmethod
    def normalize(cls, score, evaluator_type):
        """Normalize score based on evaluator type"""
        if evaluator_type not in cls.NORMALIZATION_RULES:
            return score  # Return as-is if unknown type
        
        rule = cls.NORMALIZATION_RULES[evaluator_type]
        return rule['transform'](score)
```

**Files to Modify:**
- `model/storage/mysql_storage.py` - Add voicebench storage methods
- `model/database/schema.sql` - Add new tables
- `neurons/score_normalizer.py` - Create normalization module

---

## Implementation Order

### **Phase 1: Core Functionality** (Week 1)
1. ✅ ~~Fallback LLM Judge~~ (Already done)
2. 🔧 Scoring Manager Output Format
3. 🔧 Score Normalization Logic

### **Phase 2: API Integration** (Week 2)  
4. 🔧 Scoring API Accept VoiceBench
5. 🔧 Validator Thread Processing
6. 🔧 Update & Clean Thread Integration

### **Phase 3: Storage & Dashboard** (Week 3)
7. 🔧 Database Schema Creation
8. 🔧 Vali API Endpoints
9. 🔧 Dashboard Integration

---

## Testing Strategy

### Unit Tests
- Test score normalization for each evaluator type
- Test output format generation
- Test API endpoint routing

### Integration Tests
- End-to-end test: HF model → Scoring → Storage → Dashboard
- Test fallback behavior when primary LLM fails
- Test dataset-specific score storage

### Load Tests
- Test with 50 samples per dataset
- Test concurrent evaluations
- Monitor API rate limits

---

## Configuration Updates

### Environment Variables (vali.env)
```bash
# VoiceBench Configuration
VOICEBENCH_ENABLED=true
VOICEBENCH_WEIGHT=1.0           # Weight vs v2v scoring
VOICEBENCH_TIMEOUT=1800          # 30 minutes
VOICEBENCH_MAX_SAMPLES=50        # Per dataset

# LLM Configuration (with fallback)
CHUTES_API_BASE=https://...
CHUTES_API_KEY=...
CHUTES_MODEL=Qwen/...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

### Constants File Updates
```python
# constants_voicebench.py
DATASET_WEIGHTS = {
    'alpacaeval_test': 2.0,
    'commoneval_test': 2.0,
    'wildvoice_test': 2.0,
    'openbookqa_test': 1.0,
    'mmsu_*': 1.0,  # All MMSU subjects
    'ifeval_test': 1.5,
    'bbh_test': 1.0,
    'advbench_test': 1.0
}

EVALUATOR_COSTS = {
    'open': 0.001,    # Per sample (uses GPT)
    'mcq': 0.0,       # Free (local)
    'bbh': 0.0,       # Free (local)
    'ifeval': 0.0,    # Free (local)
    'harm': 0.0       # Free (local)
}
```

---

## Success Metrics

1. **Functionality**
   - ✅ All 11 VoiceBench datasets evaluated correctly
   - ✅ Scores properly normalized and stored
   - ✅ Dashboard displays per-dataset breakdown

2. **Performance**
   - ⏱️ Full evaluation < 30 minutes
   - 💰 Cost reduction > 70% vs all-GPT approach
   - 🔄 Fallback prevents evaluation failures

3. **Reliability**
   - 📊 Status tracking for debugging
   - 🔁 Automatic retry with fallback
   - 💾 All scores persisted to database

---

## Notes

- Keep v2v scoring operational during migration
- Maintain backwards compatibility with existing dashboard
- Document all API changes for frontend team
- Consider caching evaluation results for identical inputs
- Monitor GPT API usage to optimize costs

---

## References

- VoiceBench Paper: [Link to paper]
- Original VoiceBench Repo: https://github.com/VoiceBench/VoiceBench
- Dashboard Requirements: [Link to requirements doc]
- API Documentation: [Link to API docs]