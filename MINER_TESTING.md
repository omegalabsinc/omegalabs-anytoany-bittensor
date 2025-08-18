# Miner Model Testing Guide

Test your voice-to-text model locally before deployment using the built-in server testing framework.

## API Requirements

Your model server must implement the following API endpoint:

**POST** `/api/v1/v2t`

### Request Format
```json
{
    "audio_data": "<base64-encoded-numpy-array>",
    "sample_rate": 16000
}
```

- `audio_data`: Base64-encoded NumPy array of audio samples
- `sample_rate`: Audio sample rate (typically 16000 Hz)

### Response Format
```json
{
    "text": "transcribed text output"
}
```

- Must return a JSON object with a `text` field containing the transcription
- Server must respond with HTTP 200 for successful transcriptions
- Any HTTP error status will be treated as a failed transcription

## Testing Your Model

### 1. Set Required API Keys
Either export environment variables:
```bash
export CHUTES_API_KEY="your_chutes_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

Or add them to `vali.env` file:
```bash
CHUTES_API_KEY=your_chutes_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 2. Start Your Model Server
Run your model server locally on port 8000 (or any port):
```bash
# Example - adjust for your setup
python your_model_server.py --port 8000
```

### 3. Run the Test Script
```bash
python -m tests.test_model_scoring_server --experiment_name "my-model-test"
```

### 4. Full Command Options
```bash
python -m tests.test_model_scoring_server \
    --experiment_name "my-model-v1" \
    --api_url "http://localhost:8000/api/v1/v2t" \
    --datasets commoneval wildvoice \
    --report_file "my_results.txt" \
    --logging.debug
```

### Parameters
- `--experiment_name`: Name for tracking results (required)
- `--api_url`: Your model API endpoint (default: `http://localhost:8000/api/v1/v2t`)
- `--datasets`: Specific datasets to test (default: all VoiceBench datasets)
- `--report_file`: File to save results (default: `model_evaluation_report.txt`)

### Output
- **Console**: Detailed results with scores per dataset
- **Report file**: Appended results for tracking multiple experiments
- **Combined score**: Final score used for ranking (0.0-1.0)
- **Success rate**: Percentage of successful transcriptions

#### Sample Output
```
================================================================================
EXPERIMENT: test-scoring-script
TIMESTAMP: 2025-08-18T13:53:42.930568
API URL: http://localhost:8010/api/v1/v2t
EVALUATION TIME: 130.62 seconds
--------------------------------------------------------------------------------
COMBINED SCORE: 0.7308
--------------------------------------------------------------------------------
DATASET SCORES:
  advbench_test                    1.0000  (samples: 2)
  commoneval_test                  0.6000  (samples: 2)
  ifeval_test                      0.5000  (samples: 2)
  wildvoice_test                   0.9000  (samples: 2)
  OVERALL                          0.7308
--------------------------------------------------------------------------------
STATUS: unknown
TOTAL SAMPLES: 8
SUCCESSFUL RESPONSES: 8
SUCCESS RATE: 100.0%
================================================================================
```

The test uses the same evaluation logic as validators, giving you an accurate preview of your model's performance.