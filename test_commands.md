# Test Commands for VoiceBench Integration

## Prerequisites

1. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create `vali.env` file with:
   ```bash
   DBHOST=your_db_host
   DBNAME=your_db_name
   DBUSER=your_db_user
   DBPASS=your_db_password
   WANDB_API_KEY=your_wandb_key  # Optional if using --wandb.off
   ```

## Running Commands

### Method 1: Two Terminal Setup (Recommended)

#### Terminal 1 - Start Scoring Manager (VoiceBench Integration) on Port 8003:
```bash
source venv/bin/activate
python -m neurons.scoring_api \
    --vali_hotkey YOUR_VALIDATOR_HOTKEY \
    --port 8003 \
    --netuid 21 \
    --wandb.off \
    --offline
```

#### Terminal 2 - Start Validator with API Mode:
```bash
source venv/bin/activate
python -m neurons.validator \
    --subtensor.network finney \
    --netuid 21 \
    --wallet.name omega-test \
    --wallet.hotkey omega-test-hot \
    --logging.debug \
    --wandb.off \
    --offline \
    --run_api \
    --immediate \
    --scoring_api_url http://localhost:8003
```

**Note:** The `--scoring_api_url http://localhost:8003` parameter is actually not needed since the validator code is hardcoded to use `http://localhost:8003` for the API root when not on testnet.

### Method 2: Background Scoring API
```bash
# Start scoring API in background
source venv/bin/activate
nohup python -m neurons.scoring_api \
    --vali_hotkey YOUR_VALIDATOR_HOTKEY \
    --port 8003 \
    --netuid 21 \
    --wandb.off \
    --offline > scoring_api.log 2>&1 &

# Run validator
python -m neurons.validator \
    --subtensor.network finney \
    --netuid 21 \
    --wallet.name omega-test \
    --wallet.hotkey omega-test-hot \
    --logging.debug \
    --wandb.off \
    --offline \
    --run_api \
    --immediate \
    --scoring_api_url http://localhost:8003
```

## Testing VoiceBench Integration with LLM Judge

### Test Complete Validator Workflow:

1. **Test with Hugging Face Model (Full Pipeline):**
   ```bash
   source venv/bin/activate
   python test_validator_hf_model.py
   ```

2. **Quick Test with HF Model (Limited Datasets):**
   ```bash
   source venv/bin/activate
   python test_validator_hf_model.py --quick
   ```

### Test Individual Components:

1. **Test VoiceBench Adapter:**
   ```bash
   source venv/bin/activate
   python -c "
   from neurons.voicebench_adapter import VoiceBenchEvaluator
   evaluator = VoiceBenchEvaluator()
   print('VoiceBench adapter initialized successfully')
   print('Available datasets:', list(evaluator.VOICEBENCH_DATASETS.keys()))
   "
   ```

2. **Test with Specific Model (LandCruiser):**
   ```bash
   source venv/bin/activate
   python test_specific_model.py
   ```

3. **Test Hybrid Scoring Function:**
   ```bash
   source venv/bin/activate
   python neurons/docker_inference_voicebench.py
   ```

4. **Test MySQL Connection:**
   ```bash
   source venv/bin/activate
   python -c "
   from model.storage.mysql_model_queue import init_database
   init_database()
   print('Database connection successful')
   "
   ```

### Test Via Scoring Manager Command Line:

**With VoiceBench + LLM Judge:**
```bash
source venv/bin/activate
python -m neurons.scoring_manager \
    --hf_repo_id "siddhantoon/miner_v2t" \
    --competition_id "voicebench" \
    --hotkey "5FTXYCvdLXS4tD6i87PD2dVFXSUJw4BdGomoecSfATr8hNCL" \
    --block 6155120 \
    --offline \
    --wandb.off
```

**Legacy V2V (Traditional Scoring):**
```bash
source venv/bin/activate
python -m neurons.scoring_manager \
    --hf_repo_id "siddhantoon/miner_v2t" \
    --competition_id "v2v" \
    --hotkey "5FTXYCvdLXS4tD6i87PD2dVFXSUJw4BdGomoecSfATr8hNCL" \
    --block 6155120 \
    --offline \
    --wandb.off
```

## Production Commands

### GPU Node (Scoring API):
```bash
source venv/bin/activate
python -m neurons.scoring_api \
    --vali_hotkey YOUR_VALIDATOR_HOTKEY \
    --port 8003 \
    --netuid 21 \
    --auto_update
```

### CPU Node (Validator):
```bash
source venv/bin/activate
python -m neurons.validator \
    --subtensor.network finney \
    --netuid 21 \
    --wallet.name omega-test \
    --wallet.hotkey omega-test-hot \
    --scoring_api_url http://GPU_NODE_IP:8003 \
    --auto_update
```

## Monitoring Commands

### Check Logs:
```bash
# Scoring API logs
tail -f scoring_api.log

# Validator logs (if using nohup)
tail -f nohup.out

# System logs
journalctl -u your-validator-service -f
```

### Check Ports:
```bash
# Check if scoring API is running on port 8003
netstat -tulpn | grep :8003
# or
lsof -i :8003
```

### Check Docker Containers:
```bash
# List running containers (for model evaluation)
docker ps

# Clean up Docker resources
docker system prune -f
```

## Troubleshooting

### Common Issues:

1. **MySQL Connection Error:**
   ```bash
   pip install mysqlclient PyMySQL
   ```

2. **VoiceBench Path Error:**
   ```bash
   # Ensure VoiceBench is available at /home/salman/anmol/VoiceBench
   ls -la /home/salman/anmol/VoiceBench
   ```

3. **GPU Memory Issues:**
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Clear GPU memory
   python -c "
   import torch
   torch.cuda.empty_cache()
   print('GPU cache cleared')
   "
   ```

4. **Docker Permission Issues:**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Then logout and login again
   ```

## Configuration Files

### VoiceBench Configuration:
Edit `constants_voicebench.py` to modify:
- Dataset selection
- Evaluation weights
- Timeout settings
- Resource limits

### Example Environment Variables:
```bash
export VOICEBENCH_ENABLED=true
export VOICEBENCH_WEIGHT=1.0
export S2S_WEIGHT=0.0
export VOICEBENCH_TIMEOUT=1800
```

## Expected Output

When running successfully, you should see:
- Scoring API: "Starting scoring API on port 8003"
- Validator: "Starting comprehensive VoiceBench evaluation across all datasets..."
- VoiceBench: "VoiceBench evaluation completed. Overall score: X.XXX"

## Performance Notes

- **Full VoiceBench evaluation** takes ~15-30 minutes per model
- **Memory usage**: ~40GB GPU memory recommended
- **Disk space**: Ensure >50GB free for model downloads and Docker containers
- **Network**: Stable internet required for HuggingFace model downloads