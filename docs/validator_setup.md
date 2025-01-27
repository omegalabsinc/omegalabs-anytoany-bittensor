# Validator Setup

Hello validator, welcome to SN21. We have a 2-node validator system:
1. A GPU-node which runs the `neurons/scoring_api.py` script. **We strongly recommend NOT having your validator wallet on this node.** SN21 miners submit arbitrary docker containers for scoring, so not having your validator wallet on this node is a good peace of mind to make sure that malicious miners cannot access your validator wallet. This GPU-node cannot be on Runpod or any other such Docker-container VM service which does not allow creating a container within a container.
2. A CPU-node which runs the actual `neurons/validator.py` script and sets weights. Make sure to have you validator wallet on this node. This can be hosted on Runpod or anywhere else.

## Node 1: Scoring Node Setup (GPU; runs model scoring)

## Requirements
- Python 3.11+ with pip
- GPU with at least 40 GB of VRAM; NVIDIA RTXA6000 is a good choice
- At least 40 GB of CPU RAM
- At least 300 GB of free storage space
- Install libatlas-base-dev: `apt install libatlas-base-dev`

## Running with PM2
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-anytoany-bittensor.git
cd omegalabs-anytoany-bittensor
```
2. Install the requirements: `apt install libatlas-base-dev`
3. Run the scoring API:
```bash
pm2 start auto_updating_scoring_api.sh --name omega-a2a-scoring-api -- \
    --vali_hotkey {your_vali_wallet_hotkey_address} \
    --port {port} \
    --logging.debug
```
### Recommended
- Setting up wandb. Open the `vali.env` file in the repo root directory and set the `WANDB_API_KEY`. Alternatively, you can disable W&B with `WANDB=off` in Step 2.

**Important**: Make note of the public IP address of the scoring node and the port which the API is running on (8080 by default). This URL will be needed for the Validator Node to make scoring requests to the Scoring Node.

3. Check your logs: `pm2 logs omega-a2a-scoring-api`

## Node 2: Validator Node Setup (CPU; sets weights)

## Requirements
- Python 3.11+ with pip
- At least 16 GB of CPU RAM
- At least 200 GB of free storage space
- Install libatlas-base-dev: `apt install libatlas-base-dev`

## Running with Docker
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-anytoany-bittensor.git
cd omegalabs-anytoany-bittensor
```
2. Run the validator:
```bash
make validator WALLET_NAME={wallet} WALLET_HOTKEY={hotkey} PORT={port}
```
### Recommended
- Setting up wandb. Open the `vali.env` file in the repo root directory and set the `WANDB_API_KEY`. Alternatively, you can disable W&B with `WANDB=off` in Step 2.

3. Check your logs: `make check-vali-logs`

## Running with PM2
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-anytoany-bittensor.git
cd omegalabs-anytoany-bittensor
```
2. Install the requirements: `apt install libatlas-base-dev` and `pip install -e .`
3. Run the validator script:
```bash
pm2 start auto_updating_validator.sh --name omega-a2a-validator -- \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```
### Recommended
- Setting up wandb. Set environment variable with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with `--wandb.off`
4. Check the logs: `pm2 logs omega-a2a-validator`
