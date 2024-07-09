<div align="center">

# OMEGA Labs Bittensor Any-to-Any Subnet
[![OMEGA](galactic_a2a.png)](https://omegatron.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
### Be, and it becomes ...
---
</div>

- [Introduction](#introduction)
- [Why Any-to-Any?](#why-any-to-any-)
- [Roadmap](#roadmap-)
- [Getting Started](#getting-started-)
  - [For Miners](#for-miners)
  - [For Validators](#for-validators)
- [Current A2A Architecture](#current-a2a-architecture-)
- [The Future of A2A](#the-future-of-omega-a2a-)
- [Incentives](#incentives-)
- [Acknowledgements](#acknowledgements-)
---

## Introduction

**OMEGA Any-to-Any** is a decentralized, open-source AI project built on the Bittensor blockchain by OMEGA Labs. Our mission is to create state-of-the-art (SOTA) multimodal any-to-any models by attracting the world's top AI researchers to train on Bittensor, taking advantage of Bittensor's incentivized intelligence platform. Our goal is to establish a self-sustaining, well-resourced research lab, where participants are rewarded for contributing compute and/or research insight.

**MainNet UID**: 21

**TestNet UID**: 157

## Why Any-to-Any? 🧠📚🌃🎧🎥

- **Multimodal First**: A2A jointly models all modalities (text, image, audio, video) at once, with the belief that true intelligence lies in the associative representations present at the intersection of all modalities.
- **Unified Fundamental Representation of Reality**: The [Platonic Representation Hypothesis](https://phillipi.github.io/prh/) suggests that as AI models increase in scale and capability, they converge towards a shared, fundamental representation of reality. A2A models, by jointly modeling all modalities, are uniquely positioned to capture this underlying structure, potentially accelerating the path towards more general and robust AI.
- **Decentralized Data Collection**: Thanks to our [SN24 data collection](https://github.com/omegalabsinc/omegalabs-bittensor-subnet), we leverage a fresh stream of data that mimics real-world demand distribution for training and evaluation. By frequently refreshing our data collection topics based on gaps in the current data, we avoid the issue of underrepresented data classes (see [this paper](https://arxiv.org/abs/2404.04125) for more discussion on this issue). Through self-play, our SN's best checkpoints can learn from each other and pool their intelligence.
- **Incentivized Research**: World class AI researchers and engineers already love open source. With Bittensor's model for incentivizing intelligence, researchers can be permissionlessly compensated for their efforts and have their compute subsidized according to their productivity.
- **Bittensor Subnet Orchestrator**: Incorporates specialist models from other Bittensor subnets, acting as a high-bandwidth, general-purpose router. By being the best open source natively multimodal model, future AI projects can leverage our rich multimodal embeddings to bootstrap their own expert models.
- **Public-Driven Capability Expansion**: Public demand dictates which capabilities the model learns first through the decentralized incentive structure.
- **Beyond Transformers**: Integrate emerging state-of-the-art architectures like [early fusion transformers](https://arxiv.org/pdf/2405.09818), [diffusion transformers](https://arxiv.org/pdf/2401.08740), [liquid neural networks](https://arxiv.org/pdf/2006.04439), and [KANs](https://arxiv.org/pdf/2404.19756). 

## Roadmap 🚀

### Phase 1: Foundation (Remainder of Q2 2024)

- [x] Design a hard-to-game validation mechanism that rewards deep video understanding
- [ ] Produce the first checkpoint with SOTA image and video understanding capabilities with our ImageBind + Llama-3 architecture as a proof-of-concept starting point
- [ ] Generalize the validation mechanism to enable broad architecture search and new multimodal tokenization methods
- [ ] Onboard 20+ top AI researchers from frontier labs and open source projects
- [ ] Expand SN24 data collection beyond YouTube to include multimodal websites (e.g. Reddit, blogposts) and synthetic data pipelines
- [ ] Launch OMEGA Focus screen recording app, providing rich data for modelling long-horizon human workflows, combatting the hallucination and distraction problem found in top closed-source LLMs

### Phase 2: Fully Multimodal (Q3 2024)

- [ ] Produce the first any-to-any checkpoint natively modelling all modalities that can beat other OSS models on top multimodal and reasoning benchmarks
- [ ] Develop a user-friendly interface for miners and validators to interact with the subnet's top models
- [ ] Onboard 50 more top AI researchers from top labs and open source research collectives
- [ ] Publish a research paper on A2A's architecture, incentive model, and performance
- [ ] Release open source multimodal embedding models (based on our top A2A checkpoint's internal embedding space) for other labs to condition their models on
- [ ] Integrate a framework that can auto-evaluate all the models & commodities produced by other subnets on Bittensor which our top models can then interact with, both through tool-use and through native communication in the latent-space via projection modules

### Phase 3: Exponential Open Research Progress (Q4 2024)

- [ ] Produce the first any-to-any OSS checkpoint that beats all closed-source SOTA general intelligence models
- [ ] Establish partnerships with AI labs, universities, and industry leaders to drive adoption
- [ ] Expand our one-stop-shop Bittensor model evaluation and router framework to arbitrary open source and closed-source checkpoints and APIs
- [ ] Implement task-driven learning, with OMEGA Labs routinely curating high-signal tasks for model trainers to master
- [ ] Start crafting an entirely new "online" validation mechanism that rewards miners for producing agentic models that can complete real-world tasks
- [ ] Use our top checkpoint to power up the multimodal intelligence features of the OMEGA Focus app

### Phase 4: Agentic Focus (Q1 2025)

- [ ] Launch our agent-focused "online" validation mechanism centered around long-range task completion
- [ ] Achieve SOTA performance on agent benchmarks
- [ ] Use OMEGA Focus as an outlet to provide OMEGA digital twin companions to users
- [ ] Launch an app store for A2A-powered applications leveraging our open source models
- [ ] Reach 10M+ users with the OMEGA Focus app

## Getting Started 🏁

### For Miners

#### Requirements
- Python 3.11+ with pip
- GPU with at least 40 GB of VRAM; NVIDIA RTXA6000 is a good choice, or use a 1024xH100 if you wanna train a **really** good model :sunglasses:
- At least 40 GB of CPU RAM
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Setup
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-anytoany-bittensor.git
cd omegalabs-anytoany-bittensor
```
2. Install the requirements:
  - Using docker: `make build-and-run`
  - Using your local Python: `pip install -e .`
3. Log into Huggingface: `huggingface-cli login`. Make sure your account has access to Llama-3-8B on HF, you can get access [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
4. Download the base model and datasets: `make download-everything`
5. Start a finetuning run: `make finetune-x1`
  - Tweak `config/8B_lora.yaml` to change the hyperparameters of the training run.
6. Upload the model to Huggingface:
```
python miner_utils/upload_model.py \
    --hf_repo_id {HF REPO ID e.g. omegalabsinc/omega_agi_model} \
    --wallet.name {miner wallet name} \
    --wallet.hotkey {miner hotkey} \
    --model_dir {the directory that the checkpoint is saved in e.g. output_checkpoints/experiment_1/} \
    --epoch 0 \
    --netuid NETUID
```
NOTE: If you want to run on testnet, simply add `--subtensor.network test` at the end of the command and use `--netuid 157`.

### For Validators

#### Requirements
- Python 3.11+ with pip
- GPU with at least 40 GB of VRAM; NVIDIA RTXA6000 is a good choice
- At least 40 GB of CPU RAM
- At least 300 GB of free storage space
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.
- Install libatlas-base-dev: `apt install libatlas-base-dev`

#### Running with Docker
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-anytoany-bittensor.git
cd omegalabs-anytoany-bittensor
```
2. Run the validator:
```bash
make validator WALLET_NAME={wallet} WALLET_HOTKEY={hotkey} PORT={port}
```
##### Recommended
- Setting up wandb. Open the `vali.env` file in the repo root directory and set the `WANDB_API_KEY`. Alternatively, you can disable W&B with `WANDB=off` in Step 2.
<details>
  <summary>To run with manually updating validator</summary>
  
  Simply run the following command instead:
  ```bash
  make manual-validator WALLET_NAME={wallet} WALLET_HOTKEY={hotkey} PORT={port}
  ```
</details>
3. Check your logs: `make check-vali-logs`

#### Running with PM2
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
##### Recommended
- Setting up wandb. Set environment variable with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with `--wandb.off`
<details>
  <summary>To run with manually updating validator</summary>
  
  Simply run the following command instead:
  ```bash
  pm2 start neurons/validator.py --name omega-a2a-validator -- \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
  ```
</details>
4. Check the logs: `pm2 logs omega-a2a-validator`

## Current A2A Architecture 🤖

At launch, we are starting with an approach that builds on one of the most powerful and mainstream LLMs as a backbone: Llama 3. A multilayer perceptron (MLP) projects Imagebind embeddings directly into the pretrained Llama 3 internal state, allowing a finetuned model to develop video understanding.

The provided mining repo with default settings will train both:

-  the encoding projection layer, which encodes Imagebind embeddings into Llama 3 states, and
-  a LoRA adapter which allows the underlying LLM to develop multimodal understanding

There are several immediate areas miners can investigate in order to produce a checkpoint with improved multimodal understanding:

- Train for longer,
- Find better hyperparameters: learning rate, optimizer, batch sizes, gradient accumulation, etc.,
- Use additional datasets (more diverse, or even larger than the SN24 dataset),
- Try different LoRA parameters, or finetune all parameters.

In the near future we'll enable much deeper architecture search, allowing researchers to experiment with different LLM backbones, vastly different encoders and decoders.

## The Future of OMEGA A2A 🔮

OMEGA A2A is poised to revolutionize the AI landscape by harnessing the power of Bittensor's incentivized intelligence model and attracting the world's top AI researchers. Our mission is to push the boundaries of what's possible in decentralized AI, focusing on:

- Developing fully multimodal, any-to-any models that outperform all other open-source solutions
- Creating an AI gateway framework to seamlessly integrate and evaluate models from across the Bittensor ecosystem and beyond
- Implementing task-driven learning and agent-focused validation to create models capable of completing complex, real-world tasks
- Powering up the OMEGA Focus app with cutting-edge multimodal intelligence and personalized digital twin companions

As we progress, we will explore fully decentralized infrastructure and governance to ensure a truly democratized AI ecosystem. Our research will explore groundbreaking architectures beyond transformers and attention mechanisms, pushing the limits of AI capabilities.

By hyper-connecting to the Ω SN24, we will access diverse, high-quality data that fuels our models' growth and enables them to tackle a wide range of tasks. Innovative monetization strategies will be implemented to sustain and grow the ecosystem, ensuring long-term success and viability.

Through the tireless efforts of our decentralized OMEGA A2A research collective, we aim to showcase the immense potential of Bittensor's incentivized intelligence model and establish ourselves as leaders in the AI research community and beyond.

Join us on this transformative journey as we shape the future of decentralized AI, unlocking new possibilities and creating a more accessible, powerful, and innovative AI ecosystem for all. :rocket:

## Incentives 🎂

We aim to collectively push forward the SOTA of multimodal and agentic AI research. The incentives in this subnet will evolve as we add modalities and tasks, but they will consistently reflect this underlying goal.

The initial reward structure has two parts:

- Video understanding: can your checkpoint understand and accurately caption video embeddings?
- Language capabilities: does your checkpoint retain the language capabilities of the LLM backbone?

As we improve the incentive scheme over time to create better and more diverse multimodal capabilities, we'll give ample notice and detail of upcoming changes.

## Acknowledgements 🙏

Thank you to [Nous Research](https://github.com/NousResearch/finetuning-subnet), [MyShell](https://github.com/myshell-ai/MyShell-TTS-Subnet), and [Impel](https://github.com/impel-intelligence/dippy-bittensor-subnet) for the structure of the miner chain model upload and validator comparison scoring logic!
