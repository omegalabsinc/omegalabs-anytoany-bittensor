# OMEGA Labs Bittensor Any-to-Any (A2A) Subnet

**OMEGA Any-to-Any** is a decentralized, open-source AI project built on the Bittensor blockchain by OMEGA Labs. Our mission is to create state-of-the-art (SOTA)  multimodal any-to-any models by attracting the world's top AI researchers to train on Bittensor, taking advantage of Bittensor's incentivized intelligence platform. Our goal is to establish a self-sustaining, well-resourced research lab, where participants are rewarded for contributing compute and/or research insight.

## Why Any-to-Any? 🧠📚🌃🎧🎥

- **Multimodal First**: A2A jointly models all modalities (text, image, audio, video) at once, with the belief that true intelligence lies in the associative representations present at the intersection of all modalities.
- **Decentralized Data Collection**: Thanks to our [SN24 data collection](https://github.com/omegalabsinc/omegalabs-bittensor-subnet), we leverage a fresh stream of data that mimics real-world demand distribution for training and evaluation. 
- **Incentivized Participation**: World class AI researchers and engineers already love open source. With Bittensor's model for incentivizing intelligence, researchers can be permissionlessly compensated for their efforts and have their compute subsidized according to their productivity.
- **Mixture-of-Experts Architecture**: Incorporates specialist models from other Bittensor subnets, acting as a high-bandwidth, general-purpose router. By being the best open source natively multimodal model, future AI projects can leverage our rich multimodal embeddings to bootstrap their own expert models.
- **Iterative Capability Expansion**: Public demand dictates which capabilities the model learns first through the decentralized incentive structure.
- **Innovative Research Directions**: Integrate emerging state-of-the-art architectures like [early fusion transformers](https://arxiv.org/pdf/2405.09818), [diffusion transformers](https://arxiv.org/pdf/2401.08740), [liquid neural networks](https://arxiv.org/pdf/2006.04439), and [KANs](https://arxiv.org/pdf/2404.19756).

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

1. Clone the repo and install dependencies
2. Configure your miner settings
3. Run the miner script and start earning rewards

### For Validators

1. Review the validation code in `validation/`
2. Set up your validation environment
3. Participate in the validation process to ensure model quality

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

## The Future of A2A 🔮

A2A aims to showcase the power of Bittensor's incentivized intelligence model by attracting and properly incentivizing the best AI researchers worldwide. As the project evolves, we will explore:

- Fully decentralized infrastructure and governance
- Fundamental AI research on architectures beyond transformers and attention mechanisms
- Hyper-connection to the Ω SN24 for diverse, high-quality data
- Innovative monetization strategies to sustain and grow the ecosystem

We believe this decentralized OMEGA A2A research collective will build a strong reputation for Bittensor in the AI research community and beyond. Join us on this epic journey to push the boundaries of decentralized AI! 🌟

## Incentives 🎂

We aim to collectively push forward the SOTA of multimodal and agentic AI research. The incentives in this subnet will evolve as we add modalities and tasks, but they will consistently reflect this underlying goal.

The initial reward structure has two parts:

- Video understanding: can your checkpoint understand and accurately caption video embeddings?
- Language capabilities: does your checkpoint retain the language capabilities of the LLM backbone?

As we improve the incentive scheme over time to create better and more diverse multimodal capabilities, we'll give ample notice and detail of upcoming changes.

## Acknowledgements 🙏

We would like to express our gratitude to the Bittensor Root Validators for their support and belief in our vision. Your funding is critical to the success of OMEGA A2A and the growth of the Bittensor ecosystem.

