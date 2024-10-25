build-and-run: a2a sh-headless

sh:
	docker run -it --rm \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		-v ~/.bittensor:/root/.bittensor \
		a2a

sh-headless:
	docker run -it --rm --detach \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		-v ~/.bittensor:/root/.bittensor \
		--name a2a \
		a2a
	docker attach a2a

NETUID ?= 21
WALLET_NAME ?= $(error "Please specify WALLET_NAME=...")
WALLET_HOTKEY ?= $(error "Please specify WALLET_HOTKEY=...")
PORT ?= 8091
WANDB ?= on
WANDBOFF :=
ifeq ($(WANDB), off)
	WANDBOFF := --wandb.off
endif

validator: a2a
	docker run -it --detach --restart always \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		-v ~/.bittensor:/root/.bittensor \
		-e TQDM_DISABLE=True \
		--env-file vali.env \
		--name omega-a2a-validator \
		a2a \
		bash auto_updating_validator.sh --netuid $(NETUID) --wallet.name $(WALLET_NAME) --wallet.hotkey $(WALLET_HOTKEY) --port $(PORT) $(WANDBOFF) --logging.trace
	
manual-validator: a2a
	docker run -it --detach --restart always \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		-v ~/.bittensor:/root/.bittensor \
		-e TQDM_DISABLE=True \
		--env-file .env \
		--name omega-a2a-validator \
		a2a \
		python neurons/validator.py --netuid $(NETUID) --wallet.name $(WALLET_NAME) --wallet.hotkey $(WALLET_HOTKEY) --port $(PORT) $(WANDBOFF) --logging.trace

check-vali-logs:
	docker logs omega-a2a-validator --follow --tail 100

check-miner-logs:
	docker logs a2a --follow --tail 100

a2a:
	docker build -t $@ -f Dockerfile .

models: checkpoints/sd2-1/sd21-unclip-h.ckpt

checkpoints/sd2-1/%:
	mkdir -p $(@D)
	wget https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/main/$* -O $@


# ===== from within container ===== #
CKPT_OUTPUT ?= output_checkpoints/experiment_1
OPTIONS ?=
finetune-x1:
	tune run tune_recipes/lora_finetune_single_device.py --config config/8B_lora.yaml \
		checkpointer.output_dir=$(CKPT_OUTPUT) $(OPTIONS)

finetune-x%:
	tune run --nnodes 1 --nproc_per_node $* \
		tune_recipes/lora_finetune_distributed.py --config config/8B_lora.yaml \
		checkpointer.output_dir=$(CKPT_OUTPUT) $(OPTIONS) \
		gradient-accumulation-steps=32

# eg. make eval_baseline, or make eval_finetune
eval%:
	tune run tune_recipes/eleuther_eval.py --config config/eleuther_eval$*.yaml

mm_eval:
	tune run tune_recipes/mm_eval.py --config config/mm_eval.yaml

# included as an example: replace prompt as required
gen:
	tune run tune_recipes/gen.py --config config/gen.yaml prompt="definition of inference in one word"

# included as an example: replace prompt as required
mmgen:
	tune run tune_recipes/gen.py --config config/gen.yaml image="media/cactus.png" prompt="Caption the image\nImage:\n{image}"
	# tune run tune_recipes/gen.py --config config/gen.yaml image="media/cactus.png" prompt="Image:\n{image}\nCaption the preceding image."

# download+process SAM dataset into imagebind+clip embeddings
sam_llava: ds/sam_llava/00.caption.pt
ds/sam_llava/00.caption.pt:
	python ds/export_sam_llava.py --output-dir=$(@D)

coco_llava_instruct: ds/coco_llava_instruct/train2014/COCO_train2014_000000002270.jpg ds/coco_llava_instruct/llava_instruct_150k.json
ds/coco_llava_instruct/coco_train2014.zip:
	mkdir -p $(@D)
	wget https://huggingface.co/datasets/BAAI/DataOptim/resolve/main/images/coco/train2014.zip?download=true -O $@

ds/coco_llava_instruct/train2014/COCO_train2014_000000002270.jpg: ds/coco_llava_instruct/coco_train2014.zip
	cd ds/coco_llava_instruct && unzip coco_train2014.zip

ds/coco_llava_instruct/llava_instruct_150k.json:
	wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json?download=true -O $@

vision_flan: ds/vision_flan/00.caption.pt
ds/vision_flan/00.caption.pt: ds/vision_flan/images_191task_1k/.done ds/vision_flan/annotation_191-task_1k.json
	python ds/export_vision_flan.py --output-dir=$(@D)

ds/vision_flan/images_191task_1k/.done: ds/vision_flan/image_191-task_1k.zip 
	cd ds/vision_flan && unzip image_191-task_1k.zip
	touch $@

ds/vision_flan/%:
	wget https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/resolve/main/$*?download=true -O $@

bagel: ds/bagel/bagel-input-output-v1.0.parquet
ds/bagel/bagel-%-v1.0.parquet:
	mkdir -p ds/bagel
	wget https://huggingface.co/datasets/jondurbin/bagel-llama-3-v1.0/resolve/main/bagel-$*-v1.0.parquet?download=true -O $@

download-base-model:
	tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir checkpoints/Meta-Llama-3-8B-Instruct

download-datasets: download-sam_llava-dataset download-coco_llava_instruct-dataset download-vision_flan-dataset

download-%-dataset:
	mkdir -p ds/$*
	wget https://huggingface.co/datasets/xzistance/$*/resolve/main/output.parquet -O ds/$*/output.parquet

download-everything: download-base-model download-datasets
