# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict

import torch
from torch import nn
from omegaconf import DictConfig
from PIL import Image

from torchtune import config, utils
from torchtune.utils._generation import sample
from torchtune.models import convert_weights
from torchtune.data import Message

from models.tokenizer import START_IMAGE, END_IMAGE
from imagebind.models.imagebind_model import ModalityType
from diffusers import DiffusionPipeline

from models import add_proj_convert_weights, _BASE_TRAINABLE

log = utils.get_logger("DEBUG")
add_proj_convert_weights()


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._embed_model = self._setup_embed_model(model_cfg=cfg.embed_model)
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._image_ids = self._tokenizer.encode(START_IMAGE + END_IMAGE, add_eos=False, add_bos=False)
        self.use_clip = cfg.model.use_clip
        if self.use_clip:
            self._clip_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=self._dtype).to(self._device)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    def _setup_embed_model(
        self,
        model_cfg: DictConfig,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

#         if self._quantization_mode is not None:
#             model = self._quantizer.quantize(model)
#             model = model.to(device=self._device, dtype=self._dtype)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Embed model is initialized with precision {self._dtype}.")

#         # Ensure the cache is setup on the right device
#         with self._device:
#             model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model


    def mm_process_prompt(self, prompt):
        return prompt.replace("{image}", f"{START_IMAGE}0{END_IMAGE}")

    def get_image_embed(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path).convert('RGB')
            img = self._embed_model.transform_from_pil(img).unsqueeze(0).to(device=self._device, dtype=self._dtype)
            return self._embed_model({ModalityType.VISION: img})[ModalityType.VISION].squeeze(0)

    def get_clip_embed(self, image_path):
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            img = self._clip_pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
            img = img.to(device=self._device, dtype=self._dtype)
            return self._clip_pipe.image_encoder(img).image_embeds.squeeze(0)

    def extract_mm_context(self, image_path, tokens):
        context = {}
        in_image_embed = False
        for idx, tok in enumerate(tokens):
            in_image_embed = in_image_embed and not tok == self._image_ids[1]
            if in_image_embed:
                #tokens[idx] # to support multiple embeds: get the value, match it up with the sample embed
                context[idx] = {
                    "ib_embed": self.get_image_embed(image_path),
                    "clip_embed": self.get_clip_embed(image_path) if self.use_clip else None,
                }
            in_image_embed = in_image_embed or tok == self._image_ids[0]
        return context

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        messages = [
            Message(
                role="user",
                content=self.mm_process_prompt(cfg.prompt),
            ),
            Message(
                role="assistant",
                content="",
            )
        ]
        tokens, mask = self._tokenizer.tokenize_messages(messages)
        tokens = tokens[:-2] # strip eot and eos
        mm_context = [self.extract_mm_context(cfg.image, tokens)] # context should be a list, batch-id indexed
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        self._model.tok_embeddings.set_context(mm_context)
        self._model.output.set_context(mm_context)

        bos_id = self._tokenizer.tt_model.encode("<|begin_of_text|>", allowed_special="all")[0]
        allowed_id = self._tokenizer.tt_model.encode("<|eot_id|><|start_image|><|end_image|>", allowed_special="all")
        disallowed_tokens = list(set(range(bos_id, bos_id + 256)) - set(allowed_id))
        # self._model.output.weight.data[disallowed_tokens, :] = 0

        def custom_generate_next_token(model, input_pos, x, temperature=1.0, top_k=None):
            model.tok_embeddings.set_context([])
            model.output.set_context([])
            # x: [1, s]
            # input_pos: [s]
            logits = model(x, input_pos)
            # logits: [1, s, v] where v is vocab_size
            # for sampling we extract the logits for the
            # last token and convert to shape: [v]
            logits = logits[0, -1]
            # logits[disallowed_tokens] = float("-inf")
            # sample the next token
            token = sample(logits, temperature, top_k)
            if token in disallowed_tokens:
                return torch.tensor([self._tokenizer.eos_id]).to(x)
            return token

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            log.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                custom_generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                eos_id=self._tokenizer.eos_id,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            log.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        t0 = time.perf_counter()
        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_id=self._tokenizer.eos_id,
            custom_generate_next_token=custom_generate_next_token,
        )
        t = time.perf_counter() - t0

        log.info(self._tokenizer.decode(generated_tokens))

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens) - prompt.size(0)
        tokens_sec = tokens_generated / t
        log.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        log.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        log.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
