# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List

import torch
from torch import nn
from omegaconf import DictConfig
from PIL import Image

from torchtune import config, utils
from torchtune.utils._generation import sample
from torchtune.models import convert_weights
from torchtune.data import Message

from models.tokenizer import START_IMAGE, END_IMAGE, START_AUDIO, END_AUDIO, START_VIDEO, END_VIDEO
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
        self._quantizer = config.instantiate(cfg.inference.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)
        self.prompt_template = cfg.inference.prompt_template
        perception_tokens = cfg.model.perception_tokens
        self._perception_tokens = ("0 " * perception_tokens)[:perception_tokens]
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
        with self._device:
            self._model.setup_caches(max_batch_size=cfg.batch_size, dtype=self._dtype)

        self._embed_model = self._setup_embed_model(model_cfg=DictConfig({"_component_": "models.imagebind_huge"}))
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._mm_ids_start = self._tokenizer.encode(START_IMAGE + START_AUDIO + START_VIDEO, add_eos=False, add_bos=False)
        self._mm_ids_end = self._tokenizer.encode(END_IMAGE + END_AUDIO + END_VIDEO, add_eos=False, add_bos=False)
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
        log.debug(f"Model is initialized with precision {self._dtype}.")

        return model

    def _setup_embed_model(
        self,
        model_cfg: DictConfig,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.debug(f"Embed model is initialized with precision {self._dtype}.")

        return model

    def mm_process_prompt(self, prompt):
        return (
            prompt
                .replace("{image}", f"{START_IMAGE}{self._perception_tokens}{END_IMAGE}")
                .replace("{audio}", f"{START_AUDIO}{self._perception_tokens}{END_AUDIO}")
                .replace("{video}", f"{START_VIDEO}{self._perception_tokens}{END_VIDEO}")
            )

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

    def extract_mm_context(self, video_ib_embed, tokens):
        context = {}
        in_mm_embed = False
        for idx, tok in enumerate(tokens):
            in_mm_embed = in_mm_embed and not tok in self._mm_ids_end
            if in_mm_embed:
                #tokens[idx] # to support multiple embeds: get the value, match it up with the sample embed
                context[idx] = {
                    "ib_embed": video_ib_embed.to(dtype=self._dtype, device=self._device),
                }
            in_mm_embed = in_mm_embed or tok in self._mm_ids_start
        return context

    @torch.no_grad()
    def generate(self, cfg: DictConfig, video_ib_embed: List[float]):
        messages = [
            Message(
                role="user",
                content=self.mm_process_prompt(self.prompt_template),
            ),
            Message(
                role="assistant",
                content="",
            )
        ]
        tokens, mask = self._tokenizer.tokenize_messages(messages)
        tokens = tokens[:-2] # strip eot and eos
        mm_context = [self.extract_mm_context(video_ib_embed, tokens)] # context should be a list, batch-id indexed
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        self._model.tok_embeddings.set_context(mm_context)
        self._model.output.set_context(mm_context)

        bos_id = self._tokenizer.tt_model.encode("<|begin_of_text|>", allowed_special="all")[0]
        allowed_id = self._tokenizer.tt_model.encode(f"<|eot_id|>{START_IMAGE}{END_IMAGE}{START_AUDIO}{END_AUDIO}{START_VIDEO}{END_VIDEO}", allowed_special="all")
        disallowed_tokens = list(set(range(bos_id, bos_id + 256)) - set(allowed_id))
        # self._model.output.weight.data[disallowed_tokens, :] = 0

        def custom_generate_next_token(model, input_pos, x, temperature=1.0, top_k=None):
            model.tok_embeddings.set_context([])
            model.output.set_context([])
            # x: [1, s]
            # input_pos: [s]
            logits = model(x, input_pos=input_pos)
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

        cleaned_tokens = [t for t in generated_tokens[len(prompt):] if t not in disallowed_tokens + allowed_id]
        caption = self._tokenizer.decode(cleaned_tokens)

        # log.debug(f"Generated caption: {caption} in {t:.02f} sec")

        return caption


    @torch.no_grad()
    def generate_batch(self, cfg: DictConfig, video_ib_embed: torch.Tensor):
        batch_dim = video_ib_embed.size(0)
        messages = [
            Message(
                role="user",
                content=self.mm_process_prompt(self.prompt_template),
            ),
            Message(role="assistant", content="")
        ]
        tokens, mask = self._tokenizer.tokenize_messages(messages)
        tokens = tokens[:-2] # strip eot and eos
        mm_context = [self.extract_mm_context(e, tokens) for e in video_ib_embed] # context should be a list, batch-id indexed
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device).expand(batch_dim, -1).clone()
        prompt_length = prompt.size(1)

        self._model.tok_embeddings.set_context(mm_context)
        self._model.output.set_context(mm_context)

        bos_id = self._tokenizer.tt_model.encode("<|begin_of_text|>", allowed_special="all")[0]
        allowed_id = self._tokenizer.tt_model.encode(f"<|eot_id|>{START_IMAGE}{END_IMAGE}{START_AUDIO}{END_AUDIO}{START_VIDEO}{END_VIDEO}", allowed_special="all")
        disallowed_tokens = list(set(range(bos_id, bos_id + 256)) - set(allowed_id))

        def generate_next_token(model, input_pos, x, temperature=1.0, top_k=None):
            # x: [B, s]
            # input_pos: [s]
            # logits: [B, s, v] where v is vocab_size
            logits = model(x, input_pos=input_pos)[:, -1]
            tokens = sample(logits, temperature, top_k)
            return torch.tensor([
                [self._tokenizer.eos_id if t in disallowed_tokens else t for t in toks]
                for toks in tokens
            ]).to(x.device)

        generated_tokens = prompt.clone()
        # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
        stop_token_reached = torch.zeros(batch_dim, dtype=torch.bool, device=prompt.device)

        # generate the first tokens conditioned on the prompt
        tokens = generate_next_token(
            self._model,
            input_pos=torch.arange(0, prompt_length, device=prompt.device),
            x=prompt,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        eot_reached_b = tokens == self._tokenizer.eot_id
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

        self._model.tok_embeddings.set_context([])
        self._model.output.set_context([])

        input_pos = torch.tensor([prompt_length], device=prompt.device)
        for _ in range(cfg.max_new_tokens - 1):
            tokens = generate_next_token(
                self._model, input_pos=input_pos, x=tokens, temperature=cfg.temperature, top_k=cfg.top_k
            )
            eot_reached_b |= tokens == self._tokenizer.eot_id
            tokens *= ~eot_reached_b
            generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
            if eot_reached_b.all():
                print('eot_reached_b.all()')
                break
            input_pos += 1

        captions = []
        for caption_tokens in generated_tokens.tolist():
            captions.append(self._tokenizer.decode(caption_tokens[prompt.size(1):]))
        return captions


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
