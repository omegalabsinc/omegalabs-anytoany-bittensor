import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import sys
import time
from tqdm import tqdm
import argparse
from pathlib import Path

from typing import Any, Callable, Dict, List, Tuple, Optional
from omegaconf import DictConfig, OmegaConf

from omegaconf import DictConfig

from torchtune import config, utils
from torchtune.models import convert_weights
from torchtune.data import Message

from diffusers import DiffusionPipeline

from imagebind.models.imagebind_model import ModalityType

from models.tokenizer import START_IMAGE, END_IMAGE
from torchtune.modules.tokenizers._tiktoken import (
    START_HEADER_ID,
    END_HEADER_ID
)

from models import add_proj_convert_weights

log = utils.get_logger("DEBUG")
add_proj_convert_weights()

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--config', type=Path)
    return a.parse_args()

try:
    import lmms_eval
    from lmms_eval.evaluator import simple_evaluate
    from lmms_eval.tasks import get_task_dict
    from lmms_eval.api.model import lmms
    from lmms_eval import utils as lmms_utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.registry import register_model

except ImportError:
    log.error(
        "Need llmss_eval"
    )
    sys.exit(1)

DEFAULT_IMAGE_TOKEN = "{image}"

@register_model("mmllama")
class _EvalWrapper(lmms):

    def __init__(
        self,
        cfg_path: str,
        *,
        # batch size /device are already in config
        batch_size: int = 32,
        device = None
    ):
        super().__init__()
        self._cfg = OmegaConf.load(cfg_path)
        self._device = utils.get_device(device=self._cfg.device)
        self._dtype = utils.get_dtype(dtype=self._cfg.dtype)
        self._quantizer = config.instantiate(self._cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        self._max_seq_length = self._cfg.max_seq_length
        self._batch_size = self._cfg.batch_size

        self._temperature = self._cfg.temperature
        self._top_k = self._cfg.top_k

        utils.set_seed(seed=self._cfg.seed)

        checkpointer = config.instantiate(self._cfg.checkpointer)

        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=self._cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._image_bind = self._setup_image_bind(model_cfg=self._cfg.image_bind)
        self._clip_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=self._dtype).to(self._device)

        self._tokenizer = config.instantiate(self._cfg.tokenizer)
        self._image_ids = self._tokenizer.encode(START_IMAGE + END_IMAGE, add_eos=False, add_bos=False)

        bos_id = self._tokenizer.tt_model.encode("<|begin_of_text|>", allowed_special="all")[0]
        allowed_id = self._tokenizer.tt_model.encode("<|eot_id|>", allowed_special="all")
        self._disallowed_tokens = list(set(range(bos_id, bos_id + 256)) - set(allowed_id))

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
            model.setup_caches(max_batch_size=self._batch_size, dtype=self._dtype)

        return model

    def _setup_image_bind(
        self,
        model_cfg: DictConfig,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Embed model is initialized with precision {self._dtype}.")

        return model

    def mm_process_prompt(self, prompt):
        return prompt.replace("{image}", f"{START_IMAGE}0{END_IMAGE}")

    def get_image_embed(self, image):
        with torch.no_grad():
            img = self._image_bind.transform_from_pil(image).unsqueeze(0).to(device=self._device, dtype=self._dtype)
            return self._image_bind({ModalityType.VISION: img})[ModalityType.VISION].squeeze(0)

    def get_clip_embed(self, image):
        with torch.no_grad():
            img = self._clip_pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
            img = img.to(device=self._device, dtype=self._dtype)
            return self._clip_pipe.image_encoder(img).image_embeds.squeeze(0)

    def extract_mm_context(self, image, tokens):
        context = {}
        in_image_embed = False
        for idx, tok in enumerate(tokens):
            in_image_embed = in_image_embed and not tok == self._image_ids[1]
            if in_image_embed:
                #tokens[idx] # to support multiple embeds: get the value, match it up with the sample embed
                ib_embed = self.get_image_embed(image)
                clip_embed = self.get_clip_embed(image)
                context[idx] = {"ib_embed": ib_embed, "clip_embed": clip_embed}
            in_image_embed = in_image_embed or tok == self._image_ids[0]
        return context

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, text: str, **kwargs) -> List[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        return self._tokenizer.encode(text=text, add_bos=True, add_eos=False)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self._tokenizer.decode(tokens, truncate_at_eos=True)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    @torch.no_grad()
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float | bool]]:
        return super().loglikelihood(requests)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def multinomial_sample_batch(self, probs):
        q = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / q, dim=-1).to(dtype=torch.int)


    def sample_batch(
        self, logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None
    ) -> torch.Tensor:
        # scale the logits based on temperature
        logits = logits / max(temperature, 1e-5)

        # keep only the top_k logits if this is specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            # select the very last value from the top_k above as the pivot
            pivot = v.select(-1, -1).unsqueeze(-1)
            # set everything smaller than pivot value to inf since these
            # should be pruned
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        # compute the probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # sample the next token
        tokens = self.multinomial_sample_batch(probs)
        return tokens


    def generate_next_token_batch(
        self,
        input_pos: torch.Tensor,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        # x: [B, s]
        # input_pos: [B, s]
        logits = self._model(x, input_pos)

        # logits: [B, s, v] where v is vocab_size
        # for sampling we extract the logits for the
        # last token and convert to shape: [B, v]
        logits = logits[:, -1]

        tokens = self.sample_batch(logits, temperature, top_k)
        tokens = torch.tensor([
            self._tokenizer.eos_id if t in self._disallowed_tokens else t
            for t in tokens
        ]).to(x)
        return tokens


    @torch.inference_mode()
    def batch_generate(
            self,
            prompt: torch.Tensor,
            max_generated_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            eos_id: Optional[int] = None,
            custom_generate_next_token: Optional[Callable] = None,
    ) -> torch.Tensor:

        prompt_length = prompt.shape[1]

        if self._model.max_seq_len < (prompt_length + max_generated_tokens) - 1:
            raise ValueError(
                f"Models maximum seq length {self._model.max_seq_len} should be >= "
                f"{(prompt_length + max_generated_tokens)} - 1"
            )

        if custom_generate_next_token is None:
            custom_generate_next_token = self.generate_next_token_batch

        generated_tokens = [prompt]

        token = self.generate_next_token_batch(
            input_pos=torch.arange(0, prompt_length, device=prompt.device),
            x=prompt,
            temperature=temperature,
            top_k=top_k
        ).clone()

        generated_tokens.append(token.view(-1,1))

        # generation starts at position=prompt_length and continues till
        # we get the requested number of tokens or we hit eos_id
        input_pos = torch.tensor([prompt_length], device=prompt.device)

        for _ in range(max_generated_tokens - 1):
            token = custom_generate_next_token(
                input_pos=input_pos,
                x=token.view(-1,1),
                temperature=temperature,
                top_k=top_k,
            ).clone()

            generated_tokens.append(token.view(-1,1))

            # if eos_id is not None and token == eos_id:
            #     break

            # update the position before we generate the next token
            input_pos += 1

        return torch.cat(generated_tokens, dim=1).tolist()

    def batch_decode(self, generated_tokens):
        res = []
        for seq in generated_tokens:
            res.append(self.tok_decode(seq))
        return res

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.

        re_ords = lmms_utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            contexts = list(contexts)
            contexts = [START_HEADER_ID+"user"+END_HEADER_ID + context for context in contexts]

            fill = self.batch_size - len(contexts)

            for _ in range(fill):
                contexts.append("filler")

            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # Set default values for until and max_new_tokens
            until = [self.tok_decode([self.eot_token_id])]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            for i in range(len(contexts)):
                if visuals is not [] and DEFAULT_IMAGE_TOKEN not in contexts[i]:
                    contexts[i] = f"{DEFAULT_IMAGE_TOKEN}\n{contexts[i]}"

            tokens = [
                torch.tensor(self._tokenizer.tokenize_messages([
                    Message(role="user", content=self.mm_process_prompt(context)),
                    Message(role="assistant", content="")
                ])[0][:-2], dtype=torch.int, device=self._device)
                for context in contexts
            ]
            # tokens = [torch.tensor(self.tok_encode(self.mm_process_prompt(context), add_bos=True, add_eos=False), dtype=torch.int, device=self._device) for context in contexts]
            mm_context = [self.extract_mm_context(visual, token) for token, visual in zip(tokens, visuals)]

            reversed_tokens = [token.flip(dims=(0,)) for token in tokens]
            padded_reversed_tokens = pad_sequence(reversed_tokens, batch_first=True, padding_value=self._tokenizer.pad_id)
            prompt = torch.stack([padded_token.flip(dims=(0,)) for padded_token in padded_reversed_tokens])

            self._model.tok_embeddings.set_context(mm_context)
            self._model.output.set_context(mm_context)

            def custom_generate_next_token(input_pos, x, temperature, top_k):
                self._model.tok_embeddings.set_context([])
                self._model.output.set_context([])
                return self.generate_next_token_batch(input_pos, x, temperature, top_k)

            if self._quantization_mode is not None:
                log.info("Starting compilation to improve generation performance ...")
                custom_generate_next_token = torch.compile(
                    utils.generate_next_token, mode="max-autotune", fullgraph=True
                )
                _ = utils.generate(
                    prompt=prompt,
                    max_generated_tokens=2,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    eos_id=self._tokenizer.eos_id,
                    custom_generate_next_token=custom_generate_next_token,
                )

            max_new_tokens = None
            if "max_new_tokens" in gen_kwargs:
                max_new_tokens = gen_kwargs["max_new_tokens"]
            temperature = None
            if "temperature" in gen_kwargs:
                temperature = gen_kwargs["temperature"]
            top_k = None
            if "top_k" in gen_kwargs:
                top_k = gen_kwargs["top_k"]

            generated_tokens = self.batch_generate(
                prompt=prompt,
                max_generated_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                temperature= temperature if temperature is not None else self._temperature,
                top_k=top_k if top_k is not None else self._top_k,
                eos_id=self._tokenizer.eos_id,
                custom_generate_next_token=custom_generate_next_token,
            )

            generated_tokens = generated_tokens[:self._batch_size-fill]

            generated_tokens = [resp[prompt.shape[1]:] for resp in generated_tokens]
            print(self.batch_decode(generated_tokens))

            filtered_tokens = list(self._tokenizer._get_all_special_tokens_with_ids().values())

            generated_tokens_filtered = list(map(lambda tokens: list(filter(lambda token: token not in filtered_tokens, tokens)), generated_tokens))

            text_outputs = self.batch_decode(generated_tokens_filtered)
            print(text_outputs)
            # print(text_outputs)
            res.extend(text_outputs)

            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res


class MMEvalRecipe():

    def __init__(self, cfg: DictConfig, cfg_path: str) -> None:
        self._cfg = cfg
        self._cfg_path = cfg_path

    def setup(self) -> None:
        self._limit = self._cfg.limit
        self._tasks = list(self._cfg.tasks)
        self._device = utils.get_device(device=self._cfg.device)
        self._model_name = self._cfg.model_name

        utils.set_seed(seed=self._cfg.seed)

    @torch.no_grad()
    def evaluate(self) -> None:
        t1 = time.time()
        # Task initialization API changed between v0.4.1 and 0.4.2
        try:
            lmms_eval.tasks.initialize_tasks()
        except Exception:
            pass

        log.info(f"Running evaluation on {self._tasks} tasks.")

        eleuther_output = simple_evaluate(
            model=self._model_name,
            tasks=self._tasks,
            limit=self._limit,
            model_args=f"cfg_path={self._cfg_path}"
        )

        log.info(f"Eval completed in {time.time() - t1:.02f} seconds.")
        for task, res in eleuther_output["results"].items():
            log.info(f"{task}: {res}")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    args = parse_args()
    config.log_config(recipe_name="MMEvalRecipe", cfg=cfg)
    recipe = MMEvalRecipe(cfg=cfg, cfg_path=args.config)
    recipe.setup()
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
