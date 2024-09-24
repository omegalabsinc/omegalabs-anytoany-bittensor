from typing import List
import warnings

import torch
from torch import nn, Tensor
from torchvision import transforms

from torchtune.models.llama3 import lora_llama3_8b, llama3_8b
from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear
from torchtune.modules import TransformerDecoder

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from imagebind.models import imagebind_model
    from models.imagebind_wrapper import get_imagebind_v2, V2_PATH

IMAGEBIND_DIM = 1024
CLIP_DIM = 768


class MMEmbedding(nn.Embedding):
    def __init__(self, e, perception_tokens=1, use_clip=False):
        super().__init__(
            num_embeddings=e.num_embeddings,
            embedding_dim=e.embedding_dim,
            padding_idx=e.padding_idx,
            max_norm=e.max_norm,
            norm_type=e.norm_type,
            scale_grad_by_freq=e.scale_grad_by_freq,
            sparse=e.sparse,
        )
        self._perception_tokens = perception_tokens
        self._context = []
        self._use_clip = use_clip

        dim_in = IMAGEBIND_DIM + (CLIP_DIM if use_clip else 0)
        dim_out = e.embedding_dim * perception_tokens

        self.proj_to_llama = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.LayerNorm(dim_out),
            nn.Linear(dim_out, dim_out),
        )

    def set_context(self, context):
        self._context = context

    def forward(self, input: Tensor) -> Tensor:
        r = super().forward(input)
        # self._context is first indexed by batch idx
        for b, context_dict in enumerate(self._context):
            # then by sequence idx
            for s, embed in context_dict.items():
                # and then must be transformed from imagebind dim -> llama3 dim
                if self._use_clip:
                    llama_embed = self.proj_to_llama(torch.cat([embed["ib_embed"], embed["clip_embed"]]))
                else:
                    llama_embed = self.proj_to_llama(torch.cat([embed["ib_embed"]]))
                r[b, s:s+self._perception_tokens] = llama_embed.view(self._perception_tokens, -1)
        return r


class MMLinear(nn.Linear):
    def __init__(self, o):
        super().__init__(
            in_features=o.in_features,
            out_features=o.out_features,
            bias=(o.bias != None)
        )
        self._context = []

        dim_out = CLIP_DIM
        dim_in = o.in_features
        self.proj_from_llama = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.LayerNorm(dim_out),
            nn.Linear(dim_out, dim_out),
        )

    def set_context(self, context):
        self._context = context

    def forward(self, input_bsd: Tensor) -> Tensor:
        # self._context has the indexes of image llama tokens: process these with proj_from_llama
        self._clip_projections = []
        # # self._context is first indexed by batch idx
        # for b, context_dict in enumerate(self._context):
        #     # then by sequence idx
        #     for s, embed in context_dict.items():
        #         # and then must be transformed from llama3 dim -> clip dim
        #         self._clip_projections.append((
        #             self.proj_from_llama(input_bsd[b, s]),
        #             (embed["clip_embed"] if "clip_embed" in embed else None) # terrible
        #         ))
        r = super().forward(input_bsd)
        return r



def lora_mmllama3_8b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
    perception_tokens: int = 2,
    use_clip: bool = False
) -> TransformerDecoder:
    llama3 = lora_llama3_8b(
        lora_attn_modules,
        apply_lora_to_mlp,
        apply_lora_to_output,
        lora_rank,
        lora_alpha,
        quantize_base,
    )
    llama3.tok_embeddings = MMEmbedding(llama3.tok_embeddings, perception_tokens, use_clip)
    llama3.output = MMLinear(llama3.output)
    return llama3


def mmllama3_8b(
    perception_tokens: int = 2,
    use_clip: bool = False
) -> TransformerDecoder:
    llama3 = llama3_8b()
    llama3.tok_embeddings = MMEmbedding(llama3.tok_embeddings, perception_tokens, use_clip)
    llama3.output = MMLinear(llama3.output)
    return llama3


def imagebind_huge(use_v2: bool=True):
    if use_v2:
        imagebind = get_imagebind_v2(path=V2_PATH).imagebind_huge(pretrained=True)
    else:
        imagebind = imagebind_model.imagebind_huge(pretrained=True)
    imagebind.transform_from_pil = transforms.Compose([
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    return imagebind

