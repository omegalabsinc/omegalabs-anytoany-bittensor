from typing import Any, Tuple, List, Mapping
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    validate_messages,
)
import numpy as np

from models.tokenizer import START_IMAGE, END_IMAGE
from ds.embed_caption_dataset import EmbedCaptionDataset


image_caption_templates = [{
    "user": "Image:\n{image}\nCaption the previous image.",
    "assistant": "{caption}"
}, {
    "user": "Write a response that appropriately captions the image.\nImage:\n{image}",
    "assistant": "Caption:\n{caption}"
}]#, {
    # "user": "Create an image that matches the caption.\nCaption:\n{caption}",
    # "assistant": "Image:\n{image}"
# }]


class CaptionInstructDataset(IterableDataset):
    def __init__(self, dataset_path, tokenizer, train_on_input=False, max_seq_len=512, world_size=1, rank=0, perception_tokens=1):
        self._data = EmbedCaptionDataset(dataset_path, world_size, rank)
        self._len = len(self._data) * len(image_caption_templates)
        self._data = iter(self._data)
        self._tokenizer = tokenizer
        self._image_ids = self._tokenizer.encode(START_IMAGE + END_IMAGE, add_eos=False, add_bos=False)
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len
        self._idx = 0
        self._perception_tokens = ("0 " * perception_tokens)[:perception_tokens]

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        template_idx = self._idx % len(image_caption_templates)
        if template_idx == 0:
            self._sample = next(self._data)
        self._idx += 1
        return self._prepare_sample(
            image_caption_templates[template_idx],
            self._sample
        )

    def _format_template(self, template: str, sample: Mapping[str, Any]):
        return template.replace(
            "{caption}", sample["caption"]
        ).replace(
            "{image}", START_IMAGE + self._perception_tokens + END_IMAGE
        )

    def _prepare_sample(self, template: str, sample: Mapping[str, Any]):
        prompt = self._format_template(template["user"], sample)
        response = self._format_template(template["assistant"], sample)
        messages = [
            Message(role="user", content=prompt, masked=(not self.train_on_input)),
            Message(role="assistant", content=response),
        ]

        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        if not self.train_on_input:
            # don't learn that "<|start_header_id|>" always comes after <|eot_id|>
            try:
                first_false_idx = mask.index(False) # <|start_header_id|>
                mask[first_false_idx:first_false_idx+2] = [True, True]
            except ValueError:
                pass

        # ensure within-image tags tokens are masked out
        mask = np.array(mask)
        image_embed_mask_ids = []
        context = {}
        in_image_embed = False
        for idx, tok in enumerate(tokens):
            is_image_begin = tok == self._image_ids[0]
            is_image_end = tok == self._image_ids[1]
            in_image_embed = in_image_embed and not is_image_end
            if is_image_begin:
                #context[idx+1] = {k: sample[k] for k in ["ib_embed", "clip_embed"]}
                context[idx+1] = {k: sample[k] for k in ["ib_embed"]}
            if in_image_embed:
                image_embed_mask_ids.append(idx)
            in_image_embed = in_image_embed or is_image_begin
        if image_embed_mask_ids:
            mask[image_embed_mask_ids] = True

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels, context

