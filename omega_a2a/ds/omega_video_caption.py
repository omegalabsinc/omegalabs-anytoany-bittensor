from typing import Any, Tuple, List, Mapping

import torch
from torch.utils.data import IterableDataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    validate_messages,
)
import numpy as np
from datasets import load_dataset

from models.tokenizer import START_VIDEO, END_VIDEO


caption_templates = [{
    "user": "Video:\n{video}\nCaption the previous video.",
    "assistant": "{caption}"
}, {
    "user": "Write a response that appropriately captions the video.\nImage:\n{video}",
    "assistant": "Caption:\n{caption}"
}]


class OmegaVideoCaptionDataset(IterableDataset):
    def __init__(self, length, tokenizer, train_on_input=False, max_seq_len=512, world_size=1, rank=0, perception_tokens=1):
        self._data = load_dataset("omegalabsinc/omega-multimodal", streaming=True)
        self._len = length // world_size
        self._data = iter(self._data["train"].take(length + rank))
        self._tokenizer = tokenizer
        self._image_ids = self._tokenizer.encode(START_VIDEO + END_VIDEO, add_eos=False, add_bos=False)
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len
        self._world_size = world_size
        self._index = self._rank = rank
        self._perception_tokens = ("0 " * perception_tokens)[:perception_tokens]
        for _ in range(self._rank): self._sample = next(self._data)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if (self._index // self._world_size) >= self._len:
            raise StopIteration()

        template_index = self._index % len(caption_templates)
        if template_index == 0:
            for _ in range(self._world_size): self._sample = next(self._data)

        sample = self._prepare_sample(
            caption_templates[template_index],
            self._sample
        )
        self._index += self._world_size
        return sample

    def _format_template(self, template: str, sample: Mapping[str, Any]):
        return template.replace(
            "{caption}", sample["description"]
        ).replace(
            "{video}", START_VIDEO + self._perception_tokens + END_VIDEO
        )

    def _prepare_sample(self, template: str, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
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
                context[idx+1] = {"ib_embed": torch.tensor(sample["video_embed"])}
            if in_image_embed:
                image_embed_mask_ids.append(idx)
            in_image_embed = in_image_embed or is_image_begin
        if image_embed_mask_ids:
            mask[image_embed_mask_ids] = True

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels, context
