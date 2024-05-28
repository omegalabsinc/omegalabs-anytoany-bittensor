from typing import Any, Tuple, List, Mapping
from pathlib import Path
from functools import reduce
from operator import add
from sortedcontainers import SortedList

import torch
from torch.utils.data import IterableDataset
from torchtune.config import instantiate
from torchtune import utils


class EvenBatcher(IterableDataset):
    def __init__(self, dataset, batch_size, tokenizer, ignore_index, world_size=1, rank=0, buffer_size=100, len_steps=5, perception_tokens=1):
        self._dataset = instantiate(
            dataset,
            tokenizer=tokenizer,
            world_size=world_size,
            rank=rank,
            perception_tokens=perception_tokens,
        )
        self._pad_id = tokenizer.pad_id
        self._ignore_index = ignore_index
        self._batch_size = batch_size
        self._sample_buffer = SortedList(key=lambda l_s: l_s[0]) # only sort on len, not on tensors
        self._buffer_size = buffer_size
        self._len_steps = len_steps
        self._index = 0
        self._epoch_complete = False

    def __len__(self):
        return len(self._dataset) // self._batch_size

    def __iter__(self):
        return self

    def _fill_buffer(self):
        try:
            while not self._epoch_complete and len(self._sample_buffer) < self._buffer_size:
                sample = next(self._dataset)
                tokens, labels, context = sample
                self._sample_buffer.add((len(tokens), sample))
        except StopIteration:
            self._epoch_complete = True

    def __next__(self):
        self._fill_buffer()

        if len(self._sample_buffer) < self._batch_size or self._index >= len(self):
            self._epoch_complete = False
            raise StopIteration()

        len_proportion = (self._index % self._len_steps) / self._len_steps
        batch_idx = int(len_proportion * (len(self._sample_buffer) - self._batch_size))
        assert batch_idx <= (len(self._sample_buffer) - self._batch_size)
        self._index += 1

        return self._padded_collate([
            self._sample_buffer.pop(batch_idx)[-1] # discard length
            for _ in range(self._batch_size)
        ])

    def _padded_collate(self, batch_list):
        batch, context_batch = [], []
        for input_ids, labels, context in batch_list:
            batch.append((input_ids, labels))
            context_batch.append(context)
        return utils.padded_collate(batch, self._pad_id, self._ignore_index), context_batch

