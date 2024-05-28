from typing import Any, Tuple, List, Mapping
from pathlib import Path
from functools import reduce
from operator import add

import torch
from torch.utils.data import IterableDataset
from torchtune.config import instantiate


class RoundRobinDataset(IterableDataset):
    def __init__(self, datasets, tokenizer, world_size=1, rank=0, perception_tokens=1):
        self._ds_cfg = datasets
        self._tokenizer = tokenizer
        self._world_size = world_size
        self._rank = rank
        self._perception_tokens = perception_tokens
        self._reset_datasets()
        self._len = reduce(add, self._ds_lengths)

    def _reset_datasets(self):
        self._datasets = [
            instantiate(cfg, tokenizer=self._tokenizer, world_size=self._world_size, rank=self._rank, perception_tokens=self._perception_tokens)
            for cfg in self._ds_cfg
        ]
        self._ds_indexes = [0 for d in self._datasets]
        self._ds_lengths = [len(ds) for ds in self._datasets]

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if not self._datasets:
            self._reset_datasets()
            raise StopIteration()

        # take next sample from ds with lowest progression
        _, next_ds_idx = sorted([
                (ds_idx / ds_len, i)
                for i, (ds_idx, ds_len)
                in enumerate(zip(self._ds_indexes, self._ds_lengths))
        ])[0]

        try:
            sample = next(self._datasets[next_ds_idx])
            self._ds_indexes[next_ds_idx] += 1
            return sample
        except StopIteration:
            del self._datasets[next_ds_idx], self._ds_indexes[next_ds_idx], self._ds_lengths[next_ds_idx]
            return next(self)

