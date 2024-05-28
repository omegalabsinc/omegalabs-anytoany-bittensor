from pathlib import Path
from itertools import chain
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


class SingleEmbedCaptionDataset(IterableDataset):
    def __init__(self, dataset_path, world_size=1, rank=0):
        self.dataset_path = dataset_path
        self._world_size = world_size
        self._index = self._rank = rank
        self._len = None

    def __len__(self):
        if self._len is None:
            self._len = (pd.read_parquet(self.dataset_path).shape[0] - self._rank) // self._world_size
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._rank:
            data = pd.read_parquet(self.dataset_path)
            self.ib_embeds = torch.from_numpy(np.stack(data.ib_embed.to_numpy()))
            self.clip_embeds = torch.from_numpy(np.stack(data.clip_embed.to_numpy()))
            self.captions = data.caption.tolist()
            assert len(self.ib_embeds) == len(self.clip_embeds) == len(self.captions)

        if self._index >= self.ib_embeds.size(0):
            del self.ib_embeds, self.clip_embeds, self.captions
            raise StopIteration

        item = {
            "caption": self.captions[self._index],
            "ib_embed": self.ib_embeds[self._index],
            "clip_embed": self.clip_embeds[self._index],
        }
        self._index += self._world_size
        return item


class EmbedCaptionDataset(IterableDataset):
    def __init__(self, dataset_path, world_size=1, rank=0):
        self.dataset_paths = sorted(list(Path('.').glob(dataset_path)))
        self._world_size = world_size
        self._rank = rank
        self._iter = chain.from_iterable(
            iter(SingleEmbedCaptionDataset(path, world_size=self._world_size, rank=self._rank))
            for path in self.dataset_paths
        )
        self._len = None

    def __len__(self):
        if self._len is None:
            self._len = reduce(
                add, [
                    len(SingleEmbedCaptionDataset(path, world_size=self._world_size, rank=self._rank))
                    for path in self.dataset_paths
                ]
            )
        return self._len

    def __iter__(self):
        return self._iter


if __name__ == "__main__":
    d = EmbedCaptionDataset(
        "ds/sam_llava/output.parquet",
    )
    for i, item in enumerate(d):
        if i % 1024 == 0: print(i)
