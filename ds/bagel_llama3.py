from typing import Any, Tuple, List, Mapping
import re

from torch.utils.data import IterableDataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    validate_messages
)
from datasets import load_dataset

import numpy as np

turn_re = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>(.*)")

class BagelLlama3Dataset(IterableDataset):
    def __init__(self, parquet_path, tokenizer, train_on_input=False, len_proportion=0.9, max_seq_len=512, world_size=1, rank=0, **kwargs):
        self._data = load_dataset("parquet", data_files={"train": parquet_path})["train"]
        self._tokenizer = tokenizer
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len
        self._world_size = world_size
        self._index = self._rank = rank
        self._outer_index = 0
        # we need len_proportion because this dataset has samples we want to reject
        # in a distributed setting this causes end-epoch deadlock due to different number of samples
        # per process -> solve by finishing the epoch on a deterministic but reduced number of samples
        self._len = int(len_proportion * len(self._data)) // self._world_size

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._outer_index >= self._len:
            raise StopIteration()

        sample = self._prepare_sample(self._data[self._index])
        self._index += self._world_size
        if sample:
            self._outer_index += 1 # only increment when we actually return a sample
        else:
            sample = next(self)
        return sample

    def _io_to_messages(self, i: str, o: str) -> List[Message]:
        io = i + o
        turns = io.replace("<|begin_of_text|>", "").split("<|eot_id|>")
        messages = []
        for t in turns:
            if not t: continue
            m = turn_re.search(t)
            assert m != None
            role = m.group(1)
            messages.append(Message(
                role=role,
                content=m.group(2),
                masked=(role in ("system", "user") and not self.train_on_input)
            ))
        return messages

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        if sample['source'] in ('comedy-snippets_plain_text', 'cinematika_scenes_plain_text'):
            return None
        messages = self._io_to_messages(sample["input"], sample["output"])
        try:
            validate_messages(messages)
        except ValueError:
            return None

        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        if not self.train_on_input:
            # don't learn that "<|start_header_id|>assistant" always comes after <|eot_id|>
            try:
                first_false_idx = mask.index(False)
                mask[first_false_idx:first_false_idx+3] = [True, True, True]
                mask = mask[:len(tokens)]
            except ValueError:
                pass

        mask = np.array(mask)
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels, {}
