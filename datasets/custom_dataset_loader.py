import os
import json
import numpy as np
import mlx.core as mx
import random


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.indices = list(range(len(dataset)))
        self._index = 0

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
            self.current_index = 0
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        batch = [self.dataset[i] for i in batch_indices]
        if self.collate_fn:
            return self.collate_fn(batch)
        self._index += 1
        return batch