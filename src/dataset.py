from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import random
from toolz import groupby, interleave

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info


class VecDataset(Dataset):
    def __init__(self, features, labels):

        assert features.shape[0] == len(labels)
        super(Dataset, self).__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

    def __len__(self):
        return self.features.shape[0]


class SentDataset(Dataset):
    def __init__(self, tokens: List[List[str]], labels: List[str]):
        assert (len(tokens) == len(labels))
        self.tokens, self.labels = tokens, labels
        self.labels = labels
        self.n = len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index]

    def __len__(self):
        return self.n


class SmoteDataset(SentDataset):
    def __init__(self, tokens: List[List[str]], labels: List[str]):
        super(SmoteDataset, self).__init__(tokens, labels)

        groups = groupby(lambda x: x[1], zip(tokens, labels))
        self.num_labels = len(groups.keys())
        self.sample_groups = {
            key: [example[0] for example in examples]
            for key, examples in sorted(groups.items(), key=lambda x: x[0])
        }

    def get_random_class_sample(self, label):
        features = random.choice(self.sample_groups[label])
        return features

    @staticmethod
    def from_SentDataset(sent_dataset):
        return SmoteDataset(sent_dataset.tokens, sent_dataset.labels)


def make_offsets(tokens):
    offsets = [len(token_list) for token_list in tokens]
    tok_ct = sum(offsets)
    for idx, token_list_len in enumerate(offsets):
        if idx > 0:
            offsets[idx] += offsets[idx - 1]
    assert offsets[-1] == tok_ct
    return [0] + offsets[:-1]


def gen_vec_collate_fn(class_map):

    if class_map is not None:

        def vec_collate_fn(batch):
            features, labels = zip(*batch)
            return torch.stack(features), torch.tensor(
                [class_map[label] for label in labels], dtype=torch.long)

        return vec_collate_fn
    else:

        def vec_collate_fn(batch):
            features, labels = zip(*batch)
            return torch.stack(features), torch.tensor(labels,
                                                       dtype=torch.long)

        return vec_collate_fn


TokenMap = Dict[Union[str, int], int]
