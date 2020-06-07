import gzip
import os
import re
import random

from dataset import VecDataset, SmoteDataset

import numpy as np
from toolz.sandbox.core import unzip
import torch

PREPROC_REG = {}


def register_preproc(fn):
    PREPROC_REG[fn.__name__.replace('_', '-')] = fn
    return fn


def make_categorical_transform(feature_set):
    feature_map = {
        feature: idx
        for idx, feature in enumerate(sorted(list(feature_set)))
    }

    def categorical_transform(feature):
        return feature_map[feature]

    return categorical_transform


def _all_numerical(text, label_idx, pattern=r','):
    features = list(
        zip(*[
            re.split(pattern, point.strip())
            for point in text.strip().split('\n')
        ]))
    if label_idx < 0:
        label_idx = len(features) + label_idx
    labels = [x for x in features[label_idx]]
    actual_features = torch.stack([
        torch.tensor([float(x) for x in features[idx]], dtype=torch.float)
        for idx in range(len(features)) if idx != label_idx
    ]).permute(1, 0)
    return VecDataset(actual_features, labels)


@register_preproc
def covtype(text):
    return _all_numerical(text, -1)


class Preprocessor:
    def __init__(self, mode):
        self.mode = mode

    def preprocess_file(self, file_path, out_path=None):
        with open(file_path) as in_f:
            text = in_f.read()
        return PREPROC_REG[self.mode](text)

    def preprocess_text(self, text):
        return PREPROC_REG[self.mode](text)
