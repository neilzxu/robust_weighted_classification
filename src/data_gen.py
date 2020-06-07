import csv
import random

import preprocess
from dataset import VecDataset

from absl import app, flags
import numpy as np
from scipy.stats import bernoulli
import torch
from torch.utils.data import TensorDataset
import wandb


def power_data_gen(FLAGS):
    if FLAGS.synth_seed is not None:
        np.random.seed(FLAGS.synth_seed)
        random.seed(FLAGS.synth_seed)
    datasets = []
    assert FLAGS.power_one_prob <= 1
    assert 0 < FLAGS.power_scalar and FLAGS.power_scalar <= 1.
    pwr = FLAGS.power_scalar / (FLAGS.power_one_prob) - 1
    for ct in [FLAGS.synth_train_ct, FLAGS.synth_dev_ct, FLAGS.synth_test_ct]:
        x = np.random.uniform(0, 1, size=(ct, FLAGS.synth_dim))
        y = [
            int(x) for x in list(
                bernoulli.rvs(FLAGS.power_scalar * np.power(x[:, 0], pwr)))
        ]
        datasets.append(VecDataset(torch.tensor(x, dtype=torch.float), y))

    print(f'ERM threshold at x={np.power(0.5 / FLAGS.power_scalar, 1 / pwr)}')

    return datasets


_SYNTH_MAP = {'power': power_data_gen}


def synth_gen(synth_mode, FLAGS):
    return _SYNTH_MAP[synth_mode](FLAGS)
