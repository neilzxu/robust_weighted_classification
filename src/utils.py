from typing import List
from collections import Counter
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
from torch.utils.data import TensorDataset

# optdigits, image, covtype


def make_class_map(datasets):
    def get_labels(dataset):
        return dataset.labels

    class_ct = Counter(
        [label for dataset in datasets for label in get_labels(dataset)])

    return {
        label: idx
        for idx, (label, _) in enumerate(
            sorted(list(class_ct.items()), key=lambda x: -x[1]))
    }


def get_stats(true_labels: List[str],
              predicted_labels: List[str],
              all_labels=None):
    '''Calculates class wise stats given true and predicted labels'''
    if all_labels is None:
        all_labels = set(true_labels + predicted_labels)
    true_ct = Counter(true_labels)
    pred_ct = Counter(predicted_labels)
    correct_ct = Counter(list(zip(true_labels, predicted_labels)))

    class_stats = {
        label: {
            'correct': correct_ct.get((label, label), 0),
            'predicted': pred_ct.get(label, 0),
            'true': true_ct.get(label, 0)
        }
        for label in all_labels
    }

    for label, label_stats in class_stats.items():
        assert label_stats['correct'] <= label_stats[
            'predicted'] and label_stats['correct'] <= label_stats[
                'true'], f'Label: {label}, Stats: {label_stats}'

    stat_fns = [
        ('precision', lambda stats: 0 if stats['predicted'] == 0 else stats[
            'correct'] / stats['predicted']),
        ('recall', lambda stats: 0
         if stats['true'] == 0 else stats['correct'] / stats['true']),
        ('f1',
         lambda stats: 0 if stats['correct'] == 0 else 2 * stats['precision'] *
         stats['recall'] / (stats['precision'] + stats['recall']))
    ]
    for label in all_labels:
        for stat_name, stat_fn in stat_fns:
            class_stats[label][stat_name] = stat_fn(class_stats[label])

    global_stats = {
        'accuracy':
        sum(entry['correct']
            for entry in class_stats.values()) / len(true_labels),
        'macro_precision':
        float(np.mean([entry['precision'] for entry in class_stats.values()])),
        'macro_recall':
        float(np.mean([entry['recall'] for entry in class_stats.values()])),
        'macro_f1':
        float(np.mean([entry['f1'] for entry in class_stats.values()])),
    }
    return global_stats, class_stats


def class_stats_histogram(class_stats,
                          stat_fn,
                          stat_name,
                          cmp_fn=lambda x: x,
                          **kwargs):
    stat_values = [(key, stat_fn(value)) for key, value in class_stats.items()]
    max_stat_key, max_stat_value = max(stat_values, key=lambda x: cmp_fn(x[1]))
    fig = plt.figure()
    ax = plt.gca()
    y, _, _ = ax.hist([value for _, value in stat_values], **kwargs)
    best_line = mlines.Line2D([max_stat_value, max_stat_value],
                              [y.min(), y.max()],
                              color='red',
                              linestyle='--')
    ax.add_line(best_line)
    ax.set_title(
        f'Stat name: {stat_name}, Worst stat key: {max_stat_key}, Worst stat value: {max_stat_value:.5f}'
    )
    return fig
