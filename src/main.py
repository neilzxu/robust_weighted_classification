from typing import List, Union

import gzip
import os
import json
import pickle
from pprint import pprint
import tarfile
import tempfile

import args
from data_gen import synth_gen
from dataset import SmoteDataset
from model import *
from preprocess import Preprocessor
from trainer import Trainer
from utils import get_stats, make_class_map, class_stats_histogram

from absl import app, flags
import numpy as np
import torch
import wandb
import yaml
FLAGS = flags.FLAGS


def get_device(cuda, cuda_device):
    if cuda:
        torch.cuda.set_device(cuda_device)
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def run_test(model, test_data, device):
    if isinstance(model, VecModel):
        print(f'Model weights: {model.w.weight}')
        print(f'Model biases: {model.w.bias}')

    model.to(device)
    predictions = model.predict(test_data, device=device)
    stats, class_stats = get_stats(test_data.labels,
                                   predictions,
                                   all_labels=list(model.class_map.keys()))

    normal_metrics = {
        f'Test {metric}': value
        for metric, value in stats.items()
    }
    class_risk_metrics = {
        f'Test class {idx} accuracy':
        class_stats[model.idx_to_class_map[idx]]['recall']
        for idx in range(2)
    }
    worst_risk_class, worst_risk_entry = max(list(class_stats.items()),
                                             key=lambda x: 1 - x[1]['recall'])

    entry = {
        **normal_metrics,
        **class_risk_metrics, 'Test risk histogram':
        wandb.Image(
            class_stats_histogram(class_stats,
                                  lambda x: 1 - x['recall'],
                                  'Risk',
                                  bins=np.arange(11) * 0.1,
                                  range=(0, 1))),
        'Test class histogram':
        wandb.Image(
            class_stats_histogram(class_stats,
                                  lambda x: x['true'],
                                  'Frequency',
                                  cmp_fn=lambda x: -x,
                                  bins=10)),
        'Test worst class risk':
        1 - worst_risk_entry['recall'],
        'Test worst class risk label':
        worst_risk_class
    }
    wandb.log(entry)
    pprint(entry)

    with open(os.path.join(wandb.run.dir, 'test_result.json'), 'w') as out_f:
        out_f.write(json.dumps(class_stats))


def main(argv):

    argdict = {
        flag.name: flag.value
        for flag in FLAGS.flags_by_module_dict()['args']
    }
    pprint(argdict)
    argdict['tags'] = argdict['tags'].split(',')
    wandb.init(project='extreme-classification',
               name=argdict['name'],
               tags=argdict['tags'])
    wandb.config.update(argdict)

    device = get_device(FLAGS.cuda, FLAGS.cuda_device)

    if FLAGS.mode == 'train':
        if FLAGS.preprocess_mode == 'synth':
            train_data, dev_data, test_data = synth_gen(
                FLAGS.synth_mode, FLAGS)
            class_map = {
                idx: idx
                for idx in set(train_data.labels + dev_data.labels)
            }

        else:
            preprocessor = Preprocessor(FLAGS.preprocess_mode)
            train_data = preprocessor.preprocess_file(FLAGS.train_path)
            dev_data = preprocessor.preprocess_file(FLAGS.dev_path)

        class_map = {
            label: idx
            for idx, label in enumerate(
                sorted(set(train_data.labels + dev_data.labels)))
        }
        model = VecModel(train_data[0][0].shape[0],
                         class_map,
                         None,
                         enable_binary=FLAGS.enable_binary)
        model.init_norms(train_data)

        trainer = Trainer(FLAGS)
        train_tmp_dir = tempfile.TemporaryDirectory()
        os.chdir(train_tmp_dir.name)
        trainer.train(train_data,
                      dev_data,
                      model,
                      device=device,
                      train_tmp_dir='.')

        if FLAGS.model_metric is not None:
            model.load_state_dict(
                torch.load(f'model_{FLAGS.model_metric}.state'))
        if FLAGS.preprocess_mode == 'synth':
            run_test(model, test_data, device)
        elif FLAGS.test_path is not None:
            test_data = preprocessor.preprocess_file(FLAGS.test_path)
            run_test(model, test_data, device)

    else:  # test mode
        test_data = preprocessor.preprocess_file(FLAGS.test_path)
        test_tmp_dir = tempfile.TemporaryDirectory()
        wandb.restore('model.pkl',
                      run_path=FLAGS.model_run_path,
                      root=test_tmp_dir.name)
        pickle_f_bytes = open(os.path.join(test_tmp_dir.name, 'model.pkl'),
                              'rb').read()
        params = pickle.loads(pickle_f_bytes)
        model = VecModel(*params)
        state_name = f'model_{FLAGS.model_metric}.state'
        wandb.restore(state_name,
                      run_path=FLAGS.model_run_path,
                      root=test_tmp_dir.name)
        model.load_state_dict(
            torch.load(os.path.join(test_tmp_dir.name, state_name)))
        run_test(model, test_data, device)


if __name__ == '__main__':
    app.run(main)
