from typing import Dict

from collections import Counter
import datetime
import os
import pickle
from pprint import pprint
import re
import tarfile
import tempfile

from loss import *
from dataset import gen_vec_collate_fn
from utils import get_stats, class_stats_histogram

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb


class Trainer:
    def __init__(self, flags):
        self.flags = flags

    def create_loss(self, model, train_data, dev_data):

        loss_type = self.flags.loss_type

        train_ct_dict = Counter(train_data.labels)
        dev_ct_dict = Counter(dev_data.labels)

        cts = {model.class_map[key]: ct for key, ct in train_ct_dict.items()}

        dev_cts = {model.class_map[key]: ct for key, ct in dev_ct_dict.items()}

        if not set(dev_cts.keys()).issubset(set(cts.keys())):
            print(
                'Warning! ' +
                f'Dev labels not in train set: {set(dev_ct_dict.keys()) - set(train_ct_dict.keys())}'
                + f'Dev labels {set(dev_ct_dict.keys())}')

        def make_ce_loss(*args, **kwargs):
            return nn.CrossEntropyLoss(*args, **kwargs)

        class_freqs = torch.tensor([cts[key] for key in sorted(cts.keys())],
                                   dtype=torch.float)
        class_probs = class_freqs / torch.sum(class_freqs)
        class_inv_freqs = 1. / class_freqs
        class_weights = class_inv_freqs / torch.sum(
            class_inv_freqs) * class_inv_freqs.shape[0]

        cvar_match_res = re.fullmatch(r'(h)?cvar(\_var\_reg)?',
                                      self.flags.loss_type)

        if self.flags.loss_type == 'ce':
            loss = make_ce_loss(reduction='sum')
        # The CVaR loss family can include variance regularization as well as normal loss
        elif self.flags.loss_type == 'hcvar':
            weights = torch.pow(class_probs, self.flags.hcvar_temp)
            weights = weights / torch.sum(weights) * self.flags.hcvar_alpha

            loss = HCVaRLoss(alpha=weights,
                             loss=make_ce_loss(reduction='none'),
                             var_reg=is_var_reg,
                             strat=self.flags.cvar_lambda_strat)
        elif self.flags.loss_type == 'cvar':
            loss = CVaRLoss(classes=len(cts),
                            alpha=self.flags.cvar_alpha,
                            loss=make_ce_loss(reduction='none'),
                            var_reg=is_var_reg,
                            strat=self.flags.cvar_lambda_strat)
        elif loss_type == 'class_weighted_ce':
            loss = make_ce_loss(reduction='sum',
                                weight=torch.tensor(class_weights).float())
        else:
            raise KeyError(f'Loss key {loss} does not exist')
        return loss

    def create_lr_scheduler(self, optimizer, loss_fn):
        decay_sched = None
        if self.flags.lr_decay_type == 'geometric':
            min_factor = self.flags.lr_decay_min / self.flags.lr
            rate = np.power(min_factor, 1 / self.flags.epochs)
            decay_sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: np.power(rate, x))
        elif self.flags.lr_decay_type == 'linear':
            min_factor = self.flags.lr_decay_min / self.flags.lr
            factor_range = 1 - min_factor
            decay_sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: (1 - x / self.flags.epochs) * factor_range
                + min_factor)
        elif self.flags.lr_decay_type == 'plateau':
            decay_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.flags.lr_decay_factor,
                patience=self.flags.lr_decay_patience,
                threshold=self.flags.lr_decay_threshold,
                cooldown=self.flags.lr_decay_cooldown,
                min_lr=self.flags.lr_decay_min)

        return decay_sched

    def _get_saved_states(self, save_dir):
        return [
            filename for filename in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, filename))
            and re.fullmatch(r'model\_(\d+)\.state', filename)
        ]

    def _update_metric(self, metric_name, epoch, value):
        if metric_name not in self.best_metrics or value > self.best_metrics[
                metric_name][1]:
            self.best_metrics[metric_name] = (epoch, value)
            return True
        return False

    def _save_best_metric(self, save_dir: str, model, epoch: int,
                          stats_dict: Dict[str, float], class_stats):
        did_update = False
        for metric_name in stats_dict:
            did_update = self._update_metric(metric_name, epoch,
                                             stats_dict[metric_name])

        if did_update:
            for metric, (best_epoch, val) in self.best_metrics.items():
                if best_epoch == epoch:
                    state_path = os.path.join(save_dir,
                                              f'model_{metric}.state')
                    torch.save(model.state_dict(), state_path)
                    wandb.save(state_path)
                wandb.run.summary[f'Best dev {metric} epoch'] = best_epoch
                wandb.run.summary[f'Best dev {metric}'] = val

        self._update_metric('class 0 accuracy',
                            class_stats[model.idx_to_class_map[0]]['recall'],
                            epoch)
        self._update_metric('class 1 accuracy',
                            class_stats[model.idx_to_class_map[1]]['recall'],
                            epoch)

    def _epoch_closure(self, train_data, model, epoch, loss_fn, optimizer,
                       device):
        epoch_loss = 0
        batch_ct = len(train_data) // self.flags.batch_size
        if isinstance(loss_fn, HCVaRLoss):
            classwise_losses = {}
        for batch_idx, batch in enumerate(
                DataLoader(train_data,
                           batch_size=self.flags.batch_size,
                           collate_fn=gen_vec_collate_fn(model.class_map))):
            features_batch, label_batch = [item.to(device) for item in batch]
            logit_batch = model.logits(features_batch)
            loss = loss_fn(logit_batch, label_batch)
            if isinstance(loss_fn, HCVaRLoss):
                for key in loss_fn.prev_loss:
                    classwise_losses[key] = classwise_losses.get(
                        key, 0) + loss_fn.prev_loss[key]

            optimizer.zero_grad()
            (loss / logit_batch.shape[0]).backward()
            optimizer.step()
            # Logging / data collection
            epoch_loss += loss.item() / len(train_data)
        if isinstance(loss_fn, HCVaRLoss):
            wandb.log({
                'epoch': epoch,
                **{
                    f'Class {key} loss': loss
                    for key, loss in classwise_losses.items()
                }, 'CVaR Lambda': loss_fn.threshold.item()
            })
        return epoch_loss

    def _make_class_dist_plot(self, dataset, **kwargs):
        fig = plt.figure()
        ax = plt.gca()
        label_cts = Counter(dataset.labels)
        min_label, min_ct = min(list(label_cts.items()), key=lambda x: x[1])
        max_label, max_ct = max(list(label_cts.items()), key=lambda x: x[1])
        y, _, _ = ax.hist(list(label_cts.values()), **kwargs)
        ax.set_title(
            f'Min label: {min_label}, Min ct: {min_ct}\nMax label: {max_label}, Max ct: {max_ct}'
        )
        min_line = mlines.Line2D([min_ct, min_ct], [y.min(), y.max()],
                                 color='red',
                                 linestyle='--')
        ax.add_line(min_line)
        max_line = mlines.Line2D([max_ct, max_ct], [y.min(), y.max()],
                                 color='red',
                                 linestyle='--')
        ax.add_line(max_line)
        return fig

    def train(self,
              train_data,
              dev_data,
              model,
              train_tmp_dir,
              device=torch.device('cpu')):
        with open(os.path.join(train_tmp_dir, f'model.pkl'), 'wb') as out_f:
            pickle.dump(model.save_params(), out_f)
            wandb.save(os.path.join(train_tmp_dir, 'model.pkl'))

        print(set(dev_data.labels).difference(set(train_data.labels)))
        model = model.to(device)

        self.best_metrics: Dict[str, Tuple[int, float]] = {}

        loss_fn = self.create_loss(model, train_data, dev_data).to(device)

        tot_params = [{'params': list(model.parameters())}]

        if self.flags.loss_type in ['hcvar', 'cvar']:
            tot_params += [{
                'params': loss_fn.threshold,
                'lr': 10 * self.flags.lr,
                'momentum': 0.
            }]
        if self.flags.optimizer == 'SGD':
            optimizer = torch.optim.SGD(tot_params,
                                        lr=self.flags.lr,
                                        momentum=self.flags.momentum)
        else:
            optimizer = torch.optim.Adam(tot_params, lr=self.flags.lr)
        decay_sched = self.create_lr_scheduler(optimizer, loss_fn)

        epochs = self.flags.epochs
        train_plot = self._make_class_dist_plot(train_data)
        wandb.log({'Train class dist': wandb.Image(train_plot)})

        for epoch in range(epochs):
            start_time = datetime.datetime.now()
            model.train()

            epoch_loss = self._epoch_closure(train_data, model, epoch, loss_fn,
                                             optimizer, device)

            epoch_time = datetime.datetime.now() - start_time

            # Decay learning rate

            log_entry = {
                'epoch': epoch,
                'Train loss/sample': epoch_loss,
                'Time': epoch_time.total_seconds(),
                'Samples/sec': len(train_data) / epoch_time.total_seconds(),
            }
            if self.flags.lr_decay_type:
                current_lr = optimizer.param_groups[0]['lr']
                log_entry['Current LR'] = current_lr
                if self.flags.lr_decay_type == 'plateau':
                    decay_sched.step(epoch_loss)
                else:
                    decay_sched.step()

            if ((epoch + 1) % self.flags.dev_interval == 0
                    or epoch == epochs - 1):
                model.eval()
                dev_preds = model.predict(dev_data, device=device)
                global_stats, class_stats = get_stats(
                    dev_preds,
                    dev_data.labels,
                    all_labels=list(model.class_map.keys()))

                for metric_name in global_stats:
                    self._save_best_metric(train_tmp_dir,
                                           model,
                                           epoch,
                                           stats_dict=global_stats,
                                           class_stats=class_stats)

                worst_risk_class, worst_risk_entry = max(
                    list(class_stats.items()),
                    key=lambda x: 1 - x[1]['recall'])
                log_entry = {
                    **{
                        f'Dev {metric}': val
                        for metric, val in global_stats.items()
                    },
                    **log_entry,
                    **{
                        f'Class {idx} dev risk': 1 - class_stats[model.idx_to_class_map[idx]]['recall']
                        for idx in range(2)
                    }, 'Dev risk histogram': wandb.Image(
                        class_stats_histogram(class_stats,
                                              lambda x: 1 - x['recall'],
                                              'Risk',
                                              range=(0, 1),
                                              bins=np.arange(11) * 0.1)),
                    'Dev worst class risk': 1 - worst_risk_entry['recall'],
                    'Dev worst class risk label': worst_risk_class
                }
            pprint(log_entry)
            wandb.log(log_entry)
        # At the end save best metrics over the course of the run
        wandb.log({
            'Dev class histogram':
            wandb.Image(
                class_stats_histogram(class_stats,
                                      lambda x: x['true'],
                                      'Frequency',
                                      cmp_fn=lambda x: -x,
                                      bins=20))
        })
        pprint('Best metrics: ')
        pprint(self.best_metrics)
