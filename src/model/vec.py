from dataset import gen_vec_collate_fn

import torch
from torch import nn
from torch.utils.data import DataLoader


class VecModel(nn.Module):
    def __init__(self,
                 in_dim,
                 class_map,
                 hidden_dims=None,
                 act_fn='relu',
                 enable_binary=True):
        super(VecModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.class_map = class_map
        self.idx_to_class_map = [
            x for x, _ in sorted(self.class_map.items(), key=lambda x: x[1])
        ]
        self.label_ct = len(class_map)
        self.enable_binary = enable_binary

        next_out = self.label_ct if hidden_dims is None else hidden_dims[0]
        if self.label_ct == 2 and self.enable_binary:
            final_out = 1
        else:
            final_out = self.label_ct

        self.w = nn.Linear(in_dim, final_out)
        with torch.no_grad():
            self.w.weight.data.uniform_(-1, 1)
            self.w.bias.data.uniform_(-1, 1)

        if hidden_dims is not None:

            layers = [self.w] + [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ] + [nn.Linear(hidden_dims[-1], final_out)]
            self.w = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

        self.norm_w = nn.Parameter(torch.ones(
            self.in_dim, device=next(self.parameters()).device).float(),
                                   requires_grad=False)
        self.norm_b = nn.Parameter(torch.zeros(
            self.in_dim, device=next(self.parameters()).device).float(),
                                   requires_grad=False)

    def init_norms(self, dataset):
        self.norm_b.data = torch.mean(dataset.features, dim=0)
        self.norm_w.data = 1. / torch.std(dataset.features, dim=0)
        w_finites = torch.isfinite(self.norm_w.data)
        if not torch.all(w_finites):
            self.norm_w.data[~w_finites] = torch.ones(
                torch.sum(~w_finites), device=self.norm_w.data.device).float()

    def logits(self, x):
        x = (x - self.norm_b) * self.norm_w
        if self.label_ct == 2 and self.enable_binary:
            single_logit = torch.sigmoid(self.w(x))
            return torch.log(torch.cat([single_logit, 1 - single_logit],
                                       dim=1))
        else:
            return self.w(x)

    def forward(self, x):
        return torch.argmax(self.logits(x), dim=1)

    def get_probs(self, x):
        if self.label_ct == 2 and self.enable_binary:
            return self.logits(x).exp()
        else:
            softmax = nn.Softmax(dim=1, device=next(self.parameters()).device)
            return softmax(self.logits(x))

    def predict(self, dev_data, device):
        results = []
        for idx, batch in enumerate(
                DataLoader(dev_data,
                           batch_size=128,
                           collate_fn=gen_vec_collate_fn(self.class_map))):
            features_batch, _ = [item.to(device) for item in batch]
            results.append(self(features_batch))
        predictions = torch.cat(results)
        return [self.idx_to_class_map[idx] for idx in list(predictions)]

    def save_params(self):
        return (self.in_dim, self.class_map, self.hidden_dims, self.act_fn)
