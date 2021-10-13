# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

from typing import List

import torch as th
from torch import nn
from torch.nn import functional as F
from utils.loaders.bprmf import BprmfDataset


class BPRMF(nn.Module):
    def __init__(
            self,
            dataset: BprmfDataset,
            dim_embed_global: int,
            rate_reg: float
    ) -> None:
        super(BPRMF, self).__init__()

        self.embed_user = nn.Embedding(dataset.n_users, dim_embed_global)
        nn.init.xavier_uniform_(self.embed_user.weight,
                                gain=nn.init.calculate_gain('relu'))
        self.embed_item = nn.Embedding(dataset.n_items, dim_embed_global)
        nn.init.xavier_uniform_(self.embed_item.weight,
                                gain=nn.init.calculate_gain('relu'))

        self.rate_reg = rate_reg

    def forward(self, *input):
        return self._compute_loss(*input)

    def _compute_loss(self, users, items_pos, items_neg):
        embed_user = self.embed_user(users)  # (batch_size, dim_embed_global)
        embed_item_pos = self.embed_item(items_pos)  # (batch_size, dim_embed_global)
        embed_item_neg = self.embed_item(items_neg)  # (batch_size, dim_embed_global)

        score_pos = th.sum(embed_user * embed_item_pos, dim=1)  # (batch_size)
        score_neg = th.sum(embed_user * embed_item_neg, dim=1)  # (batch_size)

        loss_base = (-1.0) * F.logsigmoid(score_pos - score_neg)
        loss_base = th.mean(loss_base)

        loss_reg = self._l2_loss(
            embed_user) + self._l2_loss(embed_item_pos) + self._l2_loss(embed_item_neg)
        loss_reg = self.rate_reg * loss_reg
        return loss_base, loss_reg

    def _l2_loss(self, embedd):
        return th.sum(embedd.pow(2).sum(1) / 2.0)

    def predict(self, users, items):
        embed_user = self.embed_user(users)  # (n_eval_users, dim_embed_global)
        embed_item = self.embed_item(items)  # (n_eval_items, dim_embed_global)
        return th.matmul(embed_user, embed_item.transpose(0, 1))  # (n_eval_users, n_eval_items)
