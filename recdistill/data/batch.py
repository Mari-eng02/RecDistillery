from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InteractionBatch:
    users: torch.Tensor
    pos_items: torch.Tensor
    neg_items: torch.Tensor

    @property
    def unique_users(self) -> torch.Tensor:
        return self.users.unique()

    @property
    def unique_items(self) -> torch.Tensor:
        return torch.cat([self.pos_items, self.neg_items]).unique()


@dataclass
class RRDAuxBatch:
    users: torch.Tensor
    interesting_items: torch.Tensor
    uninteresting_items: torch.Tensor


@dataclass
class UnKDAuxBatch:
    users: torch.Tensor
    pos_items: torch.Tensor
    neg_items: torch.Tensor
