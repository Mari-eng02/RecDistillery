from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from recdistill.data.batch import InteractionBatch
from recdistill.distillers.base import Distiller
from recdistill.teachers.state import TeacherState


class FTDistiller(Distiller):
    def __init__(self, lambda_td: float = 1e-3, entity_sample_size: int = 0):
        super().__init__()
        self.lambda_td = lambda_td
        self.entity_sample_size = int(entity_sample_size)
        self.register_buffer("_teacher_users", torch.empty(0), persistent=False)
        self.register_buffer("_teacher_items", torch.empty(0), persistent=False)

    def on_train_start(self, teacher_state: TeacherState, dataset) -> None:
        del dataset
        device = teacher_state.device
        self._teacher_users = teacher_state.user_embeddings.detach().to(device)
        self._teacher_items = teacher_state.item_embeddings.detach().to(device)

    def _cosine_similarity(self, x: Tensor, y: torch.Tensor) -> Tensor:
        x_norm = F.normalize(x, dim=-1, eps=1e-8)
        y_norm = F.normalize(y, dim=-1, eps=1e-8)
        return x_norm @ y_norm.T

    def _sample_entities(self, users: torch.Tensor, items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.entity_sample_size <= 0:
            return users, items

        total = int(users.numel() + items.numel())
        if total <= self.entity_sample_size:
            return users, items

        user_count = int(users.numel())
        item_count = int(items.numel())
        keep_users = min(user_count, max(1, round(self.entity_sample_size * user_count / total)))
        keep_items = min(item_count, max(1, self.entity_sample_size - keep_users))

        if keep_users + keep_items > self.entity_sample_size:
            keep_items = max(1, self.entity_sample_size - keep_users)

        users = users[torch.randperm(user_count, device=users.device)[:keep_users]]
        items = items[torch.randperm(item_count, device=items.device)[:keep_items]]
        return users, items

    def compute_loss(self, student: nn.Module, batch: InteractionBatch, aux_batch: object | None = None) -> Tensor:
        del aux_batch
        u_indices = batch.unique_users
        i_indices = batch.unique_items
        u_indices, i_indices = self._sample_entities(u_indices, i_indices)

        device = self._teacher_users.device if self._teacher_users.numel() else next(self.parameters()).device
        t_u = self._teacher_users[u_indices].to(device)
        t_i = self._teacher_items[i_indices].to(device)

        s_all_u = student.get_all_user_embeddings().to(device)
        s_all_i = student.get_all_item_embeddings().to(device)
        s_u = s_all_u[u_indices]
        s_i = s_all_i[i_indices]

        E_t = torch.cat([t_u, t_i], dim=0)
        E_s = torch.cat([s_u, s_i], dim=0)

        A_t = self._cosine_similarity(E_t, E_t)
        A_s = self._cosine_similarity(E_s, E_s)
        L_FTD = torch.sum((A_t - A_s) ** 2)
        normalizer = max(1, int(batch.users.numel()))
        return (L_FTD / normalizer) * self.lambda_td
