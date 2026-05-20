from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from recdistill.data.batch import InteractionBatch
from recdistill.distillers.base import Distiller
from recdistill.teachers.state import TeacherState


class GroupMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_groups: int,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_dim, num_groups * hidden_dim)
        self.fc2 = nn.Conv1d(
            in_channels=num_groups * hidden_dim,
            out_channels=num_groups * out_dim,
            kernel_size=1,
            groups=num_groups,
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        hidden = self.relu(self.fc1(x))
        hidden = hidden.unsqueeze(-1)
        out = self.fc2(hidden)
        out = out.view(batch_size, self.num_groups, -1)
        return out


class HTDistiller(Distiller):
    """Hierarchical Topology Distillation (adapted).

    Uses `TeacherState` for teacher embeddings and the student's
    `get_all_*_embeddings()` API to index student embeddings.
    """

    def __init__(
        self,
        lambda_td: float = 1e-3,
        alpha: float = 0.5,
        num_groups: int = 40,
        topology_mode: str = "group_pe",
        initial_tau: float = 1.0,
        min_tau: float = 1e-10,
        decay_epochs: int = 100,
        entity_sample_size: int = 0,
    ):
        super().__init__()
        self.lambda_td = lambda_td
        self.alpha = alpha
        self.K = num_groups
        if topology_mode not in ["group_pp", "group_pe"]:
            raise ValueError("topology_mode must be 'group_pp' or 'group_pe'")
        self.topology_mode = topology_mode

        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.decay_epochs = decay_epochs
        self.entity_sample_size = int(entity_sample_size)
        self.tau = initial_tau
        self._epoch = 0

        self.register_buffer("_teacher_users", torch.empty(0), persistent=False)
        self.register_buffer("_teacher_items", torch.empty(0), persistent=False)

        self.v_user = None
        self.v_item = None
        self.f_user = None
        self.f_item = None
        self._teacher_dim = None

    def on_train_start(self, teacher_state: TeacherState, dataset) -> None:
        del dataset
        device = teacher_state.device
        self._teacher_users = teacher_state.user_embeddings.detach().to(device)
        self._teacher_items = teacher_state.item_embeddings.detach().to(device)

        d_t = teacher_state.embedding_dim
        self._teacher_dim = d_t

        self.v_user = nn.Sequential(nn.Linear(d_t, self.K), nn.Softmax(dim=1))
        self.v_item = nn.Sequential(nn.Linear(d_t, self.K), nn.Softmax(dim=1))

        self.to(device)

    def _ensure_student_adapters(self, student_dim: int, device: torch.device) -> None:
        if self._teacher_dim is None:
            raise RuntimeError("HTDistiller is not initialized. Call on_train_start first.")

        needs_init = self.f_user is None or self.f_item is None
        if not needs_init:
            user_in = int(self.f_user.fc1.in_features)
            item_in = int(self.f_item.fc1.in_features)
            needs_init = user_in != student_dim or item_in != student_dim

        if needs_init:
            hidden_dim = max(1, (student_dim + self._teacher_dim) // 2)
            self.f_user = GroupMLP(student_dim, hidden_dim, self._teacher_dim, self.K).to(device)
            self.f_item = GroupMLP(student_dim, hidden_dim, self._teacher_dim, self.K).to(device)

    def on_epoch_start(self) -> None:
        self._update_temperature(self._epoch)
        self._epoch += 1

    def _update_temperature(self, current_epoch: int) -> None:
        if current_epoch >= self.decay_epochs:
            self.tau = self.min_tau
        else:
            ratio = current_epoch / self.decay_epochs
            self.tau = self.initial_tau * ((self.min_tau / self.initial_tau) ** ratio)
        self.tau = max(self.tau, self.min_tau)

    def _cosine_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return F.normalize(x, dim=-1, eps=1e-8) @ F.normalize(y, dim=-1, eps=1e-8).T

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

    def _compute_ga_loss(self, t_emb: Tensor, s_emb: Tensor, v_net: nn.Module, f_net: nn.Module):
        alpha = v_net(t_emb) + 1e-10
        z_soft = F.gumbel_softmax(alpha.log(), tau=self.tau, hard=False, dim=-1)
        expert_outputs = f_net(s_emb)
        recon = torch.sum(expert_outputs * z_soft.unsqueeze(-1), dim=1)
        loss = torch.sum((t_emb - recon) ** 2, dim=-1).sum()
        return loss, z_soft

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
        self._ensure_student_adapters(student_dim=int(s_all_u.size(1)), device=device)
        s_u = s_all_u[u_indices]
        s_i = s_all_i[i_indices]

        loss_user, z_soft_u = self._compute_ga_loss(t_u, s_u, self.v_user, self.f_user)
        loss_item, z_soft_i = self._compute_ga_loss(t_i, s_i, self.v_item, self.f_item)
        L_GA = loss_user + loss_item

        E_t = torch.cat([t_u, t_i], dim=0)
        E_s = torch.cat([s_u, s_i], dim=0)

        with torch.no_grad():
            z_u = self.v_user(t_u).argmax(dim=1)
            z_i = self.v_item(t_i).argmax(dim=1) + self.K
            z_all = torch.cat([z_u, z_i], dim=0)

            ones = torch.ones_like(z_all, dtype=torch.float).unsqueeze(1)
            counts = torch.zeros(2 * self.K, 1, device=z_all.device)
            counts.scatter_add_(0, z_all.unsqueeze(1), ones)
            counts = counts + 1e-10

            P_t_sum = torch.zeros(2 * self.K, E_t.shape[1], device=E_t.device)
            P_s_sum = torch.zeros(2 * self.K, E_s.shape[1], device=E_s.device)
            P_t_sum.index_add_(0, z_all, E_t)
            P_s_sum.index_add_(0, z_all, E_s)

            active_groups = z_all.unique()
            P_t = (P_t_sum / counts)[active_groups]
            P_s = (P_s_sum / counts)[active_groups]

            Z = F.one_hot(z_all, num_classes=2 * self.K).float()
            M = Z @ Z.T

        sim_tt = self._cosine_similarity(E_t, E_t) * M
        sim_ss = self._cosine_similarity(E_s, E_s) * M

        valid_mask = sim_tt > 0.0
        sim_tt_filtered = sim_tt[valid_mask]
        sim_ss_filtered = sim_ss[valid_mask]
        L_entity = torch.sum((sim_tt_filtered - sim_ss_filtered) ** 2)

        if self.topology_mode == "group_pp":
            proto_tt = self._cosine_similarity(P_t, P_t).view(-1)
            proto_ss = self._cosine_similarity(P_s, P_s).view(-1)
            L_group = torch.sum((proto_tt - proto_ss) ** 2)
        else:
            proto_dist_t = self._cosine_similarity(P_t, E_t).view(-1)
            proto_dist_s = self._cosine_similarity(P_s, E_s).view(-1)
            L_group = torch.sum((proto_dist_t - proto_dist_s) ** 2)

        L_TD = L_entity + L_group
        HTD_loss = L_TD * self.alpha + L_GA * (1 - self.alpha)
        normalizer = max(1, int(batch.users.numel()))
        return (HTD_loss / normalizer) * self.lambda_td
