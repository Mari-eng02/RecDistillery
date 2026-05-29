from __future__ import annotations

import random
import os

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from recdistill.data.interactions import InteractionDataset
from recdistill.samplers import BPRNegativeSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class PositiveInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, interactions: list[tuple[int, int]]):
        self._interactions = interactions

    def __len__(self) -> int:
        return len(self._interactions)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self._interactions[idx]


class BPRBatchCollator:
    def __init__(self, negative_sampler: BPRNegativeSampler):
        self.negative_sampler = negative_sampler

    def __call__(self, batch_rows: list[tuple[int, int]]):
        users = torch.tensor([user for user, _ in batch_rows], dtype=torch.long)
        pos_items = torch.tensor([item for _, item in batch_rows], dtype=torch.long)
        neg_items = torch.tensor(
            [self.negative_sampler.sample(user) for user, _ in batch_rows],
            dtype=torch.long,
        )
        return users, pos_items, neg_items


def build_train_loader(
    dataset: InteractionDataset,
    batch_size: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    pair_dataset = PositiveInteractionDataset(dataset.interactions)
    negative_sampler = BPRNegativeSampler(dataset)

    loader_kwargs = dict(
        dataset=pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=BPRBatchCollator(negative_sampler),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(**loader_kwargs)


def build_lightgcn_graph(dataset: InteractionDataset) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    item_offset = dataset.num_users
    for user, item in dataset.interactions:
        item_node = item_offset + item
        rows.extend([user, item_node])
        cols.extend([item_node, user])
    return torch.tensor([rows, cols], dtype=torch.long)


def prepare_distiller_trainable_modules(distiller, student_dim: int, device: torch.device) -> None:
    if distiller is None:
        return
    children = getattr(distiller, "distillers", None)
    if children is not None:
        for child in children:
            prepare_distiller_trainable_modules(child, student_dim, device)
        return
    prepare = getattr(distiller, "_ensure_student_adapters", None)
    if callable(prepare):
        prepare(student_dim=int(student_dim), device=device)
