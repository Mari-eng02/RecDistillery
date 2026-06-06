from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


class TeacherScorer(Protocol):
    """Protocol implemented by scorer-only teacher representations.

    A scorer computes item scores for one user without requiring explicit
    user/item embedding matrices. Import adapters use it for prediction exports,
    top-k rankings, or dense score matrices.
    """

    def to(self, device: torch.device | str): ...
    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor: ...


@dataclass
class PrecomputedScoresScorer:
    """Dense precomputed teacher score matrix.

    Attributes:
        scores: Tensor with shape `[num_users, num_items]`. Each row contains
            the teacher scores for all candidate items of one user.
    """

    scores: torch.Tensor

    def __post_init__(self) -> None:
        self.scores = torch.as_tensor(self.scores, dtype=torch.float32)
        if self.scores.ndim != 2:
            raise ValueError("Precomputed score matrix must be a rank-2 tensor.")

    @property
    def num_users(self) -> int:
        return int(self.scores.size(0))

    @property
    def num_items(self) -> int:
        return int(self.scores.size(1))

    def to(self, device: torch.device | str) -> "PrecomputedScoresScorer":
        return PrecomputedScoresScorer(scores=self.scores.to(device))

    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        """Return the score vector for a user, padded or truncated to `num_items`."""
        if user < 0 or user >= self.num_users:
            raise IndexError(f"User index out of bounds for precomputed scores: {user}")
        scores = self.scores[user]
        if scores.numel() >= num_items:
            return scores[:num_items]
        padded = scores.new_full((num_items,), float("-inf"))
        padded[: scores.numel()] = scores
        return padded


@dataclass
class PrecomputedTopKScorer:
    """Sparse scorer backed by precomputed ranked items.

    Attributes:
        topk_items: Integer tensor with shape `[num_users, top_k]`.
        topk_scores: Optional score tensor aligned with `topk_items`.
        fill_value: Score assigned to items that are absent from the top-k list.
        num_items_override: Optional catalog size when it cannot be inferred
            from the maximum item id.
    """

    topk_items: torch.Tensor
    topk_scores: torch.Tensor | None = None
    fill_value: float = float("-inf")
    num_items_override: int | None = None

    def __post_init__(self) -> None:
        self.topk_items = torch.as_tensor(self.topk_items, dtype=torch.long)
        if self.topk_items.ndim != 2:
            raise ValueError("Precomputed top-k items must be a rank-2 tensor.")
        if self.topk_scores is not None:
            self.topk_scores = torch.as_tensor(self.topk_scores, dtype=torch.float32)
            if self.topk_scores.shape != self.topk_items.shape:
                raise ValueError("Precomputed top-k scores must have the same shape as top-k items.")

    @property
    def num_users(self) -> int:
        return int(self.topk_items.size(0))

    @property
    def top_k(self) -> int:
        return int(self.topk_items.size(1))

    @property
    def num_items(self) -> int:
        if self.num_items_override is not None:
            return int(self.num_items_override)
        valid = self.topk_items[self.topk_items >= 0]
        return int(valid.max().item() + 1) if valid.numel() else 0

    def to(self, device: torch.device | str) -> "PrecomputedTopKScorer":
        return PrecomputedTopKScorer(
            topk_items=self.topk_items.to(device),
            topk_scores=self.topk_scores.to(device) if self.topk_scores is not None else None,
            fill_value=float(self.fill_value),
            num_items_override=self.num_items_override,
        )

    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        """Expand one user's top-k ranking into a full score vector."""
        if user < 0 or user >= self.num_users:
            raise IndexError(f"User index out of bounds for precomputed top-k: {user}")
        scores = torch.full((num_items,), float(self.fill_value), dtype=torch.float32, device=self.topk_items.device)
        items = self.topk_items[user]
        valid = (items >= 0) & (items < num_items)
        if not bool(valid.any()):
            return scores
        valid_items = items[valid]
        if self.topk_scores is not None:
            valid_scores = self.topk_scores[user][valid].to(dtype=torch.float32, device=scores.device)
        else:
            ranks = torch.arange(valid_items.numel(), dtype=torch.float32, device=scores.device)
            valid_scores = -ranks
        scores[valid_items] = valid_scores
        return scores


@dataclass
class TeacherState:
    """Framework-neutral teacher representation used by distillation.

    A teacher can be represented either by user/item embeddings or by a scorer.
    The same state object is used for native teachers, imported checkpoints,
    prediction JSON files, and serialized `.teacher` artifacts.

    Attributes:
        user_embeddings: Optional user embedding matrix.
        item_embeddings: Optional item embedding matrix.
        metadata: Free-form provenance and mapping information.
        scorer: Optional scorer-only representation.
    """

    user_embeddings: torch.Tensor | None = None
    item_embeddings: torch.Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    scorer: TeacherScorer | None = None

    def __post_init__(self) -> None:
        if self.user_embeddings is not None:
            self.user_embeddings = torch.as_tensor(self.user_embeddings, dtype=torch.float32)
            if self.user_embeddings.ndim != 2:
                raise ValueError("TeacherState.user_embeddings must be a rank-2 tensor.")
        if self.item_embeddings is not None:
            self.item_embeddings = torch.as_tensor(self.item_embeddings, dtype=torch.float32)
            if self.item_embeddings.ndim != 2:
                raise ValueError("TeacherState.item_embeddings must be a rank-2 tensor.")
        if (self.user_embeddings is None) != (self.item_embeddings is None):
            raise ValueError("TeacherState requires both user and item embeddings, or neither.")
        if self.user_embeddings is not None and self.item_embeddings is not None:
            if self.user_embeddings.size(1) != self.item_embeddings.size(1):
                raise ValueError("TeacherState user/item embeddings must share the same embedding dimension.")
        if self.user_embeddings is None and self.scorer is None:
            raise ValueError("TeacherState requires embeddings or a scorer.")

    @property
    def num_users(self) -> int:
        if self.user_embeddings is not None:
            return int(self.user_embeddings.size(0))
        if hasattr(self.scorer, "num_users"):
            return int(getattr(self.scorer, "num_users"))
        if "num_users" in self.metadata:
            return int(self.metadata["num_users"])
        raise ValueError("TeacherState.num_users is unavailable without embeddings, scorer metadata, or num_users metadata.")

    @property
    def num_items(self) -> int:
        if self.item_embeddings is not None:
            return int(self.item_embeddings.size(0))
        if hasattr(self.scorer, "num_items"):
            return int(getattr(self.scorer, "num_items"))
        if "num_items" in self.metadata:
            return int(self.metadata["num_items"])
        raise ValueError("TeacherState.num_items is unavailable without embeddings, scorer metadata, or num_items metadata.")

    @property
    def embedding_dim(self) -> int:
        if self.user_embeddings is None:
            raise ValueError("TeacherState has no embedding representation.")
        return int(self.user_embeddings.size(1))

    @property
    def device(self) -> torch.device:
        if self.user_embeddings is not None:
            return self.user_embeddings.device
        if isinstance(self.scorer, PrecomputedScoresScorer):
            return self.scorer.scores.device
        if isinstance(self.scorer, PrecomputedTopKScorer):
            return self.scorer.topk_items.device
        return torch.device("cpu")

    @property
    def has_embeddings(self) -> bool:
        return self.user_embeddings is not None and self.item_embeddings is not None

    def to(self, device: torch.device | str) -> "TeacherState":
        """Return a copy of the teacher state moved to `device`."""
        scorer = self.scorer.to(device) if self.scorer is not None else None
        return TeacherState(
            user_embeddings=self.user_embeddings.to(device) if self.user_embeddings is not None else None,
            item_embeddings=self.item_embeddings.to(device) if self.item_embeddings is not None else None,
            metadata=dict(self.metadata),
            scorer=scorer,
        )
