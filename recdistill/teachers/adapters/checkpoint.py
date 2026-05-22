from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from recdistill.teachers.serialization import (
    SUPPORTED_TEACHER_FORMAT_VERSIONS,
    teacher_state_from_payload,
)
from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import PrecomputedScoresScorer, PrecomputedTopKScorer, TeacherState


class CheckpointAdapter:
    name = "checkpoint"

    def can_load(self, source: TeacherSource) -> bool:
        if _matches(source.format) or _matches(source.framework):
            return True
        if source.path is None:
            return False
        if Path(source.path).suffix.lower() not in {".teacher", ".pt", ".pth", ".ckpt"}:
            return False
        try:
            payload = _load_checkpoint(source.path)
        except Exception:
            return False
        return _can_build_teacher_state(payload)

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
        if source.path is None:
            raise ValueError("CheckpointAdapter requires --input.")
        payload = _load_checkpoint(source.path)
        state = _teacher_state_from_checkpoint(payload)
        state.metadata.update(source.metadata)
        state.metadata.setdefault("source_path", str(source.path))
        state.metadata.setdefault("source_format", "checkpoint")
        if source.model_name:
            state.metadata.setdefault("model_name", source.model_name)
        if device is not None:
            return state.to(device)
        return state


def _matches(value: str | None) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in {
        "checkpoint",
        "torch_checkpoint",
        "torch",
        "teacher",
    }


def _load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def _can_build_teacher_state(payload: Any) -> bool:
    if isinstance(payload, TeacherState):
        return True
    if not isinstance(payload, dict):
        return False
    if payload.get("format_version") in SUPPORTED_TEACHER_FORMAT_VERSIONS:
        return True
    return (
        _has_any(payload, "user_embeddings", "user_embedding", "user_emb", "users")
        and _has_any(payload, "item_embeddings", "item_embedding", "item_emb", "items")
    ) or _has_any(payload, "scores", "score_matrix", "teacher_scores") or _has_any(
        payload, "topk_items", "top_items", "rankings"
    )


def _teacher_state_from_checkpoint(payload: Any) -> TeacherState:
    if isinstance(payload, TeacherState):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    if payload.get("format_version") in SUPPORTED_TEACHER_FORMAT_VERSIONS:
        return teacher_state_from_payload(payload)
    if _has_any(payload, "scores", "score_matrix", "teacher_scores"):
        scores = _pick(payload, "scores", "score_matrix", "teacher_scores")
        scorer = PrecomputedScoresScorer(scores=torch.as_tensor(scores, dtype=torch.float32))
        return TeacherState(
            scorer=scorer,
            metadata={"representation": "scores", "num_users": scorer.num_users, "num_items": scorer.num_items},
        )
    if _has_any(payload, "topk_items", "top_items", "rankings"):
        topk_items = _pick(payload, "topk_items", "top_items", "rankings")
        topk_scores = _pick_optional(payload, "topk_scores", "top_scores", "ranking_scores")
        scorer = PrecomputedTopKScorer(
            topk_items=torch.as_tensor(topk_items, dtype=torch.long),
            topk_scores=torch.as_tensor(topk_scores, dtype=torch.float32) if topk_scores is not None else None,
            num_items_override=int(payload["num_items"]) if payload.get("num_items") is not None else None,
        )
        return TeacherState(
            scorer=scorer,
            metadata={"representation": "topk", "num_users": scorer.num_users, "num_items": scorer.num_items},
        )

    user_embeddings = _pick(payload, "user_embeddings", "user_embedding", "user_emb", "users")
    item_embeddings = _pick(payload, "item_embeddings", "item_embedding", "item_emb", "items")
    return TeacherState(
        user_embeddings=torch.as_tensor(user_embeddings, dtype=torch.float32),
        item_embeddings=torch.as_tensor(item_embeddings, dtype=torch.float32),
        metadata={"representation": "embeddings"},
    )


def _has_any(payload: dict[str, Any], *keys: str) -> bool:
    return any(key in payload and payload[key] is not None for key in keys)


def _pick(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    raise KeyError(f"Unable to find any of {keys} in checkpoint payload.")


def _pick_optional(payload: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None
