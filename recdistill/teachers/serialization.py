from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from recdistill.teachers.state import PrecomputedScoresScorer, PrecomputedTopKScorer, TeacherState
from recdistill.tracking import utc_now_iso


TEACHER_FORMAT_VERSION = "recdistill.teacher.v2"
SUPPORTED_TEACHER_FORMAT_VERSIONS = {"recdistill.teacher.v1", TEACHER_FORMAT_VERSION}


def _scorer_to_payload(state: TeacherState) -> dict[str, Any] | None:
    scorer = state.scorer
    if isinstance(scorer, PrecomputedScoresScorer):
        return {
            "type": "precomputed_scores",
            "scores": scorer.scores.detach().cpu(),
        }
    if isinstance(scorer, PrecomputedTopKScorer):
        return {
            "type": "precomputed_topk",
            "topk_items": scorer.topk_items.detach().cpu(),
            "topk_scores": scorer.topk_scores.detach().cpu() if scorer.topk_scores is not None else None,
            "fill_value": float(scorer.fill_value),
            "num_items": scorer.num_items,
        }
    return None


def _scorer_from_payload(payload: dict[str, Any]) -> object | None:
    scorer_payload = payload.get("scorer")
    if not isinstance(scorer_payload, dict):
        return None
    scorer_type = str(scorer_payload.get("type") or "").lower()
    if scorer_type == "precomputed_scores":
        return PrecomputedScoresScorer(scores=torch.as_tensor(scorer_payload["scores"], dtype=torch.float32))
    if scorer_type == "precomputed_topk":
        return PrecomputedTopKScorer(
            topk_items=torch.as_tensor(scorer_payload["topk_items"], dtype=torch.long),
            topk_scores=(
                torch.as_tensor(scorer_payload["topk_scores"], dtype=torch.float32)
                if scorer_payload.get("topk_scores") is not None
                else None
            ),
            fill_value=float(scorer_payload.get("fill_value", float("-inf"))),
            num_items_override=int(scorer_payload["num_items"]) if scorer_payload.get("num_items") is not None else None,
        )
    raise ValueError(f"Unsupported serialized teacher scorer type: {scorer_type!r}")


def teacher_state_to_payload(
    state: TeacherState,
    *,
    framework: str | None = None,
    model_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = dict(state.metadata)
    if metadata:
        merged_metadata.update(metadata)
    if framework is not None:
        merged_metadata["framework"] = framework
    if model_name is not None:
        merged_metadata["model_name"] = model_name

    scorer_payload = _scorer_to_payload(state)
    if scorer_payload is None and not state.has_embeddings:
        raise ValueError("Cannot serialize a non-embedding teacher with an unsupported scorer type.")

    return {
        "format_version": TEACHER_FORMAT_VERSION,
        "created_at_utc": utc_now_iso(),
        "framework": framework or merged_metadata.get("framework") or merged_metadata.get("source"),
        "model_name": model_name or merged_metadata.get("model_name"),
        "num_users": state.num_users,
        "num_items": state.num_items,
        "user_embeddings": state.user_embeddings.detach().cpu() if state.user_embeddings is not None else None,
        "item_embeddings": state.item_embeddings.detach().cpu() if state.item_embeddings is not None else None,
        "scorer": scorer_payload,
        "local_to_public_user_id": merged_metadata.get("local_to_public_user_id"),
        "local_to_public_item_id": merged_metadata.get("local_to_public_item_id"),
        "public_to_local_user_id": merged_metadata.get("public_to_local_user_id"),
        "public_to_local_item_id": merged_metadata.get("public_to_local_item_id"),
        "metadata": merged_metadata,
    }


def teacher_state_from_payload(payload: dict[str, Any]) -> TeacherState:
    if payload.get("format_version") not in SUPPORTED_TEACHER_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported teacher format: {payload.get('format_version')!r}. "
            f"Expected one of {sorted(SUPPORTED_TEACHER_FORMAT_VERSIONS)}."
        )
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "framework",
        "model_name",
        "created_at_utc",
        "num_users",
        "num_items",
        "local_to_public_user_id",
        "local_to_public_item_id",
        "public_to_local_user_id",
        "public_to_local_item_id",
    ):
        if key in payload and payload[key] is not None:
            metadata.setdefault(key, payload[key])
    metadata.setdefault("source", "recdistill_native_teacher")
    return TeacherState(
        user_embeddings=(
            torch.as_tensor(payload["user_embeddings"], dtype=torch.float32)
            if payload.get("user_embeddings") is not None
            else None
        ),
        item_embeddings=(
            torch.as_tensor(payload["item_embeddings"], dtype=torch.float32)
            if payload.get("item_embeddings") is not None
            else None
        ),
        metadata=metadata,
        scorer=_scorer_from_payload(payload),
    )


def save_teacher_state(
    path: str | Path,
    state: TeacherState,
    *,
    framework: str | None = None,
    model_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = teacher_state_to_payload(
        state,
        framework=framework,
        model_name=model_name,
        metadata=metadata,
    )
    torch.save(payload, output_path)
    return payload


def load_teacher_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported teacher payload type: {type(payload)!r}")
    return payload
