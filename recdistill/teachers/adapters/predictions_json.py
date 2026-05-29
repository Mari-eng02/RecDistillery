from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import PrecomputedTopKScorer, TeacherState


class PredictionsJsonAdapter:
    name = "predictions_json"

    def can_load(self, source: TeacherSource) -> bool:
        if _matches(source.format) or _matches(source.framework):
            return True
        return source.path is not None and Path(source.path).suffix.lower() == ".json"

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
        if source.path is None:
            raise ValueError("PredictionsJsonAdapter requires --input.")
        payload = json.loads(Path(source.path).read_text(encoding="utf-8"))
        rows = _prediction_rows(payload)
        scorer, metadata = _topk_scorer_from_rows(rows, source.metadata)
        metadata.update(source.metadata)
        metadata.setdefault("source_path", str(source.path))
        metadata.setdefault("source_format", "predictions_json")
        if source.model_name:
            metadata.setdefault("model_name", source.model_name)
        state = TeacherState(metadata=metadata, scorer=scorer)
        if device is not None:
            return state.to(device)
        return state


def _matches(value: str | None) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in {
        "predictions_json",
        "prediction_json",
        "json_predictions",
        "json",
    }


def _prediction_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported predictions JSON payload type: {type(payload)!r}")
    if "predictions" in payload:
        return _prediction_rows(payload["predictions"])
    required = {"user", "item"}
    if not required.issubset(payload):
        raise ValueError("Predictions JSON must contain user and item columns.")
    length = len(payload["user"])
    rows = []
    for index in range(length):
        row = {key: values[index] for key, values in payload.items() if isinstance(values, list)}
        rows.append(row)
    return rows


def _topk_scorer_from_rows(
    rows: list[dict[str, Any]],
    source_metadata: dict[str, Any] | None = None,
) -> tuple[PrecomputedTopKScorer, dict[str, Any]]:
    if not rows:
        raise ValueError("Predictions JSON contains no rows.")

    source_metadata = source_metadata or {}
    if _rows_use_integer_ids(rows):
        return _topk_scorer_from_integer_rows(rows, source_metadata)
    return _topk_scorer_from_mapped_rows(rows)


def _topk_scorer_from_integer_rows(
    rows: list[dict[str, Any]],
    source_metadata: dict[str, Any],
) -> tuple[PrecomputedTopKScorer, dict[str, Any]]:
    user_values = [_parse_non_negative_int(row["user"], field="user") for row in rows]
    item_values = [_parse_non_negative_int(row["item"], field="item") for row in rows]
    num_users = _metadata_positive_int(source_metadata, "num_users") or (max(user_values) + 1)
    num_items = _metadata_positive_int(source_metadata, "num_items") or (max(item_values) + 1)
    num_users = max(num_users, max(user_values) + 1)
    num_items = max(num_items, max(item_values) + 1)

    grouped: dict[int, list[tuple[int, float, float]]] = {index: [] for index in range(num_users)}
    for row_index, row in enumerate(rows):
        user = user_values[row_index]
        item = item_values[row_index]
        rank = _row_rank(row, row_index)
        score = _row_score(row, rank)
        grouped[user].append((item, score, rank))

    top_k = max((len(values) for values in grouped.values()), default=0)
    topk_items = torch.full((num_users, top_k), -1, dtype=torch.long)
    topk_scores = torch.full((num_users, top_k), float("-inf"), dtype=torch.float32)
    for user, values in grouped.items():
        values.sort(key=lambda entry: (entry[2], -entry[1]))
        for offset, (item, score, _) in enumerate(values):
            topk_items[user, offset] = item
            topk_scores[user, offset] = score

    scorer = PrecomputedTopKScorer(
        topk_items=topk_items,
        topk_scores=topk_scores,
        num_items_override=num_items,
    )
    metadata = {
        "representation": "topk",
        "id_space": "dataset_integer",
        "num_users": num_users,
        "num_items": num_items,
        "top_k": top_k,
    }
    return scorer, metadata


def _topk_scorer_from_mapped_rows(rows: list[dict[str, Any]]) -> tuple[PrecomputedTopKScorer, dict[str, Any]]:
    user_ids = sorted({row["user"] for row in rows}, key=_sort_key)
    item_ids = sorted({row["item"] for row in rows}, key=_sort_key)
    user_to_local = {value: index for index, value in enumerate(user_ids)}
    item_to_local = {value: index for index, value in enumerate(item_ids)}
    grouped: dict[int, list[tuple[int, float, float]]] = {index: [] for index in range(len(user_ids))}

    for row_index, row in enumerate(rows):
        user = user_to_local[row["user"]]
        item = item_to_local[row["item"]]
        rank = _row_rank(row, row_index)
        score = _row_score(row, rank)
        grouped[user].append((item, score, rank))

    top_k = max(len(values) for values in grouped.values())
    topk_items = torch.full((len(user_ids), top_k), -1, dtype=torch.long)
    topk_scores = torch.full((len(user_ids), top_k), float("-inf"), dtype=torch.float32)
    for user, values in grouped.items():
        values.sort(key=lambda entry: (entry[2], -entry[1]))
        for offset, (item, score, _) in enumerate(values):
            topk_items[user, offset] = item
            topk_scores[user, offset] = score

    scorer = PrecomputedTopKScorer(
        topk_items=topk_items,
        topk_scores=topk_scores,
        num_items_override=len(item_ids),
    )
    metadata = {
        "representation": "topk",
        "num_users": len(user_ids),
        "num_items": len(item_ids),
        "top_k": top_k,
        "local_to_public_user_id": user_ids,
        "local_to_public_item_id": item_ids,
        "public_to_local_user_id": {str(key): value for key, value in user_to_local.items()},
        "public_to_local_item_id": {str(key): value for key, value in item_to_local.items()},
    }
    return scorer, metadata


def _rows_use_integer_ids(rows: list[dict[str, Any]]) -> bool:
    return all(_is_non_negative_int_like(row.get("user")) and _is_non_negative_int_like(row.get("item")) for row in rows)


def _is_non_negative_int_like(value: Any) -> bool:
    try:
        return _parse_non_negative_int(value, field="id") >= 0
    except (TypeError, ValueError):
        return False


def _parse_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} ID must be an integer, got boolean.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field} ID cannot be empty.")
        parsed = int(stripped)
    else:
        raise TypeError(f"{field} ID must be integer-like, got {type(value)!r}.")
    if parsed < 0:
        raise ValueError(f"{field} ID must be non-negative, got {parsed}.")
    return parsed


def _metadata_positive_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _row_rank(row: dict[str, Any], fallback: int) -> float:
    if row.get("rank") is not None:
        return float(row["rank"])
    if row.get("position") is not None:
        return float(row["position"])
    return float(fallback)


def _row_score(row: dict[str, Any], rank: float) -> float:
    for key in ("score", "rating", "prediction"):
        if row.get(key) is not None:
            return float(row[key])
    return -rank


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))
