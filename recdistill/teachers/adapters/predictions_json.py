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
        "id_space": _infer_integer_id_space(
            source_metadata=source_metadata,
            user_values=user_values,
            item_values=item_values,
            num_users=num_users,
            num_items=num_items,
        ),
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


def _infer_integer_id_space(
    *,
    source_metadata: dict[str, Any],
    user_values: list[int],
    item_values: list[int],
    num_users: int,
    num_items: int,
) -> str:
    explicit = source_metadata.get("id_space")
    if explicit is not None:
        return str(explicit)

    dataset_name = source_metadata.get("dataset")
    if not dataset_name:
        return "dataset_integer"

    raw_num_users, raw_num_items = _raw_numeric_shape_from_splits(str(dataset_name))
    if raw_num_users is None or raw_num_items is None:
        return "dataset_integer"

    rows_fit_dense_shape = (max(user_values) + 1 <= num_users) and (max(item_values) + 1 <= num_items)
    raw_space_is_larger = raw_num_users > num_users or raw_num_items > num_items
    if rows_fit_dense_shape and raw_space_is_larger:
        return "internal_integer"

    return "dataset_integer"


def _raw_numeric_shape_from_splits(dataset_name: str) -> tuple[int | None, int | None]:
    try:
        from recdistill.data.datarec_loader import ITEM_COLUMNS, SPLIT_ORDER, USER_COLUMNS, _pick_column, load_split_frame
    except Exception:
        return None, None

    max_user = -1
    max_item = -1
    for split_name in SPLIT_ORDER:
        try:
            frame = load_split_frame(dataset_name, split_name, use_datarec=False).frame
            user_col = _pick_column(frame.columns, USER_COLUMNS)
            item_col = _pick_column(frame.columns, ITEM_COLUMNS)
        except Exception:
            continue
        for user_raw, item_raw in frame[[user_col, item_col]].itertuples(index=False, name=None):
            try:
                user = int(user_raw)
                item = int(item_raw)
            except (TypeError, ValueError):
                return None, None
            max_user = max(max_user, user)
            max_item = max(max_item, item)

    if max_user < 0 or max_item < 0:
        return None, None
    return max_user + 1, max_item + 1


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
