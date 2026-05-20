from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import PrecomputedScoresScorer, PrecomputedTopKScorer, TeacherState


class NumpyEmbeddingsTeacherAdapter:
    name = "numpy"

    def can_load(self, source: TeacherSource) -> bool:
        if _matches(source.framework) or _matches(source.format):
            return True
        if source.user_embeddings_path is not None and source.item_embeddings_path is not None:
            return True
        if source.score_matrix_path is not None or source.topk_items_path is not None:
            return True
        if source.path is not None and Path(source.path).suffix.lower() == ".npz":
            return True
        return False

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
        scorer = None
        user_embeddings = None
        item_embeddings = None
        if source.score_matrix_path is not None:
            scores = np.load(source.score_matrix_path)
            scorer = PrecomputedScoresScorer(scores=torch.as_tensor(scores, dtype=torch.float32))
            metadata: dict[str, Any] = {
                "source": "numpy_score_matrix",
                "score_matrix_path": str(source.score_matrix_path),
                "num_users": scorer.num_users,
                "num_items": scorer.num_items,
            }
        elif source.topk_items_path is not None:
            topk_items = np.load(source.topk_items_path)
            topk_scores = np.load(source.topk_scores_path) if source.topk_scores_path is not None else None
            scorer = PrecomputedTopKScorer(
                topk_items=torch.as_tensor(topk_items, dtype=torch.long),
                topk_scores=torch.as_tensor(topk_scores, dtype=torch.float32) if topk_scores is not None else None,
                num_items_override=int(source.metadata["num_items"]) if "num_items" in source.metadata else None,
            )
            metadata = {
                "source": "numpy_topk",
                "topk_items_path": str(source.topk_items_path),
                "topk_scores_path": str(source.topk_scores_path) if source.topk_scores_path is not None else None,
                "num_users": scorer.num_users,
                "top_k": scorer.top_k,
            }
        elif source.user_embeddings_path is not None and source.item_embeddings_path is not None:
            user_embeddings = np.load(source.user_embeddings_path)
            item_embeddings = np.load(source.item_embeddings_path)
            metadata: dict[str, Any] = {
                "source": "numpy_embeddings",
                "user_embeddings_path": str(source.user_embeddings_path),
                "item_embeddings_path": str(source.item_embeddings_path),
            }
        elif source.path is not None and Path(source.path).suffix.lower() == ".npz":
            payload = np.load(source.path, allow_pickle=True)
            metadata = {"source": "numpy_npz", "source_path": str(source.path)}
            if _has_any(payload, "scores", "score_matrix", "teacher_scores"):
                scores = _pick_array(payload, "scores", "score_matrix", "teacher_scores")
                scorer = PrecomputedScoresScorer(scores=torch.as_tensor(scores, dtype=torch.float32))
                metadata.update({"num_users": scorer.num_users, "num_items": scorer.num_items, "representation": "scores"})
            elif _has_any(payload, "topk_items", "top_items", "rankings"):
                topk_items = _pick_array(payload, "topk_items", "top_items", "rankings")
                topk_scores = _pick_optional_array(payload, "topk_scores", "top_scores", "ranking_scores")
                scorer = PrecomputedTopKScorer(
                    topk_items=torch.as_tensor(topk_items, dtype=torch.long),
                    topk_scores=torch.as_tensor(topk_scores, dtype=torch.float32) if topk_scores is not None else None,
                    num_items_override=int(payload["num_items"]) if "num_items" in payload else None,
                )
                if "num_items" in payload:
                    metadata["num_items"] = int(payload["num_items"])
                metadata.update({"num_users": scorer.num_users, "top_k": scorer.top_k, "representation": "topk"})
            else:
                user_embeddings = _pick_array(payload, "user_embeddings", "user_emb", "users")
                item_embeddings = _pick_array(payload, "item_embeddings", "item_emb", "items")
        else:
            raise ValueError(
                "Numpy teacher adapter requires --input .npz, both embedding paths, "
                "a score matrix path, or a top-k items path."
            )

        metadata.update(source.metadata)
        if source.model_name:
            metadata.setdefault("model_name", source.model_name)
        metadata.setdefault("framework", "numpy")
        state = TeacherState(
            user_embeddings=torch.as_tensor(user_embeddings, dtype=torch.float32) if user_embeddings is not None else None,
            item_embeddings=torch.as_tensor(item_embeddings, dtype=torch.float32) if item_embeddings is not None else None,
            metadata=metadata,
            scorer=scorer,
        )
        if device is not None:
            return state.to(device)
        return state


def _matches(value: str | None) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in {
        "numpy",
        "npz",
        "npy",
        "embeddings_npz",
        "embeddings_npy",
        "numpy_embeddings",
    }


def _pick_array(payload, *keys: str):
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"Unable to find any of {keys} in numpy teacher payload.")


def _pick_optional_array(payload, *keys: str):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _has_any(payload, *keys: str) -> bool:
    return any(key in payload for key in keys)
