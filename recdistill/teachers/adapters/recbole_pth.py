from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import torch

from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import TeacherState


class RecBolePthAdapter:
    name = "recbole_pth"

    def can_load(self, source: TeacherSource) -> bool:
        if _matches(source.format) or _matches(source.framework):
            return True
        if source.path is None or Path(source.path).suffix.lower() != ".pth":
            return False
        try:
            state_dict = _extract_state_dict(_load_checkpoint(source.path))
        except Exception:
            return False
        return _find_user_item_embeddings(state_dict) is not None

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
        if source.path is None:
            raise ValueError("RecBolePthAdapter requires --input.")
        payload = _load_checkpoint(source.path)
        state_dict = _extract_state_dict(payload)
        tensors = _find_user_item_embeddings(state_dict)
        if tensors is None:
            raise ValueError("Unable to find compatible user/item embedding tensors in .pth checkpoint.")
        user_embeddings, item_embeddings, user_key, item_key = tensors
        metadata = {
            "representation": "embeddings",
            "source_path": str(source.path),
            "source_format": "recbole_pth",
            "user_embedding_key": user_key,
            "item_embedding_key": item_key,
        }
        metadata.update(source.metadata)
        if source.model_name:
            metadata.setdefault("model_name", source.model_name)
        state = TeacherState(
            user_embeddings=user_embeddings.detach().cpu().to(dtype=torch.float32),
            item_embeddings=item_embeddings.detach().cpu().to(dtype=torch.float32),
            metadata=metadata,
        )
        if device is not None:
            return state.to(device)
        return state


def _matches(value: str | None) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in {
        "recbole_pth",
        "recbole",
        "pth",
    }


def _load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            if not str(exc.name or "").startswith("recbole"):
                raise
            with _temporary_recbole_stubs():
                return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and _looks_like_state_dict(candidate):
                return candidate
        if _looks_like_state_dict(payload):
            return payload
    raise TypeError("Checkpoint does not contain a tensor state dict.")


def _looks_like_state_dict(candidate: dict[Any, Any]) -> bool:
    return any(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items())


def _find_user_item_embeddings(
    state_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, str, str] | None:
    tensors = {
        key: value
        for key, value in state_dict.items()
        if torch.is_tensor(value) and value.ndim == 2 and torch.is_floating_point(value)
    }
    user_candidates = [(key, value) for key, value in tensors.items() if _is_user_embedding_key(key)]
    item_candidates = [(key, value) for key, value in tensors.items() if _is_item_embedding_key(key)]
    for user_key, user_tensor in user_candidates:
        for item_key, item_tensor in item_candidates:
            if user_tensor.size(1) == item_tensor.size(1):
                return user_tensor, item_tensor, user_key, item_key
    return None


def _is_user_embedding_key(key: str) -> bool:
    normalized = key.lower()
    return "embedding" in normalized and any(token in normalized for token in ("user", "uid"))


def _is_item_embedding_key(key: str) -> bool:
    normalized = key.lower()
    return "embedding" in normalized and any(token in normalized for token in ("item", "iid"))


class _StubConfig(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if isinstance(state, dict):
            for key in ("final_config_dict", "external_config_dict", "variable_config_dict"):
                value = state.get(key)
                if isinstance(value, dict):
                    self.update(value)


class _StubEnum:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _StubModelType:
    GENERAL = _StubEnum("GENERAL")
    SEQUENTIAL = _StubEnum("SEQUENTIAL")
    CONTEXT = _StubEnum("CONTEXT")
    KNOWLEDGE = _StubEnum("KNOWLEDGE")
    TRADITIONAL = _StubEnum("TRADITIONAL")
    DECISIONTREE = _StubEnum("DECISIONTREE")


class _StubInputType:
    POINTWISE = _StubEnum("POINTWISE")
    PAIRWISE = _StubEnum("PAIRWISE")
    LISTWISE = _StubEnum("LISTWISE")


class _StubEvaluatorType:
    RANKING = _StubEnum("RANKING")
    VALUE = _StubEnum("VALUE")


class _StubModel:
    pass


class _temporary_recbole_stubs:
    def __enter__(self) -> None:
        self._previous: dict[str, types.ModuleType | None] = {}
        modules = {
            "recbole": types.ModuleType("recbole"),
            "recbole.config": types.ModuleType("recbole.config"),
            "recbole.config.configurator": types.ModuleType("recbole.config.configurator"),
            "recbole.utils": types.ModuleType("recbole.utils"),
            "recbole.utils.enum_type": types.ModuleType("recbole.utils.enum_type"),
            "recbole.model": types.ModuleType("recbole.model"),
            "recbole.model.general_recommender": types.ModuleType("recbole.model.general_recommender"),
            "recbole.model.general_recommender.lightgcn": types.ModuleType(
                "recbole.model.general_recommender.lightgcn"
            ),
        }
        modules["recbole.config.configurator"].Config = _StubConfig
        modules["recbole.utils.enum_type"].ModelType = _StubModelType
        modules["recbole.utils.enum_type"].InputType = _StubInputType
        modules["recbole.utils.enum_type"].EvaluatorType = _StubEvaluatorType
        modules["recbole.model.general_recommender.lightgcn"].LightGCN = _StubModel
        for name, module in modules.items():
            self._previous[name] = sys.modules.get(name)
            sys.modules[name] = module

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, module in self._previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
