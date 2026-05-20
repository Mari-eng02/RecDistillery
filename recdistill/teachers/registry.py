from __future__ import annotations

import importlib
from typing import Protocol

import torch

from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import TeacherState


class TeacherAdapter(Protocol):
    name: str

    def can_load(self, source: TeacherSource) -> bool: ...

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState: ...


_ADAPTERS: dict[str, TeacherAdapter] = {}


def register_teacher_adapter(adapter: TeacherAdapter, *aliases: str) -> None:
    names = (adapter.name, *aliases)
    for name in names:
        _ADAPTERS[_normalize(name)] = adapter


def available_teacher_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def resolve_teacher_adapter(source: TeacherSource) -> TeacherAdapter:
    if source.adapter:
        return _load_adapter_object(source.adapter)

    for key in (source.format, source.framework):
        normalized = _normalize(key)
        if normalized != "auto" and normalized in _ADAPTERS:
            return _ADAPTERS[normalized]

    for adapter in dict.fromkeys(_ADAPTERS.values()):
        if adapter.can_load(source):
            return adapter

    raise ValueError(
        "No teacher adapter can load the provided source. "
        f"framework={source.framework!r}, format={source.format!r}, path={source.path!r}. "
        f"Available adapters: {', '.join(available_teacher_adapters())}"
    )


def load_teacher_state(source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
    adapter = resolve_teacher_adapter(source)
    return adapter.load(source, device=device)


def _normalize(value: str | None) -> str:
    return str(value or "auto").strip().lower().replace("-", "_")


def _load_adapter_object(import_path: str) -> TeacherAdapter:
    module_name, _, attr = import_path.partition(":")
    if not attr:
        module_name, _, attr = import_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            "Custom teacher adapter must be an import path like "
            "'package.module:AdapterClass' or 'package.module.adapter'."
        )
    module = importlib.import_module(module_name)
    candidate = getattr(module, attr)
    adapter = candidate() if isinstance(candidate, type) else candidate
    if not hasattr(adapter, "can_load") or not hasattr(adapter, "load"):
        raise TypeError(f"Custom adapter {import_path!r} does not implement TeacherAdapter.")
    return adapter
