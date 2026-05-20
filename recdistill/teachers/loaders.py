from __future__ import annotations

from pathlib import Path

import torch

from recdistill.teachers.registry import load_teacher_state as _load_teacher_state_from_source
from recdistill.teachers.registry import register_teacher_adapter
from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import TeacherState


def load_teacher(
    source: TeacherSource | str | Path,
    device: torch.device | str | None = None,
) -> TeacherState:
    if not isinstance(source, TeacherSource):
        source = TeacherSource.from_path(source)
    return _load_teacher_state_from_source(source, device=device)


def load_teacher_state(
    source: TeacherSource | str | Path,
    device: torch.device | str | None = None,
) -> TeacherState:
    """Load a teacher state from any registered RecDistill teacher source.

    This is the preferred public loader. It accepts either a TeacherSource or a
    direct artifact path and dispatches through the teacher adapter registry.
    """
    return load_teacher(source, device=device)


def register_default_teacher_adapters() -> None:
    from recdistill.teachers.adapters import (
        NativeTeacherAdapter,
        NumpyEmbeddingsTeacherAdapter,
    )

    register_teacher_adapter(NativeTeacherAdapter(), "recdistill_native", "teacher_v1", "teacher_v2")
    register_teacher_adapter(NumpyEmbeddingsTeacherAdapter(), "npz", "npy", "embeddings_npz", "embeddings_npy")


register_default_teacher_adapters()
