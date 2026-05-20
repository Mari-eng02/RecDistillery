from __future__ import annotations

from pathlib import Path

import torch

from recdistill.teachers.serialization import (
    SUPPORTED_TEACHER_FORMAT_VERSIONS,
    load_teacher_payload,
    teacher_state_from_payload,
)
from recdistill.teachers.source import TeacherSource
from recdistill.teachers.state import TeacherState


class NativeTeacherAdapter:
    name = "recdistill"

    def can_load(self, source: TeacherSource) -> bool:
        if _matches(source.framework) or _matches(source.format):
            return True
        if source.path is None:
            return False
        try:
            payload = load_teacher_payload(source.path)
        except Exception:
            return False
        return payload.get("format_version") in SUPPORTED_TEACHER_FORMAT_VERSIONS

    def load(self, source: TeacherSource, device: torch.device | str | None = None) -> TeacherState:
        if source.path is None:
            raise ValueError("Native teacher adapter requires source.path.")
        state = teacher_state_from_payload(load_teacher_payload(Path(source.path)))
        if source.model_name:
            state.metadata.setdefault("model_name", source.model_name)
        if source.metadata:
            state.metadata.update(source.metadata)
        if device is not None:
            return state.to(device)
        return state


def _matches(value: str | None) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in {
        "recdistill",
        "recdistill_native",
        "recdistill_teacher",
        "teacher_v1",
    }
