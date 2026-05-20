from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherSource:
    path: Path | None = None
    framework: str = "auto"
    format: str = "auto"
    model_name: str | None = None
    adapter: str | None = None
    user_embeddings_path: Path | None = None
    item_embeddings_path: Path | None = None
    score_matrix_path: Path | None = None
    topk_items_path: Path | None = None
    topk_scores_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        framework: str = "auto",
        format: str = "auto",
        model_name: str | None = None,
        adapter: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TeacherSource":
        return cls(
            path=Path(path),
            framework=framework,
            format=format,
            model_name=model_name,
            adapter=adapter,
            metadata=metadata or {},
        )
