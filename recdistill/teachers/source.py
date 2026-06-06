from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherSource:
    """Input descriptor consumed by teacher import adapters.

    Attributes:
        path: Main checkpoint, prediction, or `.teacher` artifact path.
        framework: Framework hint such as `recbole`, `elliot`, or `external`.
        format: Format hint such as `checkpoint`, `predictions_json`, or
            `recbole_pth`.
        model_name: Optional model/backbone name stored in metadata.
        adapter: Optional explicit adapter import path.
        user_embeddings_path: Optional external user embedding file.
        item_embeddings_path: Optional external item embedding file.
        score_matrix_path: Optional dense score matrix file.
        topk_items_path: Optional top-k item matrix file.
        topk_scores_path: Optional top-k score matrix file.
        metadata: Extra provenance, mapping, and dataset information.
    """

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
        """Create a source descriptor from one primary artifact path."""
        return cls(
            path=Path(path),
            framework=framework,
            format=format,
            model_name=model_name,
            adapter=adapter,
            metadata=metadata or {},
        )
