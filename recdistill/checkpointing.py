from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from recdistill.tracking import utc_now_iso


STUDENT_CHECKPOINT_FORMAT = "recdistill.student.v1"


def config_hash(config: dict[str, Any] | None) -> str | None:
    if config is None:
        return None
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit(repo_root: Path | None = None) -> str | None:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def enrich_student_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    enriched = dict(payload)
    enriched.setdefault("format_version", STUDENT_CHECKPOINT_FORMAT)
    enriched.setdefault("created_at_utc", utc_now_iso())
    enriched.setdefault("config_hash", config_hash(config))
    enriched.setdefault("git_commit", current_git_commit())
    enriched.setdefault("dataset", config.get("dataset"))
    enriched.setdefault("teacher", config.get("teacher_model"))
    enriched.setdefault("student", config.get("student_backbone"))
    enriched.setdefault("distiller", _distiller_from_config(config))
    metadata = dict(enriched.get("metadata") or {})
    metadata.setdefault("format_version", enriched["format_version"])
    metadata.setdefault("created_at_utc", enriched["created_at_utc"])
    metadata.setdefault("config_hash", enriched["config_hash"])
    metadata.setdefault("git_commit", enriched["git_commit"])
    metadata.setdefault("dataset", enriched.get("dataset"))
    metadata.setdefault("teacher", enriched.get("teacher"))
    metadata.setdefault("student", enriched.get("student"))
    metadata.setdefault("distiller", enriched.get("distiller"))
    enriched["metadata"] = metadata
    return enriched


def save_student_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_student_checkpoint_payload(payload)
    torch.save(enriched, checkpoint_path)
    return enriched


def load_student_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    if "student_state_dict" not in payload:
        raise KeyError(f"`student_state_dict` missing in checkpoint: {path}")
    return payload


def _distiller_from_config(config: dict[str, Any]) -> str | None:
    names = []
    if float(config.get("lambda_de") or 0.0) > 0.0:
        names.append("DE")
    if float(config.get("lambda_rrd") or 0.0) > 0.0:
        names.append("RRD")
    if float(config.get("lambda_unkd") or 0.0) > 0.0:
        names.append("UNKD")
    if float(config.get("lambda_td") or 0.0) > 0.0:
        names.append(str(config.get("td_type") or "TD").upper())
    return "_".join(names) if names else None
