"""
Aggregate fetched evaluation outputs into a single TSV.

Input files:
    fetched/<distiller>/<backbone>/<dataset>/tracked/<run_id>/perf/*_eval_top*.json

Each JSON is expected to follow the structure produced by:
    scripts/recdistill/evaluate_students.py

Output:
    One TSV row per split (val/test) per JSON file.
    fetched/fetched_eval_perf_aggregated.tsv

Usage:
    python scripts/postscript/aggregate_fetched_eval_perf.py
    python scripts/postscript/aggregate_fetched_eval_perf.py --fetched-root fetched --output custom.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_context(eval_json_path: Path, fetched_root: Path) -> dict[str, str]:
    rel = eval_json_path.relative_to(fetched_root)
    parts = rel.parts

    # Expected layout:
    # <distiller>/<backbone>/<dataset>/tracked/<run_id>/perf/<file>
    if len(parts) >= 7 and parts[3] == "tracked" and parts[5] == "perf":
        return {
            "distiller": parts[0],
            "backbone": parts[1],
            "dataset": parts[2],
            "run_id": parts[4],
            "phase": parts[3],
            "eval_json_file": str(eval_json_path),
        }

    return {"eval_json_file": str(eval_json_path)}


def _row_from_split(payload: dict[str, Any], split: str, context: dict[str, str]) -> dict[str, str]:
    split_metrics = payload.get(f"{split}_metrics", {}) if isinstance(payload, dict) else {}
    leak_key = f"leaked_users_{split}"

    return {
        **context,
        "kind": _safe_str(payload.get("kind")),
        "model": _safe_str(payload.get("student_model") or payload.get("model")),
        "teacher_model": _safe_str(payload.get("teacher_model")),
        "student_framework": _safe_str(payload.get("student_framework")),
        "artifact_path": _safe_str(payload.get("artifact_path")),
        "embedding_dim": _safe_str(payload.get("embedding_dim")),
        "top_k": _safe_str(payload.get("top_k")),
        "split": split,
        "precision": _safe_str(split_metrics.get("precision")),
        "recall": _safe_str(split_metrics.get("recall")),
        "ndcg": _safe_str(split_metrics.get("ndcg")),
        "hr": _safe_str(split_metrics.get("hr")),
        "users": _safe_str(split_metrics.get("users")),
        "leaked_users": _safe_str(payload.get(leak_key)),
        "train_interactions": _safe_str(payload.get("train_interactions")),
        "val_interactions": _safe_str(payload.get("val_interactions")),
        "test_interactions": _safe_str(payload.get("test_interactions")),
    }


def aggregate(fetched_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    eval_files = sorted(fetched_root.glob("*/*/*/tracked/*/perf/*_eval_top*.json"))

    for eval_json in eval_files:
        try:
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: could not parse {eval_json}: {exc}")
            continue

        if not isinstance(payload, dict):
            continue

        context = _extract_context(eval_json_path=eval_json, fetched_root=fetched_root)
        context["distiller"] = _safe_str(payload.get("distiller") or context.get("distiller", ""))
        context["dataset"] = _safe_str(payload.get("dataset") or context.get("dataset", ""))
        context["backbone"] = _safe_str(
            context.get("backbone") or payload.get("student_model") or payload.get("model", "")
        ).lower()

        rows.append(_row_from_split(payload, split="val", context=context))
        rows.append(_row_from_split(payload, split="test", context=context))

    return rows


def write_tsv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("distiller\tbackbone\tdataset\trun_id\tphase\teval_json_file\n")
        return

    preferred = [
        "distiller",
        "backbone",
        "dataset",
        "run_id",
        "phase",
        "kind",
        "model",
        "teacher_model",
        "student_framework",
        "embedding_dim",
        "top_k",
        "split",
        "precision",
        "recall",
        "ndcg",
        "hr",
        "users",
        "leaked_users",
        "artifact_path",
        "eval_json_file",
        "train_interactions",
        "val_interactions",
        "test_interactions",
    ]

    keys = set()
    for row in rows:
        keys.update(row.keys())

    remaining = sorted(key for key in keys if key not in preferred)
    fieldnames = [key for key in preferred if key in keys] + remaining

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fetched evaluation metrics into one TSV.")
    parser.add_argument("--fetched-root", default="fetched", help="Root fetched directory")
    parser.add_argument("--output", default="fetched/fetched_eval_perf_aggregated.tsv", help="Output TSV path")
    args = parser.parse_args()

    fetched_root = Path(args.fetched_root)
    if not fetched_root.exists():
        raise FileNotFoundError(f"Fetched root not found: {fetched_root}")

    print("Aggregating fetched evaluation metrics...")
    print(f"  Fetched root: {fetched_root}")

    rows = aggregate(fetched_root)
    print(f"  Found {len(rows)} rows from evaluation files")

    write_tsv(rows, Path(args.output))
    print(f"✓ Aggregation complete: {args.output}")


if __name__ == "__main__":
    main()
