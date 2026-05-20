"""
Aggregate evaluate_teacher outputs into a single TSV.

Input files (default pattern):
    results/<dataset>/teacher/<framework>/<teacher>/best/perf/*_eval_top*.json

Each JSON is expected to follow the structure produced by:
    scripts/recdistill/evaluate_teacher.py

Output:
    One TSV row per split (val/test) per JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_context(eval_json_path: Path, results_root: Path) -> dict[str, str]:
    rel = eval_json_path.relative_to(results_root)
    # Current layout: <dataset>/teacher/<framework>/<teacher>/best/perf/<file>
    # Older layout:   <dataset>/teacher/<teacher>/best/perf/<file>
    parts = rel.parts
    if len(parts) >= 7 and parts[1] == "teacher" and parts[5] == "perf":
        return {
            "dataset": parts[0],
            "framework": parts[2],
            "teacher": parts[3],
            "phase": parts[4],
            "eval_json_file": str(eval_json_path),
        }
    return {
        "dataset": parts[0] if len(parts) > 0 else "",
        "framework": "",
        "teacher": parts[2] if len(parts) > 2 else "",
        "phase": parts[3] if len(parts) > 3 else "",
        "eval_json_file": str(eval_json_path),
    }


def _row_from_split(payload: dict, split: str, context: dict[str, str]) -> dict[str, str]:
    split_metrics = payload.get(f"{split}_metrics", {}) if isinstance(payload, dict) else {}
    leak_key = f"leaked_users_{split}"
    return {
        **context,
        "model": _safe_str(payload.get("model")),
        "embedding_dim": _safe_str(payload.get("embedding_dim")),
        "top_k": _safe_str(payload.get("top_k")),
        "split": split,
        # Columns aligned with historical metric headers for easy comparison
        "nDCGRendle2020": _safe_str(split_metrics.get("ndcg")),
        "Recall": _safe_str(split_metrics.get("recall")),
        "Precision": _safe_str(split_metrics.get("precision")),
        "HR": _safe_str(split_metrics.get("hr")),
        # Extra diagnostics
        "users": _safe_str(split_metrics.get("users")),
        "leaked_users": _safe_str(payload.get(leak_key)),
        "teacher_path": _safe_str(payload.get("teacher_path")),
        "train_interactions": _safe_str(payload.get("train_interactions")),
        "val_interactions": _safe_str(payload.get("val_interactions")),
        "test_interactions": _safe_str(payload.get("test_interactions")),
    }


def aggregate(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    patterns = [
        "*/teacher/*/*/best/perf/*_eval_top*.json",
        "*/teacher/*/best/perf/*_eval_top*.json",
    ]
    eval_files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for eval_json in sorted(results_root.glob(pattern)):
            resolved = eval_json.resolve()
            if resolved not in seen:
                eval_files.append(eval_json)
                seen.add(resolved)

    for eval_json in eval_files:
        try:
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        context = _extract_context(eval_json_path=eval_json, results_root=results_root)
        rows.append(_row_from_split(payload, split="val", context=context))
        rows.append(_row_from_split(payload, split="test", context=context))
    return rows


def write_tsv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("dataset\tteacher\tphase\teval_json_file\n")
        return

    preferred = [
        "dataset",
        "framework",
        "teacher",
        "phase",
        "model",
        "embedding_dim",
        "top_k",
        "split",
        "nDCGRendle2020",
        "Recall",
        "Precision",
        "HR",
        "users",
        "leaked_users",
        "eval_json_file",
        "teacher_path",
        "train_interactions",
        "val_interactions",
        "test_interactions",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    remaining = sorted(k for k in keys if k not in preferred)
    fieldnames = [k for k in preferred if k in keys] + remaining

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluate_teacher metrics into one TSV.")
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument(
        "--output",
        default="results/teacher_eval_perf_aggregated.tsv",
        help="Output TSV path",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_path = Path(args.output)
    rows = aggregate(results_root=results_root)
    write_tsv(rows=rows, output_path=output_path)
    print(f"Aggregated {len(rows)} row(s) into: {output_path}")


if __name__ == "__main__":
    main()
