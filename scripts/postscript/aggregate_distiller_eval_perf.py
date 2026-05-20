"""
Aggregate student evaluation outputs into a single TSV.

Input files (default patterns):
    results/recdistill/{distiller}/{teacher}/{student}/{dataset}/{strategy}/perf/*_eval_top*.json

Each JSON is expected to follow the structure produced by:
    scripts/recdistill/evaluate_students.py

Output:
    One TSV row per split (val/test) per JSON file.
    results/recdistill/students_eval_perf_aggregated.tsv

Usage:
    python scripts/postscript/aggregate_distiller_eval_perf.py
    python scripts/postscript/aggregate_distiller_eval_perf.py --results-root results --output custom_output.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _safe_str(value) -> str:
    """Convert value to string, handling None"""
    if value is None:
        return ""
    return str(value)


def _extract_context(eval_json_path: Path, results_root: Path) -> dict[str, str]:
    """Extract metadata from supported result layouts."""
    rel = eval_json_path.relative_to(results_root)
    parts = rel.parts

    # Distilled-student evaluation layout:
    # recdistill/<distiller>/<teacher>/<student>/<dataset>/<strategy>/perf/<file>
    if len(parts) >= 8 and parts[0] == "recdistill" and parts[6] == "perf":
        return {
            "distiller": parts[1],
            "backbone": parts[3],
            "teacher": parts[2].upper(),
            "dataset": parts[4],
            "student_model": f"{parts[1]}_{parts[2].upper()}_to_{parts[3].upper()}",
            "phase": parts[5],
            "eval_json_file": str(eval_json_path),
        }

    return {"eval_json_file": str(eval_json_path)}


def _row_from_split(payload: dict, split: str, context: dict[str, str]) -> dict[str, str]:
    """Extract one row from a split (val or test)"""
    split_metrics = payload.get(f"{split}_metrics", {}) if isinstance(payload, dict) else {}
    leak_key = f"leaked_users_{split}"

    return {
        **context,
        "model": _safe_str(payload.get("student_model") or payload.get("model")),
        "teacher_model": _safe_str(payload.get("teacher_model")),
        "student_framework": _safe_str(payload.get("student_framework")),
        "artifact_path": _safe_str(payload.get("artifact_path")),
        "embedding_dim": _safe_str(payload.get("embedding_dim")),
        "top_k": _safe_str(payload.get("top_k")),
        "split": split,
        # Metrics aligned with teacher evaluation for easy comparison
        "nDCGRendle2020": _safe_str(split_metrics.get("ndcg")),
        "Recall": _safe_str(split_metrics.get("recall")),
        "Precision": _safe_str(split_metrics.get("precision")),
        "HR": _safe_str(split_metrics.get("hr")),
        # Extra diagnostics
        "users": _safe_str(split_metrics.get("users")),
        "leaked_users": _safe_str(payload.get(leak_key)),
        "train_interactions": _safe_str(payload.get("train_interactions")),
        "val_interactions": _safe_str(payload.get("val_interactions")),
        "test_interactions": _safe_str(payload.get("test_interactions")),
    }


def aggregate(results_root: Path) -> list[dict[str, str]]:
    """Aggregate all student evaluation JSON files"""
    rows: list[dict[str, str]] = []

    patterns = [
        "recdistill/*/*/*/*/*/perf/*_eval_top*.json",
    ]
    eval_files = []
    seen_paths = set()
    for pattern in patterns:
        for eval_json in sorted(results_root.glob(pattern)):
            resolved = eval_json.resolve()
            if resolved in seen_paths:
                continue
            eval_files.append(eval_json)
            seen_paths.add(resolved)

    for eval_json in eval_files:
        try:
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Could not parse {eval_json}: {e}")
            continue

        if not isinstance(payload, dict):
            continue

        context = _extract_context(eval_json_path=eval_json, results_root=results_root)
        context["dataset"] = _safe_str(payload.get("dataset") or context.get("dataset", ""))
        context["distiller"] = _safe_str(payload.get("distiller") or context.get("distiller", ""))
        context["teacher"] = _safe_str(payload.get("teacher_model") or context.get("teacher", ""))
        context["backbone"] = _safe_str(
            context.get("backbone") or payload.get("student_model") or payload.get("model", "")
        ).lower()

        # Add one row per split
        rows.append(_row_from_split(payload, split="val", context=context))
        rows.append(_row_from_split(payload, split="test", context=context))

    return rows


def write_tsv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to TSV file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("distiller\tbackbone\tdataset\tphase\teval_json_file\n")
        return

    # Define column order with preferred metrics first
    preferred = [
        "distiller",
        "backbone",
        "teacher",
        "dataset",
        "student_model",
        "phase",
        "model",
        "teacher_model",
        "student_framework",
        "embedding_dim",
        "top_k",
        "split",
        "nDCGRendle2020",
        "Recall",
        "Precision",
        "HR",
        "users",
        "leaked_users",
        "artifact_path",
        "eval_json_file",
        "train_interactions",
        "val_interactions",
        "test_interactions",
    ]

    # Collect all keys
    keys = set()
    for row in rows:
        keys.update(row.keys())

    # Build final field order
    remaining = sorted(k for k in keys if k not in preferred)
    fieldnames = [k for k in preferred if k in keys] + remaining

    # Write TSV
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate student evaluation metrics into one TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default output to results/recdistill/students_eval_perf_aggregated.tsv
  python scripts/postscript/aggregate_distiller_eval_perf.py

  # Custom output path
  python scripts/postscript/aggregate_distiller_eval_perf.py \\
    --output results/student_metrics.tsv
        """,
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root results directory",
    )
    parser.add_argument(
        "--output",
        default="results/recdistill/students_eval_perf_aggregated.tsv",
        help="Output TSV path",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    print(f"Aggregating student evaluation metrics...")
    print(f"  Results root: {results_root}")

    rows = aggregate(results_root)

    print(f"  Found {len(rows)} rows from evaluation files")

    write_tsv(rows, Path(args.output))

    print(f"Aggregation complete: {args.output}")


if __name__ == "__main__":
    main()
