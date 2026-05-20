#!/usr/bin/env python3
"""
Aggregate tracked rerun summaries under `fetched/` into a single TSV.

The script scans:
    fetched/<distiller>/<backbone>/<dataset>/tracked/<run_id>/run_recap.json

and writes one row per distiller/backbone/dataset triplet to:
    fetched/tracked_results_summary.tsv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FETCHED_ROOT = REPO_ROOT / "fetched"
OUTPUT_TSV = FETCHED_ROOT / "tracked_results_summary.tsv"


def _select_recap_payload(recap_path: Path) -> dict[str, Any] | None:
    payload = json.loads(recap_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    return first


def _extract_triplet(recap_path: Path) -> tuple[str, str, str]:
    dataset = recap_path.parent.parent.parent.name
    backbone = recap_path.parent.parent.parent.parent.name
    distiller = recap_path.parent.parent.parent.parent.parent.name
    return distiller, backbone, dataset


def _latest_by_triplet(paths: list[Path]) -> dict[tuple[str, str, str], Path]:
    selected: dict[tuple[str, str, str], Path] = {}
    for path in sorted(paths):
        key = _extract_triplet(path)
        current = selected.get(key)
        if current is None:
            selected[key] = path
            continue
        if path.parent.name > current.parent.name:
            selected[key] = path
    return selected


def _build_rows(recap_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_paths = _latest_by_triplet(recap_paths)

    for (distiller, backbone, dataset), recap_path in sorted(selected_paths.items()):
        record = _select_recap_payload(recap_path)
        if record is None:
            continue
        rows.append(
            {
                "distiller": distiller,
                "backbone": backbone,
                "dataset": dataset,
                "training_time_sec": record.get("training_time_sec"),
                "epochs": record.get("final_epoch"),
                "best_epoch": record.get("best_epoch"),
                "best_selection_score": record.get("best_selection_score"),
                "status": record.get("status"),
                "recap_path": str(recap_path),
            }
        )

    return rows


def _write_tsv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "distiller",
        "backbone",
        "dataset",
        "training_time_sec",
        "epochs",
        "best_epoch",
        "best_selection_score",
        "status",
        "recap_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    recap_paths = sorted(FETCHED_ROOT.glob("*/*/*/tracked/*/run_recap.json"))
    rows = _build_rows(recap_paths)
    _write_tsv(rows, OUTPUT_TSV)
    print(f"Aggregated {len(rows)} row(s) into: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
