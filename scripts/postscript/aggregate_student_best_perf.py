"""
Aggregate baseline student best performance files into one TSV.

Default input pattern:
    results/students/<framework>/<student>/<dataset>/best/perf/*.tsv

Output:
    A single TSV with one row per recommender row found in each performance TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CUTOFF_RE = re.compile(r"cutoff_(\d+)")
REL_RE = re.compile(r"relthreshold_(-?\d+)")
TS_RE = re.compile(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")


def parse_perf_filename(name: str) -> dict[str, str]:
    cutoff_match = CUTOFF_RE.search(name)
    rel_match = REL_RE.search(name)
    ts_match = TS_RE.search(name)
    return {
        "cutoff": cutoff_match.group(1) if cutoff_match else "",
        "rel_threshold": rel_match.group(1) if rel_match else "",
        "run_timestamp": ts_match.group(1) if ts_match else "",
    }


def flatten_simple_dict(prefix: str, obj: dict) -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in obj.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[f"{prefix}{key}"] = "" if value is None else str(value)
    return flat


def load_best_json_meta(perf_dir: Path) -> dict[str, str]:
    candidates = sorted(perf_dir.glob("bestmodelparams*.json"))
    if not candidates:
        return {}

    best_file = candidates[-1]
    try:
        payload = json.loads(best_file.read_text(encoding="utf-8"))
    except Exception:
        return {"bestmodelparams_file": str(best_file)}

    meta: dict[str, str] = {"bestmodelparams_file": str(best_file)}
    if isinstance(payload, list) and payload:
        head = payload[0] if len(payload) > 0 and isinstance(payload[0], dict) else {}
        body = payload[1] if len(payload) > 1 and isinstance(payload[1], dict) else {}
        meta.update(flatten_simple_dict("default_", head))

        if isinstance(body, dict):
            if isinstance(body.get("meta"), dict):
                meta.update(flatten_simple_dict("meta_", body["meta"]))
            if isinstance(body.get("configuration"), dict):
                meta.update(flatten_simple_dict("conf_", body["configuration"]))
            if "recommender" in body:
                meta["best_recommender"] = str(body["recommender"])
    return meta


def extract_context(tsv_path: Path, results_root: Path) -> dict[str, str]:
    rel = tsv_path.relative_to(results_root)
    parts = rel.parts
    # Current layout: students/<framework>/<student>/<dataset>/<phase>/perf/<file>
    if len(parts) >= 7 and parts[0] == "students" and parts[5] == "perf":
        dataset = parts[3]
        framework = parts[1]
        student = parts[2]
        phase = parts[4]
    # Legacy layout: <dataset>/student/<framework>/<student>/<phase>/perf/<file>
    # Older layout:  <dataset>/student/<student>/<phase>/perf/<file>
    elif len(parts) >= 7 and parts[1] == "student" and parts[5] == "perf":
        dataset = parts[0]
        framework = parts[2]
        student = parts[3]
        phase = parts[4]
    else:
        dataset = parts[0] if len(parts) > 0 else ""
        framework = ""
        student = parts[2] if len(parts) > 2 else ""
        phase = parts[3] if len(parts) > 3 else ""
    return {
        "dataset": dataset,
        "framework": framework,
        "student": student,
        "phase": phase,
        "perf_file": str(tsv_path),
    }


def read_perf_rows(tsv_path: Path) -> list[dict[str, str]]:
    try:
        with tsv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            if not reader.fieldnames:
                return []
            rows: list[dict[str, str]] = []
            for row in reader:
                normalized = {
                    key.strip(): (value.strip() if isinstance(value, str) else "")
                    for key, value in row.items()
                    if key is not None
                }
                if not normalized:
                    continue
                rows.append(normalized)
            return rows
    except Exception:
        return []


def _is_recommendation_perf_tsv(tsv_path: Path) -> bool:
    name = tsv_path.name
    return name.startswith("rec_cutoff_") and name.endswith(".tsv")


def aggregate(results_root: Path) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    patterns = [
        "students/*/*/*/best/perf/*.tsv",
        "*/student/*/*/best/perf/*.tsv",
        "*/student/*/best/perf/*.tsv",
    ]
    tsv_files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for tsv_path in sorted(results_root.glob(pattern)):
            resolved = tsv_path.resolve()
            if resolved not in seen:
                tsv_files.append(tsv_path)
                seen.add(resolved)

    for tsv_path in tsv_files:
        if not _is_recommendation_perf_tsv(tsv_path):
            continue
        perf_rows = read_perf_rows(tsv_path)
        if not perf_rows:
            continue

        context = extract_context(tsv_path, results_root)
        filename_info = parse_perf_filename(tsv_path.name)
        json_meta = load_best_json_meta(tsv_path.parent)

        for perf_row in perf_rows:
            merged = {}
            merged.update(context)
            merged.update(filename_info)
            merged.update(json_meta)
            merged.update(perf_row)
            all_rows.append(merged)

    return all_rows


def write_tsv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("dataset\tstudent\tphase\tperf_file\n")
        return

    preferred = [
        "dataset",
        "framework",
        "student",
        "phase",
        "cutoff",
        "rel_threshold",
        "run_timestamp",
        "model",
        "nDCGRendle2020",
        "Recall",
        "Precision",
        "HR",
        "perf_file",
        "bestmodelparams_file",
        "best_recommender",
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
    parser = argparse.ArgumentParser(description="Aggregate student best performance files into a single TSV.")
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument(
        "--output",
        default="results/student_best_perf_aggregated.tsv",
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
