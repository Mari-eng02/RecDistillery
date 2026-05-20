"""
Compare recommendation lists from two exports.

Supported input formats:
1) Triplet format: user<TAB>item<TAB>score (one row per recommendation)
2) Top-k compact format: user<TAB>item1,item2,item3,...

Usage:
    python scripts/recdistill/compare_recommendation_lists.py \
        --reference path/to/reference_recs.tsv \
        --candidate path/to/recdistill_recs.tsv \
        --top-k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_triplet_rows(lines: Iterable[str], top_k: int) -> dict[int, list[int]]:
    by_user: dict[int, list[int]] = {}
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        user = int(parts[0])
        item = int(parts[1])
        current = by_user.setdefault(user, [])
        if len(current) < top_k:
            current.append(item)
    return by_user


def _parse_compact_topk(lines: Iterable[str], top_k: int) -> dict[int, list[int]]:
    by_user: dict[int, list[int]] = {}
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        user = int(parts[0])
        items = [int(value) for value in parts[1].split(",") if value != ""]
        by_user[user] = items[:top_k]
    return by_user


def _looks_like_triplet_tsv(lines: list[str]) -> bool:
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        if len(parts) >= 3:
            return True
        if "," not in parts[1]:
            return True
        return False
    return True


def _load_recommendations(path: Path, top_k: int) -> dict[int, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if _looks_like_triplet_tsv(lines):
        return _parse_triplet_rows(lines, top_k)
    return _parse_compact_topk(lines, top_k)


def _jaccard(list_a: list[int], list_b: list[int]) -> float:
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _recall_at_k(reference: list[int], predicted: list[int]) -> float:
    ref = set(reference)
    if not ref:
        return 1.0 if not predicted else 0.0
    return len(ref & set(predicted)) / len(ref)


def compare_recommendations(
    reference_path: Path,
    candidate_path: Path,
    top_k: int,
    max_examples: int = 5,
) -> dict:
    reference = _load_recommendations(reference_path, top_k=top_k)
    candidate = _load_recommendations(candidate_path, top_k=top_k)

    common_users = sorted(set(reference.keys()) & set(candidate.keys()))
    missing_in_candidate = sorted(set(reference.keys()) - set(candidate.keys()))
    extra_in_candidate = sorted(set(candidate.keys()) - set(reference.keys()))

    if not common_users:
        raise ValueError("No overlapping users between reference and candidate recommendation files.")

    exact_matches = 0
    jaccards = []
    recalls = []
    mismatches: list[dict] = []

    for user in common_users:
        ref = reference[user][:top_k]
        pred = candidate[user][:top_k]
        if ref == pred:
            exact_matches += 1
        else:
            if len(mismatches) < max_examples:
                mismatches.append(
                    {
                        "user": user,
                        "reference_topk": ref,
                        "candidate_topk": pred,
                    }
                )
        jaccards.append(_jaccard(ref, pred))
        recalls.append(_recall_at_k(ref, pred))

    result = {
        "top_k": top_k,
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "reference_users": len(reference),
        "candidate_users": len(candidate),
        "common_users": len(common_users),
        "missing_in_candidate": len(missing_in_candidate),
        "extra_in_candidate": len(extra_in_candidate),
        "exact_sequence_matches": exact_matches,
        "exact_sequence_match_rate": exact_matches / len(common_users),
        "average_jaccard": float(np.mean(jaccards)),
        "min_jaccard": float(np.min(jaccards)),
        "average_recall_at_k": float(np.mean(recalls)),
        "min_recall_at_k": float(np.min(recalls)),
        "mismatch_examples": mismatches,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two recommendation list exports.")
    parser.add_argument("--reference", required=True, help="Path to reference recommendations file")
    parser.add_argument("--candidate", required=True, help="Path to candidate recommendations file")
    parser.add_argument("--top-k", type=int, default=20, help="Truncate both lists to top-k")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum mismatch examples to print/store")
    args = parser.parse_args()

    result = compare_recommendations(
        reference_path=Path(args.reference),
        candidate_path=Path(args.candidate),
        top_k=args.top_k,
        max_examples=args.max_examples,
    )

    print("\n" + "=" * 80)
    print("Recommendation Comparison")
    print("=" * 80)
    print(f"Reference users: {result['reference_users']}")
    print(f"Candidate users: {result['candidate_users']}")
    print(f"Common users: {result['common_users']}")
    print(f"Missing in candidate: {result['missing_in_candidate']}")
    print(f"Extra in candidate: {result['extra_in_candidate']}")
    print(f"Exact sequence matches: {result['exact_sequence_matches']}/{result['common_users']}")
    print(f"Exact sequence match rate: {result['exact_sequence_match_rate']:.6f}")
    print(f"Average Jaccard: {result['average_jaccard']:.6f}")
    print(f"Min Jaccard: {result['min_jaccard']:.6f}")
    print(f"Average Recall@{result['top_k']}: {result['average_recall_at_k']:.6f}")
    print(f"Min Recall@{result['top_k']}: {result['min_recall_at_k']:.6f}")

    if result["mismatch_examples"]:
        print("\nMismatch examples:")
        for sample in result["mismatch_examples"]:
            print(f"  User {sample['user']}")
            print(f"    reference: {sample['reference_topk']}")
            print(f"    candidate: {sample['candidate_topk']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {output_path}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
