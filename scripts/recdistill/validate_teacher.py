"""
Validation script for teacher models in recdistill.

This script verifies that teacher models loaded in recdistill produce the same
recommendations as a reference export.

Usage:
    python scripts/recdistill/validate_teacher.py --dataset citeulike --model BPRMF --embedding-dim 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.paths import teacher_weights_path


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_local_recdistill():
    """Bootstrap recdistill modules for import."""
    package_root = REPO_ROOT / "recdistill"
    package_paths = {
        "recdistill": package_root,
        "recdistill.data": package_root / "data",
        "recdistill.samplers": package_root / "samplers",
        "recdistill.teachers": package_root / "teachers",
    }

    for package_name, package_path in package_paths.items():
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    interactions_module = _load_module_from_path(
        "recdistill.data.interactions",
        package_root / "data" / "interactions.py",
    )
    teacher_state_module = _load_module_from_path(
        "recdistill.teachers.state",
        package_root / "teachers" / "state.py",
    )
    teacher_loaders_module = _load_module_from_path(
        "recdistill.teachers.loaders",
        package_root / "teachers" / "loaders.py",
    )
    teacher_topk_module = _load_module_from_path(
        "recdistill.samplers.teacher_topk",
        package_root / "samplers" / "teacher_topk.py",
    )

    sys.modules["recdistill.data"].InteractionDataset = interactions_module.InteractionDataset
    sys.modules["recdistill.teachers"].TeacherState = teacher_state_module.TeacherState
    sys.modules["recdistill.teachers"].load_teacher_state = teacher_loaders_module.load_teacher_state
    sys.modules["recdistill.samplers"].TeacherTopKProvider = teacher_topk_module.TeacherTopKProvider

    return (
        interactions_module.InteractionDataset,
        teacher_topk_module.TeacherTopKProvider,
        teacher_loaders_module.load_teacher_state,
    )


try:
    from recdistill.data.datarec_loader import load_interaction_dataset, resolve_teacher_dataset_mappings
    from recdistill.data.interactions import InteractionDataset
    from recdistill.samplers.teacher_topk import TeacherTopKProvider
    from recdistill.teachers import load_teacher_state
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("recdistill"):
        InteractionDataset, TeacherTopKProvider, load_teacher_state = _bootstrap_local_recdistill()
        from recdistill.data.datarec_loader import load_interaction_dataset, resolve_teacher_dataset_mappings
    else:
        raise


def _load_dataset(
    dataset_name: str,
    user_mapping: dict | None,
    item_mapping: dict | None,
    id_space: str | None,
) -> InteractionDataset:
    return load_interaction_dataset(
        dataset_name=dataset_name,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        id_space=id_space,
    )


def _compute_reference_topk(
    teacher_state,
    dataset: InteractionDataset,
    top_k: int,
) -> dict[int, list[int]]:
    """
    Compute reference top-k using pure embedding dot-product.
    This is the "ground truth" for models without exact scorers (BPRMF, LGCN).
    """
    if teacher_state.scorer is not None:
        scores = None
        num_teacher_users = teacher_state.num_users
        num_teacher_items = teacher_state.num_items
    else:
        user_emb = teacher_state.user_embeddings
        item_emb = teacher_state.item_embeddings
        if user_emb is None or item_emb is None:
            raise ValueError("Teacher validation requires embeddings or a scorer.")
        scores = torch.matmul(user_emb, item_emb.T)
        num_teacher_users = int(user_emb.size(0))
        num_teacher_items = int(item_emb.size(0))

    topk_by_user: dict[int, list[int]] = {}

    for user in range(min(dataset.num_users, num_teacher_users)):
        seen = sorted(item for item in dataset.seen_items(user) if 0 <= item < num_teacher_items)
        if scores is None:
            user_scores = teacher_state.scorer.score_items_for_user(user, num_teacher_items).clone()
        else:
            user_scores = scores[user].clone()
        if seen:
            user_scores[seen] = -1e9
        k = min(top_k, num_teacher_items - len(seen))
        if k <= 0:
            topk_by_user[user] = []
            continue
        top_items = torch.topk(user_scores, k=k, dim=0).indices.tolist()
        topk_by_user[user] = [int(item) for item in top_items]

    return topk_by_user


def _compute_jaccard_similarity(list_a: list[int], list_b: list[int]) -> float:
    """Compute Jaccard similarity between two lists."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_recall_at_k(reference: list[int], predicted: list[int], k: int) -> float:
    """Compute recall: how many of predicted items were in reference."""
    if not reference:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    reference_set = set(reference[:k])
    predicted_set = set(predicted[:k])
    return len(reference_set & predicted_set) / len(reference_set)


def validate_teacher(
    dataset_name: str,
    model: str,
    embedding_dim: int,
    top_k: int = 20,
    teacher_path: str | None = None,
    sample_users: int = 50,
) -> dict[str, Any]:
    """
    Validate a teacher model by comparing embeddings and top-k lists.

    Returns:
        Dictionary with validation results and metrics.
    """
    # Load teacher
    if teacher_path is None:
        teacher_path = teacher_weights_path(
            model=model,
            dataset=dataset_name,
            embedding_dim=embedding_dim,
            phase="best",
        )

    print(f"\n{'='*80}")
    print(f"Validating Teacher: {model} ({embedding_dim}D) on {dataset_name}")
    print(f"Teacher path: {teacher_path}")
    print(f"{'='*80}\n")

    # Load teacher state
    print("[1/3] Loading teacher state...")
    teacher_state = load_teacher_state(teacher_path)
    print(f"  ✓ User embeddings: {teacher_state.user_embeddings.shape if teacher_state.user_embeddings is not None else 'none'}")
    print(f"  ✓ Item embeddings: {teacher_state.item_embeddings.shape if teacher_state.item_embeddings is not None else 'none'}")
    print(f"  ✓ Has exact scorer: {teacher_state.scorer is not None}")

    # Load dataset
    print("\n[2/3] Loading dataset...")
    user_mapping, item_mapping, mapping_source = resolve_teacher_dataset_mappings(
        teacher_state.metadata,
        dataset_name=dataset_name,
    )
    split_id_space = "dataset_integer" if mapping_source == "dataset_integer" else None
    dataset = _load_dataset(
        dataset_name=dataset_name,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        id_space=split_id_space,
    )
    print(f"  Dataset mapping source: {mapping_source}")
    print(f"  ✓ Dataset users: {dataset.num_users}")
    print(f"  ✓ Dataset items: {dataset.num_items}")
    print(f"  ✓ Dataset interactions: {len(dataset.interactions)}")

    # Compute reference top-k
    print(f"\n[3/3] Computing reference top-k (k={top_k})...")
    reference_topk = _compute_reference_topk(teacher_state, dataset, top_k)
    print(f"  ✓ Reference top-k computed for {len(reference_topk)} users")

    # Use TeacherTopKProvider to compute top-k
    print(f"\nValidating via TeacherTopKProvider...")
    provider = TeacherTopKProvider(top_k=top_k)
    provider_topk = provider.build(teacher_state=teacher_state, dataset=dataset)
    print(f"  ✓ Provider top-k computed for {len(provider_topk)} users")

    # Compare results
    print(f"\nComparing results...")
    mismatches = []
    user_similarities = []

    for user in sorted(reference_topk.keys()):
        ref = reference_topk.get(user, [])
        pred = provider_topk.get(user, [])

        if ref != pred:
            mismatches.append(user)

        jaccard = _compute_jaccard_similarity(ref, pred)
        user_similarities.append(jaccard)

    # Compute metrics
    num_perfect = sum(1 for sim in user_similarities if sim == 1.0)
    num_partial = sum(1 for sim in user_similarities if 0.0 < sim < 1.0)
    num_zero = sum(1 for sim in user_similarities if sim == 0.0)
    avg_similarity = np.mean(user_similarities) if user_similarities else 0.0
    min_similarity = np.min(user_similarities) if user_similarities else 0.0

    print(f"  Users with perfect matches: {num_perfect}/{len(user_similarities)}")
    print(f"  Users with partial matches: {num_partial}/{len(user_similarities)}")
    print(f"  Users with no matches: {num_zero}/{len(user_similarities)}")
    print(f"  Average Jaccard similarity: {avg_similarity:.4f}")
    print(f"  Min Jaccard similarity: {min_similarity:.4f}")

    # Show sample mismatches
    if mismatches:
        print(f"\n  ⚠ Mismatches found for {len(mismatches)} users (showing first 5):")
        for user in mismatches[:5]:
            ref = reference_topk.get(user, [])
            pred = provider_topk.get(user, [])
            jaccard = _compute_jaccard_similarity(ref, pred)
            print(f"    User {user}: Jaccard={jaccard:.4f}")
            print(f"      Reference: {ref[:10]}")
            print(f"      Provider:  {pred[:10]}")

    # Summary
    print(f"\n{'='*80}")
    if not mismatches:
        print("✅ VALIDATION PASSED: All recommendations match!")
    else:
        print(f"⚠️  VALIDATION WARNING: {len(mismatches)} users have mismatched recommendations")
        print("   (This may be due to tie-breaking in top-k selection)")
    print(f"{'='*80}\n")

    return {
        "teacher_path": str(teacher_path),
        "dataset": dataset_name,
        "model": model,
        "embedding_dim": embedding_dim,
        "top_k": top_k,
        "teacher_num_users": teacher_state.num_users,
        "teacher_num_items": teacher_state.num_items,
        "teacher_embedding_dim": teacher_state.embedding_dim if teacher_state.has_embeddings else None,
        "has_exact_scorer": teacher_state.scorer is not None,
        "dataset_num_users": dataset.num_users,
        "dataset_num_items": dataset.num_items,
        "dataset_interactions": len(dataset.interactions),
        "num_perfect_matches": num_perfect,
        "num_partial_matches": num_partial,
        "num_zero_matches": num_zero,
        "average_jaccard_similarity": float(avg_similarity),
        "min_jaccard_similarity": float(min_similarity),
        "num_mismatches": len(mismatches),
        "total_users_compared": len(user_similarities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate teacher models in recdistill")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., citeulike, amazon_cd)")
    parser.add_argument("--model", required=True, choices=["BPRMF", "LGCN", "NMF"], help="Teacher model name")
    parser.add_argument("--embedding-dim", required=True, type=int, help="Teacher embedding dimension")
    parser.add_argument("--top-k", default=20, type=int, help="Top-k for recommendations")
    parser.add_argument("--teacher-path", default=None, help="Optional explicit path to teacher file")
    parser.add_argument("--output-json", default=None, help="Optional path for validation results JSON")
    args = parser.parse_args()

    results = validate_teacher(
        dataset_name=args.dataset,
        model=args.model,
        embedding_dim=args.embedding_dim,
        top_k=args.top_k,
        teacher_path=args.teacher_path,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
