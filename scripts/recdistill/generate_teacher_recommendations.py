"""
Generate and save teacher recommendations.

This script loads a teacher model and generates top-k recommendations for all users,
saving them in a format ready for distillation pipelines.

Usage:
    python scripts/recdistill/generate_teacher_recommendations.py \\
        --dataset citeulike --model BPRMF --embedding-dim 200 --top-k 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

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


def _load_dataset(dataset_name: str, user_mapping: dict | None, item_mapping: dict | None) -> InteractionDataset:
    return load_interaction_dataset(
        dataset_name=dataset_name,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save teacher recommendations")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, choices=["BPRMF", "LGCN", "NMF"], help="Teacher model name")
    parser.add_argument("--embedding-dim", required=True, type=int, help="Teacher embedding dimension")
    parser.add_argument("--top-k", default=20, type=int, help="Number of top recommendations per user")
    parser.add_argument("--teacher-path", default=None, help="Optional explicit path to teacher file")
    args = parser.parse_args()

    # Load teacher
    teacher_path = args.teacher_path or teacher_weights_path(
        model=args.model,
        dataset=args.dataset,
        embedding_dim=args.embedding_dim,
        phase="best",
    )

    print(f"\n{'='*70}")
    print(f"Generating recommendations for {args.model} ({args.embedding_dim}D)")
    print(f"Dataset: {args.dataset} | Top-k: {args.top_k}")
    print(f"{'='*70}\n")

    print(f"[1/3] Loading teacher from: {teacher_path}")
    teacher_state = load_teacher_state(teacher_path)
    print(f"  ✓ Teacher shape: {teacher_state.num_users} users × {teacher_state.num_items} items")
    print(f"  ✓ Embedding dimension: {teacher_state.embedding_dim if teacher_state.has_embeddings else 'none'}")

    print(f"\n[2/3] Loading dataset...")
    user_mapping, item_mapping, mapping_source = resolve_teacher_dataset_mappings(
        teacher_state.metadata,
        dataset_name=args.dataset,
    )
    print(f"  Dataset mapping source: {mapping_source}")
    dataset = _load_dataset(
        dataset_name=args.dataset,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    print(f"  ✓ Dataset: {dataset.num_users} users × {dataset.num_items} items")
    print(f"  ✓ Interactions: {len(dataset.interactions)}")

    print(f"\n[3/3] Generating top-{args.top_k} recommendations...")
    provider = TeacherTopKProvider(top_k=args.top_k)
    recommendations = provider.build(teacher_state=teacher_state, dataset=dataset)
    print(f"  ✓ Generated recommendations for {len(recommendations)} users")

    # Save recommendations
    output_dir = Path("results") / args.dataset / "teacher" / args.model / "best" / "recs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as TSV (user \t item1,item2,...)
    tsv_path = output_dir / f"{args.model}_{args.dataset}_{args.embedding_dim}_top{args.top_k}.tsv"
    with tsv_path.open("w", encoding="utf-8") as fp:
        for user in sorted(recommendations.keys()):
            items = recommendations[user]
            items_str = ",".join(str(item) for item in items)
            fp.write(f"{user}\t{items_str}\n")
    print(f"\n✅ Saved TSV: {tsv_path}")

    # Save as JSON
    json_path = output_dir / f"{args.model}_{args.dataset}_{args.embedding_dim}_top{args.top_k}.json"
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "top_k": args.top_k,
        "teacher_num_users": teacher_state.num_users,
        "teacher_num_items": teacher_state.num_items,
        "num_recommendations_generated": len(recommendations),
    }
    output_data = {
        "metadata": metadata,
        "recommendations": {str(user): items for user, items in recommendations.items()},
    }
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(output_data, fp, indent=2)
    print(f"✅ Saved JSON: {json_path}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
