from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

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
    from recdistill.data.datarec_loader import load_interaction_dataset
    from recdistill.data.interactions import InteractionDataset
    from recdistill.samplers.teacher_topk import TeacherTopKProvider
    from recdistill.teachers import load_teacher_state
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("recdistill"):
        InteractionDataset, TeacherTopKProvider, load_teacher_state = _bootstrap_local_recdistill()
        from recdistill.data.datarec_loader import load_interaction_dataset
    else:
        raise


def _load_dataset(dataset_name: str, user_mapping: dict | None, item_mapping: dict | None) -> InteractionDataset:
    return load_interaction_dataset(
        dataset_name=dataset_name,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for teacher exports in recdistill")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. bookcrossing or amazon_cd")
    parser.add_argument("--teacher-framework", default="auto", help="Teacher framework namespace, e.g. recbole, elliot, lenskit")
    parser.add_argument("--teacher-model", "--model", dest="model", required=True, choices=["BPRMF", "LGCN", "NMF"], help="Teacher model name")
    parser.add_argument("--embedding-dim", required=True, type=int, help="Teacher embedding dimension used in filename")
    parser.add_argument("--top-k", default=20, type=int, help="Teacher top-k to materialize for smoke testing")
    parser.add_argument("--teacher-path", default=None, help="Optional explicit path to the exported teacher pickle")
    parser.add_argument("--output-json", default=None, help="Optional path for a smoke-test summary JSON")
    args = parser.parse_args()

    teacher_path = args.teacher_path or teacher_weights_path(
        model=args.model,
        dataset=args.dataset,
        embedding_dim=args.embedding_dim,
        phase="best",
        framework=args.teacher_framework,
    )
    state = load_teacher_state(teacher_path)

    dataset = _load_dataset(
        dataset_name=args.dataset,
        user_mapping=state.metadata.get("public_to_local_user_id"),
        item_mapping=state.metadata.get("public_to_local_item_id"),
    )
    provider = TeacherTopKProvider(top_k=args.top_k)
    teacher_topk = provider.build(teacher_state=state, dataset=dataset)

    sample_users = [user for user in sorted(teacher_topk.keys()) if teacher_topk[user]][:3]
    summary = {
        "teacher_path": str(teacher_path),
        "dataset": args.dataset,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "num_users": state.num_users,
        "num_items": state.num_items,
        "teacher_embedding_dim": state.embedding_dim if state.has_embeddings else None,
        "dataset_interactions": len(dataset.interactions),
        "top_k": args.top_k,
        "sample_topk": {str(user): teacher_topk[user][: min(5, len(teacher_topk[user]))] for user in sample_users},
        "teacher_metadata_keys": sorted(state.metadata.keys()),
        "uses_exact_teacher_scorer": state.scorer is not None,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
