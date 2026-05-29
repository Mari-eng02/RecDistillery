"""
Evaluate a saved teacher model on validation/test splits.

This script:
1) loads teacher embeddings from a `.teacher` file
2) computes top-k recommendations masking training items
3) computes ranking metrics on val/test
4) saves metrics to JSON (and optionally TSV)

Usage:
    python scripts/recdistill/evaluate_teacher.py \
      --dataset citeulike --teacher-model BPRMF --embedding-dim 200 --top-k 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

    sys.modules["recdistill.data"].InteractionDataset = interactions_module.InteractionDataset
    sys.modules["recdistill.teachers"].TeacherState = teacher_state_module.TeacherState
    sys.modules["recdistill.teachers"].load_teacher_state = teacher_loaders_module.load_teacher_state

    return interactions_module.InteractionDataset, teacher_loaders_module.load_teacher_state


try:
    from recdistill.data.datarec_loader import (
        load_eval_split as _load_eval_split_datarec,
        load_train_dataset as _load_train_dataset_datarec,
        resolve_teacher_dataset_mappings,
    )
    from recdistill.data.interactions import InteractionDataset
    from recdistill.teachers import load_teacher_state
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("recdistill"):
        InteractionDataset, load_teacher_state = _bootstrap_local_recdistill()
        from recdistill.data.datarec_loader import (
            load_eval_split as _load_eval_split_datarec,
            load_train_dataset as _load_train_dataset_datarec,
            resolve_teacher_dataset_mappings,
        )
    else:
        raise


def _parse_split(
    dataset_name: str,
    split_name: str,
    num_users: int,
    num_items: int,
    user_mapping: dict[int, int] | None = None,
    item_mapping: dict[int, int] | None = None,
) -> tuple[dict[int, set[int]], int]:
    return _load_eval_split_datarec(
        dataset_name=dataset_name,
        split_name=split_name,
        teacher_num_users=num_users,
        teacher_num_items=num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def _build_train_dataset(
    dataset_name: str,
    num_users: int,
    num_items: int,
    user_mapping: dict[int, int] | None = None,
    item_mapping: dict[int, int] | None = None,
) -> tuple[InteractionDataset, int]:
    return _load_train_dataset_datarec(
        dataset_name=dataset_name,
        teacher_num_users=num_users,
        teacher_num_items=num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def _recommend_topk(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: list[int],
    train_seen: dict[int, set[int]],
    top_k: int,
    batch_size: int,
    device: torch.device,
    scorer=None,
) -> tuple[dict[int, list[int]], int]:
    num_items = int(item_embeddings.size(0))
    k = max(1, min(int(top_k), num_items))

    user_emb = user_embeddings.to(device)
    item_emb = item_embeddings.to(device)
    recs: dict[int, list[int]] = {}
    leaked_users = 0

    with torch.no_grad():
        if scorer is not None:
            scorer = scorer.to(device)
            for user in users:
                scores = scorer.score_items_for_user(user=int(user), num_items=num_items).detach().to(device)
                seen = train_seen.get(user, set())
                if seen:
                    seen_idx = torch.tensor(sorted(seen), dtype=torch.long, device=device)
                    scores[seen_idx] = float("-inf")

                top_items = torch.topk(scores, k=k, dim=0).indices.detach().cpu().tolist()
                raw_topk = [int(item) for item in top_items[:top_k]]
                if seen and any(item in seen for item in raw_topk):
                    leaked_users += 1
                recs[user] = [item for item in raw_topk if item not in seen][:top_k]
            return recs, leaked_users

        for offset in range(0, len(users), batch_size):
            batch_users = users[offset : offset + batch_size]
            user_idx = torch.tensor(batch_users, dtype=torch.long, device=device)
            scores = torch.matmul(user_emb[user_idx], item_emb.transpose(0, 1))

            for row_idx, user in enumerate(batch_users):
                seen = train_seen.get(user, set())
                if seen:
                    seen_idx = torch.tensor(sorted(seen), dtype=torch.long, device=device)
                    scores[row_idx, seen_idx] = float("-inf")

            top_items = torch.topk(scores, k=k, dim=1).indices.detach().cpu().tolist()
            for row_idx, user in enumerate(batch_users):
                seen = train_seen.get(user, set())
                raw_topk = [int(item) for item in top_items[row_idx][:top_k]]
                if seen and any(item in seen for item in raw_topk):
                    leaked_users += 1
                recs[user] = [item for item in raw_topk if item not in seen][:top_k]

    return recs, leaked_users


def _metrics_at_k(recs: dict[int, list[int]], gt: dict[int, set[int]], k: int) -> dict[str, float]:
    users = [user for user, items in gt.items() if items]
    if not users:
        return {"users": 0, "precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hr": 0.0}

    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    hr_sum = 0.0

    for user in users:
        pred = recs.get(user, [])[:k]
        truth = gt[user]
        hits = [1 if item in truth else 0 for item in pred]
        num_hits = sum(hits)

        precision_sum += num_hits / max(1, k)
        recall_sum += num_hits / max(1, len(truth))
        hr_sum += 1.0 if num_hits > 0 else 0.0

        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_len = min(len(truth), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

    total = float(len(users))
    return {
        "users": int(total),
        "precision": precision_sum / total,
        "recall": recall_sum / total,
        "ndcg": ndcg_sum / total,
        "hr": hr_sum / total,
    }


def _evaluate_split(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_seen: dict[int, set[int]],
    gt: dict[int, set[int]],
    top_k: int,
    batch_size: int,
    device: torch.device,
    scorer=None,
) -> tuple[dict[str, float], int]:
    users = sorted(user for user, items in gt.items() if items)
    if not users:
        return _metrics_at_k({}, gt, top_k), 0
    recs, leaked_users = _recommend_topk(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=users,
        train_seen=train_seen,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        scorer=scorer,
    )
    return _metrics_at_k(recs, gt, top_k), leaked_users


def _candidate_models(model: str) -> list[str]:
    upper = model.upper()
    if upper == "NMF":
        return ["NMF", "NFM"]
    if upper == "NFM":
        return ["NFM", "NMF"]
    return [upper]


def _resolve_teacher_path(model: str, dataset: str, embedding_dim: int, framework: str | None = None) -> tuple[Path, str]:
    for candidate in _candidate_models(model):
        path = Path(
            teacher_weights_path(
                model=candidate,
                dataset=dataset,
                embedding_dim=embedding_dim,
                phase="best",
                framework=framework,
            )
        )
        if path.exists():
            return path, candidate
    # fallback to first candidate path even if missing, to keep error explicit
    candidate = _candidate_models(model)[0]
    path = Path(
        teacher_weights_path(
            model=candidate,
            dataset=dataset,
            embedding_dim=embedding_dim,
            phase="best",
            framework=framework,
        )
    )
    return path, candidate


def _perf_dir_from_teacher_path(teacher_path: Path) -> Path:
    if teacher_path.parent.name in {"artifacts", "wei"}:
        return teacher_path.parent.parent / "perf"
    return teacher_path.parent / "perf"


def _path_slug(value) -> str:
    return str(value).strip().replace(" ", "_").replace("-", "_").replace("+", "_").replace("/", "_").replace("\\", "_")


def _experiment_id_from_artifact(path: Path | None) -> str | None:
    if path is None:
        return None
    stem = path.stem
    if stem.endswith("_best"):
        stem = stem[:-5]
    return stem.rsplit("_", 1)[-1] if "_" in stem else None


def _default_eval_filename(
    *,
    framework: str | None,
    model: str,
    dataset: str,
    embedding_dim: int,
    top_k: int,
    suffix: str,
    teacher_path: Path | None = None,
) -> str:
    framework_label = _path_slug(framework or "teacher")
    experiment_id = _experiment_id_from_artifact(teacher_path)
    stem = f"{framework_label}_{_path_slug(model)}_{_path_slug(dataset)}"
    if experiment_id:
        stem = f"{stem}_{experiment_id}"
    else:
        stem = f"{stem}_{int(embedding_dim)}"
    return f"{stem}_eval_top{top_k}{suffix}"


def _default_output_json(
    dataset: str,
    model: str,
    embedding_dim: int,
    top_k: int,
    framework: str | None = None,
    teacher_path: Path | None = None,
) -> Path:
    if teacher_path is not None:
        out_dir = _perf_dir_from_teacher_path(teacher_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / _default_eval_filename(
            framework=framework,
            model=model,
            dataset=dataset,
            embedding_dim=embedding_dim,
            top_k=top_k,
            suffix=".json",
            teacher_path=teacher_path,
        )
    teacher_path = Path(
        teacher_weights_path(
            model=model,
            dataset=dataset,
            embedding_dim=embedding_dim,
            phase="best",
            framework=framework,
        )
    )
    out_dir = teacher_path.parent.parent / "perf" if teacher_path.parent.name == "wei" else teacher_path.parent / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / _default_eval_filename(
        framework=framework,
        model=model,
        dataset=dataset,
        embedding_dim=embedding_dim,
        top_k=top_k,
        suffix=".json",
        teacher_path=teacher_path,
    )


def _default_output_tsv(
    dataset: str,
    model: str,
    embedding_dim: int,
    top_k: int,
    framework: str | None = None,
    teacher_path: Path | None = None,
) -> Path:
    if teacher_path is not None:
        out_dir = _perf_dir_from_teacher_path(teacher_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / _default_eval_filename(
            framework=framework,
            model=model,
            dataset=dataset,
            embedding_dim=embedding_dim,
            top_k=top_k,
            suffix=".tsv",
            teacher_path=teacher_path,
        )
    teacher_path = Path(
        teacher_weights_path(
            model=model,
            dataset=dataset,
            embedding_dim=embedding_dim,
            phase="best",
            framework=framework,
        )
    )
    out_dir = teacher_path.parent.parent / "perf" if teacher_path.parent.name == "wei" else teacher_path.parent / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / _default_eval_filename(
        framework=framework,
        model=model,
        dataset=dataset,
        embedding_dim=embedding_dim,
        top_k=top_k,
        suffix=".tsv",
        teacher_path=teacher_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher model recommendation metrics.")
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--teacher-model", "--model", dest="model", default=None, choices=["BPRMF", "LGCN", "NMF", "NFM"], help="Teacher model name")
    parser.add_argument("--teacher-framework", "--framework", dest="framework", default=None, choices=["recbole", "elliot", "lenskit"], help="Framework namespace for default teacher path")
    parser.add_argument("--embedding-dim", default=None, type=int, help="Teacher embedding dim used in filename")
    parser.add_argument("--teacher-path", default=None, help="Optional explicit path to .teacher file")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k recommendations for metrics")
    parser.add_argument("--batch-size", type=int, default=256, help="User batch size for ranking")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--assert-no-train-leak", action="store_true", help="Fail if train leakage is detected")
    parser.add_argument("--output-json", default=None, help="Output JSON path (default: results/.../perf)")
    parser.add_argument("--output-tsv", default=None, help="Optional output TSV path with summary metrics")
    args = parser.parse_args()

    if args.teacher_path:
        teacher_path = Path(args.teacher_path)
        resolved_model = args.model.upper() if args.model else None
    else:
        missing = [name for name, value in (("--dataset", args.dataset), ("--teacher-model", args.model), ("--embedding-dim", args.embedding_dim)) if value is None]
        if missing:
            parser.error(f"the following arguments are required when --teacher-path is not set: {', '.join(missing)}")
        teacher_path, resolved_model = _resolve_teacher_path(
            model=args.model,
            dataset=args.dataset,
            embedding_dim=args.embedding_dim,
            framework=args.framework,
        )
    teacher_state = load_teacher_state(teacher_path, device="cpu")
    metadata = teacher_state.metadata if isinstance(teacher_state.metadata, dict) else {}
    dataset = args.dataset or metadata.get("dataset")
    resolved_model = resolved_model or metadata.get("model_name") or metadata.get("model") or metadata.get("backbone")
    embedding_dim = args.embedding_dim or metadata.get("embedding_dim") or teacher_state.embedding_dim
    resolved_framework = args.framework or metadata.get("framework")
    run_config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    resolved_framework = resolved_framework or run_config.get("framework")
    if dataset is None:
        parser.error("--dataset is required because it could not be inferred from --teacher-path metadata.")
    if resolved_model is None:
        parser.error("--teacher-model is required because it could not be inferred from --teacher-path metadata.")
    if embedding_dim is None:
        parser.error("--embedding-dim is required because it could not be inferred from --teacher-path metadata.")
    dataset = str(dataset)
    resolved_model = str(resolved_model).upper()
    embedding_dim = int(embedding_dim)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"Evaluating teacher: {resolved_model} ({embedding_dim}D) on {dataset}")
    if args.framework:
        print(f"Framework: {args.framework}")
    print(f"Teacher path: {teacher_path}")
    print(f"Device: {device}")
    print("=" * 80)
    print(f"Teacher users/items: {teacher_state.num_users}/{teacher_state.num_items}")

    user_mapping, item_mapping, mapping_source = resolve_teacher_dataset_mappings(
        teacher_state.metadata,
        dataset_name=dataset,
    )
    print(f"Dataset mapping source: {mapping_source}")

    train_dataset, dropped_train = _build_train_dataset(
        dataset_name=dataset,
        num_users=teacher_state.num_users,
        num_items=teacher_state.num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    val_dict, dropped_val = _parse_split(
        dataset_name=dataset,
        split_name="val",
        num_users=teacher_state.num_users,
        num_items=teacher_state.num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    test_dict, dropped_test = _parse_split(
        dataset_name=dataset,
        split_name="test",
        num_users=teacher_state.num_users,
        num_items=teacher_state.num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )

    print(f"Train interactions: {len(train_dataset.interactions)} (dropped: {dropped_train})")
    print(f"Validation interactions: {sum(len(v) for v in val_dict.values())} (dropped: {dropped_val})")
    print(f"Test interactions: {sum(len(v) for v in test_dict.values())} (dropped: {dropped_test})")
    if dropped_train or dropped_val or dropped_test:
        raise RuntimeError(
            "Some split interactions were dropped while mapping dataset IDs to teacher embeddings. "
            f"dropped_train={dropped_train}, dropped_val={dropped_val}, dropped_test={dropped_test}. "
            "This usually means the artifact was trained with a different user/item indexing scheme."
        )

    val_metrics, val_leaks = _evaluate_split(
        user_embeddings=teacher_state.user_embeddings,
        item_embeddings=teacher_state.item_embeddings,
        train_seen=train_dataset.train_dict,
        gt=val_dict,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=device,
        scorer=teacher_state.scorer,
    )
    test_metrics, test_leaks = _evaluate_split(
        user_embeddings=teacher_state.user_embeddings,
        item_embeddings=teacher_state.item_embeddings,
        train_seen=train_dataset.train_dict,
        gt=test_dict,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=device,
        scorer=teacher_state.scorer,
    )

    if args.assert_no_train_leak and (val_leaks > 0 or test_leaks > 0):
        raise RuntimeError(
            "Train-item leakage detected in recommendations. "
            f"val_leaks={val_leaks}, test_leaks={test_leaks}"
        )

    print(f"\nValidation @ {args.top_k}: P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} NDCG={val_metrics['ndcg']:.4f} HR={val_metrics['hr']:.4f}")
    print(f"Test @ {args.top_k}:       P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} NDCG={test_metrics['ndcg']:.4f} HR={test_metrics['hr']:.4f}")
    print(f"Leakage users: val={val_leaks}, test={test_leaks}")

    result = {
        "dataset": dataset,
        "model": resolved_model,
        "embedding_dim": embedding_dim,
        "teacher_path": str(teacher_path),
        "top_k": args.top_k,
        "train_interactions": len(train_dataset.interactions),
        "val_interactions": int(sum(len(v) for v in val_dict.values())),
        "test_interactions": int(sum(len(v) for v in test_dict.values())),
        "dropped_train": dropped_train,
        "dropped_val": dropped_val,
        "dropped_test": dropped_test,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "leaked_users_val": val_leaks,
        "leaked_users_test": test_leaks,
    }

    output_json = Path(args.output_json) if args.output_json else _default_output_json(
        dataset=dataset,
        model=resolved_model,
        embedding_dim=embedding_dim,
        top_k=args.top_k,
        framework=resolved_framework,
        teacher_path=teacher_path if args.teacher_path else None,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {output_json}")

    if args.output_tsv:
        output_tsv = Path(args.output_tsv)
    else:
        output_tsv = _default_output_tsv(
            dataset=dataset,
            model=resolved_model,
            embedding_dim=embedding_dim,
            top_k=args.top_k,
            framework=resolved_framework,
            teacher_path=teacher_path if args.teacher_path else None,
        )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8") as fp:
        fp.write("split\tprecision\trecall\tndcg\thr\tusers\tleaked_users\n")
        fp.write(
            f"val\t{val_metrics['precision']:.8f}\t{val_metrics['recall']:.8f}\t"
            f"{val_metrics['ndcg']:.8f}\t{val_metrics['hr']:.8f}\t"
            f"{int(val_metrics['users'])}\t{val_leaks}\n"
        )
        fp.write(
            f"test\t{test_metrics['precision']:.8f}\t{test_metrics['recall']:.8f}\t"
            f"{test_metrics['ndcg']:.8f}\t{test_metrics['hr']:.8f}\t"
            f"{int(test_metrics['users'])}\t{test_leaks}\n"
        )
    print(f"Saved TSV: {output_tsv}\n")


if __name__ == "__main__":
    main()
