"""
Evaluate saved student artifacts on validation/test splits.

Supported inputs are RecDistill student checkpoints saved as `.student` or
`.distilled_student`. The checkpoint must contain a `student_state_dict`, the
student config, and the user/item shape used during training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.checkpointing import load_student_checkpoint  # noqa: E402
from recdistill.data.datarec_loader import load_eval_split, load_train_dataset  # noqa: E402
from recdistill.evaluation import evaluate_student  # noqa: E402
from recdistill.factories import build_student_model  # noqa: E402
from recdistill.paths import resolve_student_checkpoint  # noqa: E402
from recdistill.teachers import load_teacher_state  # noqa: E402
from recdistill.training import build_lightgcn_graph  # noqa: E402


def _config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def _plain_distiller_name(value: str | None) -> str:
    raw = str(value or "plain").strip().lower()
    return "plain" if raw in {"none", "plain", "no", "false", "0"} else raw


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.path:
        return Path(args.path)

    if not args.dataset or not args.student_backbone or not args.student_embedding_dim:
        raise ValueError(
            "Use --path, or provide --dataset, --student-backbone and --student-embedding-dim."
        )

    return resolve_student_checkpoint(
        dataset=args.dataset,
        distiller=args.distiller,
        teacher_model=args.teacher_model,
        teacher_framework=args.teacher_framework,
        student_backbone=args.student_backbone,
        student_framework=args.student_framework,
        student_embedding_dim=int(args.student_embedding_dim),
        output_path=None,
    )


def _resolve_dataset(args: argparse.Namespace, config: dict[str, Any], payload: dict[str, Any]) -> str:
    dataset = args.dataset or payload.get("dataset") or config.get("dataset")
    if not dataset:
        raise ValueError("Dataset is missing. Pass --dataset or use a checkpoint with config.dataset.")
    return str(dataset)


def _resolve_teacher_path(config: dict[str, Any], payload: dict[str, Any]) -> Path | None:
    raw_path = payload.get("teacher_path") or config.get("teacher_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.exists() else None


def _load_mappings(config: dict[str, Any], payload: dict[str, Any]) -> tuple[dict | None, dict | None, str | None]:
    teacher_path = _resolve_teacher_path(config, payload)
    if teacher_path is None:
        return None, None, None

    teacher_state = load_teacher_state(teacher_path, device="cpu")
    return (
        teacher_state.metadata.get("public_to_local_user_id"),
        teacher_state.metadata.get("public_to_local_item_id"),
        str(teacher_path),
    )


def _build_model(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    train_dataset,
) -> torch.nn.Module:
    backbone = args.student_backbone or _config_value(config, "student_backbone", "backbone", default=None)
    if backbone is None:
        raise ValueError("Student backbone is missing. Pass --student-backbone or use a checkpoint with config.")

    embedding_dim = args.student_embedding_dim or _config_value(
        config,
        "student_embedding_dim",
        "embedding_dim",
        default=None,
    )
    if embedding_dim is None:
        raise ValueError("Student embedding dim is missing. Pass --student-embedding-dim or use a checkpoint with config.")

    framework = args.student_framework or _config_value(
        config,
        "student_framework",
        "framework",
        default="recbole",
    )

    return build_student_model(
        backbone=str(backbone),
        dataset=train_dataset,
        embedding_dim=int(embedding_dim),
        l2_reg=0.0,
        lightgcn_layers=int(_config_value(config, "lightgcn_layers", default=2)),
        neumf_mlp_dims=_config_value(config, "neumf_mlp_dims", default="64,32,16,8"),
        neumf_dropout=float(_config_value(config, "neumf_dropout", "dropout", default=0.0)),
        framework=str(framework),
        graph_builder=build_lightgcn_graph,
    )


def _default_output_json(
    *,
    checkpoint_path: Path,
    dataset: str,
    top_k: int,
) -> Path:
    if checkpoint_path.parent.name == "wei":
        perf_dir = checkpoint_path.parent.parent / "perf"
    else:
        perf_dir = checkpoint_path.parent / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    return perf_dir / f"{checkpoint_path.stem}_{dataset}_eval_top{top_k}.json"


def _write_summary_tsv(output_tsv: Path, result: dict[str, Any]) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["split", "precision", "recall", "ndcg", "hr", "users", "leaked_users"],
            delimiter="\t",
        )
        writer.writeheader()
        for split in ("val", "test"):
            metrics = result[f"{split}_metrics"]
            writer.writerow(
                {
                    "split": split,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "ndcg": metrics["ndcg"],
                    "hr": metrics["hr"],
                    "users": metrics["users"],
                    "leaked_users": result[f"leaked_users_{split}"],
                }
            )


def evaluate_student_artifact(args: argparse.Namespace) -> Path:
    checkpoint_path = _resolve_checkpoint_path(args)
    payload = load_student_checkpoint(checkpoint_path, map_location="cpu")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    dataset = _resolve_dataset(args, config, payload)

    user_mapping, item_mapping, mapping_source = _load_mappings(config, payload)
    num_users = int(payload["num_users"])
    num_items = int(payload["num_items"])

    train_dataset, dropped_train = load_train_dataset(
        dataset_name=dataset,
        teacher_num_users=num_users,
        teacher_num_items=num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    val_gt, dropped_val = load_eval_split(
        dataset_name=dataset,
        split_name="val",
        teacher_num_users=num_users,
        teacher_num_items=num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    test_gt, dropped_test = load_eval_split(
        dataset_name=dataset,
        split_name="test",
        teacher_num_users=num_users,
        teacher_num_items=num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    if dropped_train or dropped_val or dropped_test:
        raise RuntimeError(
            "Some split interactions were dropped while mapping dataset IDs to student embeddings. "
            f"dropped_train={dropped_train}, dropped_val={dropped_val}, dropped_test={dropped_test}. "
            "This usually means the artifact was trained with a different user/item indexing scheme."
        )

    model = _build_model(args=args, config=config, train_dataset=train_dataset)
    model.load_state_dict(payload["student_state_dict"], strict=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics = evaluate_student(
        model=model,
        train_seen=train_dataset.train_dict,
        val_gt=val_gt,
        test_gt=test_gt,
        top_k=int(args.top_k),
        batch_size=int(args.batch_size),
        device=device,
        eval_val_only=False,
    )

    val_leaks = int(metrics.get("leaked_users_val", 0))
    test_leaks = int(metrics.get("leaked_users_test", 0))
    if args.assert_no_train_leak and (val_leaks > 0 or test_leaks > 0):
        raise RuntimeError(f"Train-item leakage detected: val={val_leaks}, test={test_leaks}")

    student_model = args.student_backbone or _config_value(config, "student_backbone", "backbone", default=payload.get("student"))
    student_framework = args.student_framework or _config_value(config, "student_framework", "framework", default=None)
    embedding_dim = args.student_embedding_dim or _config_value(config, "student_embedding_dim", "embedding_dim", default=None)
    distiller = _plain_distiller_name(
        args.distiller
        or payload.get("distiller")
        or _config_value(config, "distiller", default=None)
    )
    teacher_model = args.teacher_model or _config_value(config, "teacher_model", default=payload.get("teacher"))

    result = {
        "kind": "student",
        "artifact_path": str(checkpoint_path),
        "dataset": dataset,
        "distiller": distiller,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "student_framework": student_framework,
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "top_k": int(args.top_k),
        "epoch": payload.get("epoch"),
        "best_epoch": payload.get("best_epoch"),
        "best_selection_split": payload.get("best_selection_split"),
        "best_selection_metric": payload.get("best_selection_metric"),
        "best_selection_score": payload.get("best_selection_score"),
        "mapping_source_teacher": mapping_source,
        "train_interactions": len(train_dataset.interactions),
        "val_interactions": int(sum(len(items) for items in val_gt.values())),
        "test_interactions": int(sum(len(items) for items in test_gt.values())),
        "dropped_train": dropped_train,
        "dropped_val": dropped_val,
        "dropped_test": dropped_test,
        "val_metrics": metrics["val"],
        "test_metrics": metrics["test"],
        "leaked_users_val": val_leaks,
        "leaked_users_test": test_leaks,
    }

    output_json = Path(args.output_json) if args.output_json else _default_output_json(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        top_k=int(args.top_k),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    output_tsv = Path(args.output_tsv) if args.output_tsv else output_json.with_suffix(".tsv")
    _write_summary_tsv(output_tsv, result)

    print(f"Validation @ {args.top_k}: P={metrics['val']['precision']:.4f} R={metrics['val']['recall']:.4f} NDCG={metrics['val']['ndcg']:.4f} HR={metrics['val']['hr']:.4f}")
    print(f"Test @ {args.top_k}:       P={metrics['test']['precision']:.4f} R={metrics['test']['recall']:.4f} NDCG={metrics['test']['ndcg']:.4f} HR={metrics['test']['hr']:.4f}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved TSV: {output_tsv}")
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .student or .distilled_student artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/recdistill/evaluate_students.py --path results/students/recbole/BPRMF/citeulike/best/wei/recbole_BPRMF_citeulike_64.student

  python scripts/recdistill/evaluate_students.py ^
    --dataset citeulike ^
    --distiller de ^
    --teacher-model BPRMF ^
    --student-backbone BPRMF ^
    --student-embedding-dim 64
        """,
    )
    parser.add_argument("--path", default=None, help="Explicit .student or .distilled_student artifact path")
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--distiller", default="plain", help="Distiller name for path resolution; use plain for non-distilled students")
    parser.add_argument("--teacher-model", default=None, help="Teacher model used for distilled-student path resolution")
    parser.add_argument("--teacher-framework", default="recbole", choices=["recbole", "elliot", "lenskit"], help="Teacher framework used for distilled-student path resolution")
    parser.add_argument("--student-backbone", "--student-model", "--model", dest="student_backbone", default=None, help="Student backbone/model")
    parser.add_argument("--student-framework", default=None, choices=["recbole", "elliot", "lenskit"], help="Override student framework adapter")
    parser.add_argument("--student-embedding-dim", "--embedding-dim", dest="student_embedding_dim", type=int, default=None, help="Student embedding dimension")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k recommendations")
    parser.add_argument("--batch-size", type=int, default=256, help="User batch size for ranking")
    parser.add_argument("--device", default=None, help="Torch device")
    parser.add_argument("--assert-no-train-leak", action="store_true", help="Fail if recommendations include train items")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-tsv", default=None, help="Output TSV path")
    args = parser.parse_args()

    evaluate_student_artifact(args)


if __name__ == "__main__":
    main()
