"""
Train a student model with distillation from a saved teacher.

Example:
    python scripts/recdistill/train_student.py \
        --dataset citeulike \
        --teacher-framework recbole \
        --teacher-model BPRMF \
        --teacher-embedding-dim 200 \
        --student-framework recbole \
        --student-backbone BPRMF \
        --student-embedding-dim 20 \
        --lambda_de 0.1 \
        --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.evaluation import evaluate_embeddings, evaluate_student
from recdistill.experiment_runner import RecDistillExperimentRunner
from recdistill.model_validation import validate_distillation_request, validate_teacher_representation_request
from recdistill.paths import resolve_student_checkpoint_from_args, resolve_teacher_checkpoint_from_args
from recdistill.tracking import WandBRunLogger, resolve_wandb_logger
from recdistill.training import (
    build_lightgcn_graph,
    build_train_loader,
    prepare_distiller_trainable_modules,
    set_seed,
)


def resolve_teacher_path(args: argparse.Namespace) -> Path:
    return resolve_teacher_checkpoint_from_args(args)


def resolve_distiller_name(args: argparse.Namespace) -> str:
    names = []
    if float(getattr(args, "lambda_de", 0.0)) > 0:
        names.append("DE")
    if float(getattr(args, "lambda_rrd", 0.0)) > 0:
        names.append("RRD")
    if float(getattr(args, "lambda_unkd", 0.0)) > 0:
        names.append("UnKD")
    if float(getattr(args, "lambda_td", 0.0)) > 0:
        names.append(str(getattr(args, "td_type", "TD")).upper())
    return "-".join(names) if names else "NONE"


def resolve_output_path(args: argparse.Namespace) -> Path:
    return resolve_student_checkpoint_from_args(args, distiller_name=resolve_distiller_name(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a student with distillation from a saved teacher.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., citeulike, bookcrossing, amazon_cd)")
    parser.add_argument("--teacher-model", default=None, help="Teacher model name for path resolution (BPRMF/LGCN/NMF)")
    parser.add_argument("--teacher-embedding-dim", type=int, default=None, help="Teacher embedding dimension in filename")
    parser.add_argument("--teacher-path", default=None, help="Optional explicit path to teacher .teacher file")
    parser.add_argument("--teacher-framework", default="auto", help="Teacher source framework or adapter family")
    parser.add_argument("--teacher-format", default="auto", help="Teacher source artifact format")
    parser.add_argument("--teacher-adapter", default=None, help="Custom TeacherAdapter import path")
    parser.add_argument("--teacher-user-embeddings-path", default=None, help="Optional .npy user embeddings for generic teacher import")
    parser.add_argument("--teacher-item-embeddings-path", default=None, help="Optional .npy item embeddings for generic teacher import")
    parser.add_argument("--teacher-score-matrix-path", default=None, help="Optional dense user-item score matrix .npy for generic teacher import")
    parser.add_argument("--teacher-topk-items-path", default=None, help="Optional top-k item ids .npy for generic teacher import")
    parser.add_argument("--teacher-topk-scores-path", default=None, help="Optional top-k scores .npy for generic teacher import")
    parser.add_argument("--teacher-noise-scale", type=float, default=0.0, help="Relative Gaussian noise scale alpha for teacher embeddings (noise_std=alpha*base_std)")
    parser.add_argument("--teacher-noise-target", type=str, default="both", choices=["both", "user", "item"], help="Which teacher embeddings to perturb")
    parser.add_argument("--teacher-noise-seed", type=int, default=None, help="Optional seed used only for teacher noise injection")
    parser.add_argument(
        "--student-backbone",
        type=str,
        default="BPRMF",
        help="Student backbone (BPRMF, LINE, LGCN, NGCF, DGCF, SGL, SpectralCF, or NMF)",
    )
    parser.add_argument("--student-framework", type=str, default="recbole", choices=["recbole", "elliot", "lenskit"], help="Framework implementation used for the student backbone")
    parser.add_argument("--student-embedding-dim", type=int, default=64, help="Student embedding dimension")
    parser.add_argument("--lightgcn-layers", type=int, default=2, help="Number of LightGCN propagation layers when --student-backbone LGCN")
    parser.add_argument("--neumf-mlp-dims", default="64,32,16,8", help="Comma-separated NeuMF MLP hidden sizes when --student-backbone NMF")
    parser.add_argument("--neumf-dropout", type=float, default=0.0, help="NeuMF dropout when --student-backbone NMF")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Student optimizer learning rate")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="BPR L2 regularization")
    parser.add_argument("--lambda-de", type=float, default=0.0, help="DE loss weight")
    parser.add_argument("--num-experts", type=int, default=10, help="Number of DE experts")
    parser.add_argument("--temperature", type=float, default=1.0, help="DE Gumbel-Softmax temperature")
    parser.add_argument("--lambda-rrd", type=float, default=0.0, help="RRD loss weight")
    parser.add_argument("--rrd-interesting-size", type=int, default=10, help="RRD interesting set size")
    parser.add_argument("--rrd-uninteresting-size", type=int, default=50, help="RRD uninteresting set size")
    parser.add_argument("--rrd-temperature", type=float, default=1.0, help="RRD sampling temperature")
    parser.add_argument("--rrd-teacher-topk", type=int, default=500, help="Teacher top-k list size for RRD sampling")
    parser.add_argument("--lambda-unkd", type=float, default=0.0, help="UnKD loss weight")
    parser.add_argument("--unkd-sample-num", type=int, default=30, help="UnKD sampled item pairs per group")
    parser.add_argument("--unkd-group-count", type=int, default=2, help="UnKD popularity group count")
    parser.add_argument("--unkd-popularity-lambda", type=float, default=1.0, help="UnKD popularity smoothing factor")
    parser.add_argument("--unkd-rank-top-k", type=int, default=1000, help="UnKD teacher top-k list size")
    parser.add_argument("--unkd-rank-temperature", type=float, default=20.0, help="UnKD ranking temperature")
    parser.add_argument("--lambda-td", type=float, default=0.0, help="Topology distillation loss weight (HTD/FTD)")
    parser.add_argument("--td-type", type=str, default="HTD", choices=["HTD", "FTD"], help="Topology distiller type")
    parser.add_argument("--htd-alpha", type=float, default=0.5, help="HTD alpha balancing topology and group assignment losses")
    parser.add_argument("--htd-num-groups", type=int, default=40, help="HTD number of groups")
    parser.add_argument("--htd-topology-mode", type=str, default="group_pe", choices=["group_pp", "group_pe"], help="HTD group-level topology mode")
    parser.add_argument("--htd-initial-tau", type=float, default=1.0, help="HTD initial Gumbel-Softmax temperature")
    parser.add_argument("--htd-min-tau", type=float, default=1e-10, help="HTD minimum Gumbel-Softmax temperature")
    parser.add_argument("--htd-decay-epochs", type=int, default=100, help="HTD temperature decay epochs")
    parser.add_argument("--td-entity-sample-size", type=int, default=0, help="Max unique users+items used by HTD/FTD topology loss per batch (0 uses all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--output-path", default=None, help="Optional explicit output checkpoint path")
    parser.add_argument("--output-strategy", default="best", choices=["best", "bayesian", "tracked"], help="RecDistill output strategy directory used when --output-path is not set")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0 disables)")
    parser.add_argument("--skip-eval", action="store_true", help="Disable validation/test evaluation during training")
    parser.add_argument("--eval-k", type=int, default=20, help="Top-k for recommendation metrics")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="User batch size for recommendation scoring")
    parser.add_argument("--eval-val-only", action="store_true", help="Evaluate only on validation split (skip test metrics)")
    parser.add_argument("--selection-split", type=str, default="val", choices=["val", "test"], help="Split used for best-checkpoint selection")
    parser.add_argument("--selection-metric", type=str, default="ndcg", choices=["precision", "recall", "ndcg", "hr"], help="Metric used for best-checkpoint selection")
    parser.add_argument("--assert-no-train-leak", action="store_true", help="Fail if any train item appears in recommendations")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-mode", type=str, default="loss", choices=["loss", "val_metric"], help="Early-stop monitor source")
    parser.add_argument("--early-stop-metric", type=str, default="ndcg", choices=["precision", "recall", "ndcg", "hr"], help="Validation metric used when early-stop-mode=val_metric")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Early-stop patience in monitor steps")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum improvement required to reset patience")
    parser.add_argument("--early-stop-warmup", type=int, default=0, help="Do not early-stop before this epoch")
    parser.add_argument("--early-stop-restore-best", action="store_true", help="Restore best early-stop checkpoint at end")
    parser.add_argument("--wandb-log", action="store_true", help="Enable Weights & Biases experiment logging")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B run group")
    parser.add_argument("--wandb-notes", default=None, help="Optional W&B notes")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.early_stop and args.early_stop_mode == "val_metric" and args.skip_eval:
        raise ValueError("early-stop mode 'val_metric' requires evaluation enabled (remove --skip-eval).")
    if args.eval_val_only and args.selection_split == "test":
        raise ValueError("--eval-val-only is incompatible with --selection-split test.")
    if args.early_stop and args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0.")

    distiller = resolve_distiller_name(args)
    if distiller == "NONE":
        return

    teacher_framework = str(getattr(args, "teacher_framework", "") or "").strip().lower()
    validate_distillation_request(
        teacher_framework=teacher_framework if teacher_framework in {"recbole", "elliot", "lenskit"} else None,
        teacher_model=args.teacher_model,
        student_framework=args.student_framework,
        student_backbone=args.student_backbone,
        distiller=distiller,
        validate_teacher=teacher_framework in {"recbole", "elliot", "lenskit"} and bool(args.teacher_model) and not bool(args.teacher_path),
    )
    validate_teacher_representation_request(
        teacher_conf={
            "format": args.teacher_format,
            "user_embeddings_path": args.teacher_user_embeddings_path,
            "item_embeddings_path": args.teacher_item_embeddings_path,
            "score_matrix_path": args.teacher_score_matrix_path,
            "topk_items_path": args.teacher_topk_items_path,
            "topk_scores_path": args.teacher_topk_scores_path,
        },
        distiller=distiller,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)

    runner = RecDistillExperimentRunner(args)
    runner.wandb_logger = resolve_wandb_logger(args, runner.run_config())
    runner.run()


if __name__ == "__main__":
    main()
