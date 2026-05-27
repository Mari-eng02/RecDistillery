import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config_loader
from recdistill.config_integration import (
    load_recdistill_config_from_file,
    load_recdistill_experiment,
    recdistill_config_to_dict,
)
from recdistill.experiment_runner import RecDistillExperimentRunner
from recdistill.native_runner import (
    NativeModelTrainingRunner,
    native_args_from_config_file,
)
from recdistill.paths import AMAZONCD, BOOKCROSSING, BPRMF, CITEULIKE, LGCN, NMF
from recdistill.model_validation import validate_distillation_request, validate_trainable_model
from recdistill.training import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a native RecDistill student.")
    parser.add_argument("--backbone", type=str, default=BPRMF, help="Student recommendation backbone")
    parser.add_argument("--framework", type=str, default=None, choices=["recbole", "elliot", "lenskit"])
    parser.add_argument("--dataset", type=str, default=CITEULIKE, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Optional native YAML/JSON config")
    parser.add_argument(
        "--distillation",
        type=str,
        default="none",
        help="Distillation strategy. Use 'none' for plain student training.",
    )
    parser.add_argument("--teacher_model", type=str, default=None, help="Teacher model for distillation")
    parser.add_argument("--teacher-framework", type=str, default=None, choices=["recbole", "elliot", "lenskit"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--l2-reg", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None, help="Destination .student or .distilled_student artifact")
    parser.add_argument("--skip-eval", action="store_true")

    args = parser.parse_args()
    backbones = [BPRMF, "LINE", LGCN, "NGCF", "DGCF", "SGL", "ULTRAGCN", "SPECTRALCF", NMF]
    datasets = [CITEULIKE, BOOKCROSSING, AMAZONCD]
    distillation = str(args.distillation).strip().lower()
    is_plain = distillation in {"none", "plain", "no", "false", "0"}
    if not args.config:
        args.framework = args.framework or "recbole"
        args.teacher_framework = args.teacher_framework or "recbole"
        try:
            args.backbone = validate_trainable_model(args.framework, args.backbone, role="student backbone")
            if args.teacher_model is not None:
                args.teacher_model = validate_trainable_model(args.teacher_framework, args.teacher_model, role="teacher model")
            assert args.backbone in backbones, f"Invalid backbone: {args.backbone}. Choose from {backbones}"
            assert args.dataset in datasets, f"Invalid dataset: {args.dataset}. Choose from {datasets}"
        except (AssertionError, ValueError) as exc:
            parser.error(str(exc))

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "l2_reg": args.l2_reg,
        "framework": args.framework,
        "device": args.device,
        "num_workers": args.num_workers,
        "output_path": args.output_path,
        "skip_eval": True if args.skip_eval else None,
    }

    if is_plain:
        if args.config:
            train_args = native_args_from_config_file(
                args.config,
                role="student",
                fallback_dataset=args.dataset,
                fallback_backbone=args.backbone,
                overrides=overrides,
            )
            try:
                train_args.backbone = validate_trainable_model(train_args.framework, train_args.backbone, role="student backbone")
                assert train_args.backbone in backbones, f"Invalid backbone: {train_args.backbone}. Choose from {backbones}"
                assert train_args.dataset in datasets, f"Invalid dataset: {train_args.dataset}. Choose from {datasets}"
            except (AssertionError, ValueError) as exc:
                parser.error(str(exc))
        else:
            loader = get_config_loader()
            composed_config = loader.compose_student_training(
                dataset_name=args.dataset,
                model_name=args.backbone,
                framework=args.framework,
            )
            train_config = composed_config["train_student"]
            experiment_path = loader.save_generated_experiment(
                kind="student",
                name=(
                    f"{train_config['student']['framework']}_"
                    f"{train_config['student']['backbone']}_"
                    f"{train_config['dataset']}_"
                    f"{train_config['student']['embedding_dim']}"
                ),
                path_parts=[
                    train_config["student"]["framework"],
                    train_config["student"]["backbone"],
                    train_config["dataset"],
                ],
                config=composed_config,
            )
            train_args = native_args_from_config_file(
                experiment_path,
                role="student",
                fallback_dataset=args.dataset,
                fallback_backbone=args.backbone,
                overrides=overrides,
            )
            print(f"Generated student config saved to: {experiment_path}")
        NativeModelTrainingRunner(train_args).run()
    else:
        if args.config:
            config = load_recdistill_config_from_file(args.config)
            try:
                validate_distillation_request(
                    teacher_framework=config.distill_student.teacher.framework,
                    teacher_model=config.distill_student.teacher.model,
                    student_framework=config.distill_student.student.framework,
                    student_backbone=config.distill_student.student.backbone,
                    distiller=config.distill_student.distillation.strategy,
                    validate_teacher=not bool(config.distill_student.teacher.path),
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            try:
                teacher_model = args.teacher_model or args.backbone
                validate_distillation_request(
                    teacher_framework=args.teacher_framework,
                    teacher_model=teacher_model,
                    student_framework=args.framework,
                    student_backbone=args.backbone,
                    distiller=distillation,
                )
            except ValueError as exc:
                parser.error(str(exc))
            config = load_recdistill_experiment(
                dataset_name=args.dataset,
                teacher_model=args.teacher_model or args.backbone,
                distiller_strategy=distillation,
                student_backbone=args.backbone,
                teacher_framework=args.teacher_framework,
                student_framework=args.framework,
            )
            experiment_path = get_config_loader().save_generated_experiment(
                kind="recdistill",
                name=f"{distillation}_{args.teacher_framework}_{args.teacher_model or args.backbone}_{args.framework}_{args.backbone}_{args.dataset}",
                path_parts=[distillation, args.teacher_framework, args.teacher_model or args.backbone, args.framework, args.backbone, args.dataset],
                config=recdistill_config_to_dict(config),
            )
            print(f"Generated RecDistill config saved to: {experiment_path}")
        if args.output_path is not None:
            config.distill_student.runtime.output_path = args.output_path
        if args.framework is not None:
            config.distill_student.student.framework = args.framework
        if args.skip_eval:
            config.distill_student.evaluation.enabled = False
        set_seed(int(config.distill_student.runtime.seed))
        RecDistillExperimentRunner.from_config(config).run()
