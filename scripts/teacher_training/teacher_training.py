import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.native_runner import (
    NativeModelTrainingRunner,
    native_args_from_config_file,
)
from recdistill.paths import AMAZONCD, BOOKCROSSING, BPRMF, CITEULIKE, LGCN, NMF
from recdistill.model_validation import validate_trainable_model
from config import get_config_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a native RecDistill teacher.")
    parser.add_argument("--model", type=str, default=BPRMF, help="Recommendation model")
    parser.add_argument("--framework", type=str, default=None, choices=["recbole", "elliot", "lenskit"])
    parser.add_argument("--dataset", type=str, default=CITEULIKE, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Optional native YAML/JSON config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--l2-reg", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None, help="Destination .teacher artifact")
    parser.add_argument("--skip-eval", action="store_true")

    args = parser.parse_args()
    backbones = [BPRMF, "LINE", LGCN, "NGCF", "DGCF", "SGL", "ULTRAGCN", "SPECTRALCF", NMF]
    datasets = [CITEULIKE, BOOKCROSSING, AMAZONCD]
    if not args.config:
        args.framework = args.framework or "recbole"
        try:
            args.model = validate_trainable_model(args.framework, args.model, role="teacher model")
            assert args.model in backbones, f"Invalid model: {args.model}. Choose from {backbones}"
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
    if args.config:
        train_args = native_args_from_config_file(
            args.config,
            role="teacher",
            fallback_dataset=args.dataset,
            fallback_backbone=args.model,
            overrides=overrides,
        )
        try:
            train_args.backbone = validate_trainable_model(train_args.framework, train_args.backbone, role="teacher model")
            assert train_args.backbone in backbones, f"Invalid model: {train_args.backbone}. Choose from {backbones}"
            assert train_args.dataset in datasets, f"Invalid dataset: {train_args.dataset}. Choose from {datasets}"
        except (AssertionError, ValueError) as exc:
            parser.error(str(exc))
    else:
        loader = get_config_loader()
        composed_config = loader.compose_teacher_training(
            dataset_name=args.dataset,
            model_name=args.model,
            framework=args.framework,
        )
        preset_path = loader.save_generated_preset(
            kind="teacher",
            family="generated",
            name=(
                f"{composed_config['teacher']['framework']}_"
                f"{composed_config['teacher']['model']}_"
                f"{composed_config['dataset']}_"
                f"{composed_config['teacher']['embedding_dim']}"
            ),
            path_parts=[
                composed_config["teacher"]["framework"],
                composed_config["teacher"]["model"],
                composed_config["dataset"],
            ],
            config=composed_config,
        )
        train_args = native_args_from_config_file(
            preset_path,
            role="teacher",
            fallback_dataset=args.dataset,
            fallback_backbone=args.model,
            overrides=overrides,
        )
        print(f"Generated teacher config saved to: {preset_path}")

    NativeModelTrainingRunner(train_args).run()
