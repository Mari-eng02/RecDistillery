import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def _bayesian_enabled(config_path: str | Path, root_key: str) -> bool:
    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        config = json.loads(raw_text)
    else:
        import yaml

        config = yaml.safe_load(raw_text) or {}
    config = get_config_loader().resolve_config_modules(config)
    train = config.get(root_key, {}) if isinstance(config, dict) else {}
    if not isinstance(train, dict):
        return False
    optimization = train.get("optimization", {}) or {}
    bayesian = optimization.get("bayesian") or train.get("bayesian") or config.get("bayesian") or {}
    return bool(isinstance(bayesian, dict) and bayesian.get("enabled", False))


def _run_optuna(config_path: str | Path) -> None:
    optuna_script = REPO_ROOT / "scripts" / "recdistill" / "run_optuna.py"
    subprocess.run([sys.executable, str(optuna_script), "--config", str(config_path)], check=True)


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
        experiment_path = loader.save_generated_experiment(
            kind="teacher",
            name=f"{args.framework}_{args.model}_{args.dataset}",
            path_parts=[],
            config=composed_config,
        )
        train_args = native_args_from_config_file(
            experiment_path,
            role="teacher",
            fallback_dataset=args.dataset,
            fallback_backbone=args.model,
            overrides=overrides,
        )
        print(f"Generated teacher config saved to: {experiment_path}")

    if _bayesian_enabled(train_args.config_path, "train_teacher"):
        _run_optuna(train_args.config_path)
        sys.exit(0)

    NativeModelTrainingRunner(train_args).run()
