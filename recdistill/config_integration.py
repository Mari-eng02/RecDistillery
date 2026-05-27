"""
RecDistill configuration integration with the new centralized config system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import yaml
from config import get_config_loader, RecDistillConfig


def load_recdistill_config_from_file(config_path: Union[str, Path]) -> RecDistillConfig:
    """
    Load and validate RecDistill configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Validated RecDistillConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        raw_config = yaml.safe_load(fp) or {}

    normalized = normalize_recdistill_config(raw_config)
    return RecDistillConfig(**normalized)


def load_recdistill_experiment(
    dataset_name: str,
    teacher_model: str,
    distiller_strategy: str,
    student_backbone: Optional[str] = None,
    teacher_framework: str = "recbole",
    student_framework: str = "recbole",
    overrides: Optional[Dict[str, Any]] = None
) -> RecDistillConfig:
    """
    Programmatically load RecDistill experiment with centralized configurations.
    
    Args:
        dataset_name: Dataset name (amazon_cd, bookcrossing, citeulike)
        teacher_model: Teacher model (bprmf, nmf, lgcn)
        distiller_strategy: Distillation strategy (de, htd, ftd, unkd)
        student_backbone: Student backbone (default: same as teacher)
        overrides: Dict of configuration overrides
        
    Returns:
        Validated RecDistillConfig object
        
    Example:
        >>> config = load_recdistill_experiment(
        ...     dataset_name='citeulike',
        ...     teacher_model='nmf',
        ...     distiller_strategy='de',
        ...     overrides={
        ...         'distill_student.optimization.epochs': 50,
        ...         'distill_student.runtime.seed': 123
        ...     }
        ... )
    """
    if student_backbone is None:
        student_backbone = teacher_model
    
    loader = get_config_loader()
    config_dict = loader.compose_recdistill_experiment(
        dataset_name=dataset_name,
        teacher_model=teacher_model,
        distiller_strategy=distiller_strategy,
        student_backbone=student_backbone,
        teacher_framework=teacher_framework,
        student_framework=student_framework,
    )
    
    # Apply overrides if provided
    if overrides:
        config_dict = _apply_overrides(config_dict, overrides)
    
    return RecDistillConfig(**normalize_recdistill_config(config_dict))


def normalize_recdistill_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize RecDistill config dictionaries into RecDistillConfig shape.
    """
    if isinstance(config, dict) and "preset" in config and "config" in config:
        config = config["config"]

    if isinstance(config, dict):
        config = get_config_loader().resolve_config_modules(config)

    normalized = copy.deepcopy(config or {})
    if "distill_student" not in normalized:
        raise ValueError("RecDistill configs must define a top-level 'distill_student' block.")
    train_conf = normalized.get("distill_student") or {}
    normalized["distill_student"] = train_conf

    student_conf = train_conf.setdefault("student", {})
    teacher_conf = train_conf.setdefault("teacher", {})
    distill_conf = train_conf.setdefault("distillation", {})

    strategy = (
        distill_conf.get("strategy")
        or _infer_strategy_from_distillation_block(distill_conf)
        or "DE"
    )
    strategy = str(strategy).replace("-", "_").upper()
    distill_conf["strategy"] = strategy
    student_conf.pop("model", None)
    active_methods = {part for part in strategy.replace("+", "_").split("_") if part}
    if "DE" not in active_methods:
        distill_conf["lambda_de"] = 0.0

    if "model" in teacher_conf and teacher_conf["model"] is not None:
        teacher_conf["model"] = str(teacher_conf["model"]).upper()
    if "backbone" in student_conf and student_conf["backbone"] is not None:
        student_conf["backbone"] = str(student_conf["backbone"]).upper()

    for key in (
        "temperature",
        "lambda_de",
        "lambda_kl",
        "lambda_rrd",
        "lambda_unkd",
        "num_experts",
    ):
        if key in student_conf and key not in distill_conf:
            distill_conf[key] = student_conf[key]

    train_conf.setdefault("optimization", {})
    train_conf.setdefault("runtime", {})
    train_conf.setdefault("evaluation", {})
    train_conf.setdefault("early_stopping", {})
    return normalized


def recdistill_config_to_dict(config: Union[RecDistillConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a plain normalized dictionary for runner code."""
    if isinstance(config, RecDistillConfig):
        return config.model_dump()
    return normalize_recdistill_config(config)


def _infer_strategy_from_distillation_block(distill_conf: Dict[str, Any]) -> Optional[str]:
    if "topology" in distill_conf:
        topology = distill_conf.get("topology") or {}
        if isinstance(topology, dict) and topology.get("type"):
            return str(topology["type"])
        return "HTD"
    if "rrd" in distill_conf:
        return "RRD"
    if "unkd" in distill_conf:
        return "UNKD"
    if "ftd" in distill_conf:
        return "FTD"
    return None


def _apply_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply overrides to configuration dictionary using dot notation.
    
    Args:
        config_dict: Base configuration dictionary
        overrides: Dict with keys like 'distill_student.optimization.epochs'
        
    Returns:
        Configuration dictionary with overrides applied
        
    Example:
        >>> config = {'distill_student': {'optimization': {'epochs': 100}}}
        >>> overrides = {'distill_student.optimization.epochs': 50}
        >>> result = _apply_overrides(config, overrides)
        >>> result['distill_student']['optimization']['epochs']
        50
    """
    import copy
    config = copy.deepcopy(config_dict)
    
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        current = config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
    
    return config


def get_default_dataset_path(dataset_name: str, file_type: str = 'train') -> str:
    """
    Get default data path for a dataset.
    
    Args:
        dataset_name: Dataset name
        file_type: 'train', 'val', or 'test'
        
    Returns:
        Path to dataset file
        
    Example:
        >>> path = get_default_dataset_path('citeulike', 'train')
        >>> print(path)
        data/citeulike/train.tsv
    """
    loader = get_config_loader()
    dataset_cfg = loader.load_dataset_config(dataset_name)
    
    if file_type == 'train':
        return dataset_cfg.train_path
    elif file_type == 'val':
        return dataset_cfg.validation_path or dataset_cfg.train_path
    elif file_type == 'test':
        return dataset_cfg.test_path
    else:
        raise ValueError(f"Unknown file_type: {file_type}")


def get_teacher_checkpoint_path(
    dataset_name: str,
    model_name: str,
    checkpoint_dir: str = "results/teacher_models"
) -> Path:
    """
    Get expected path for teacher model checkpoint.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name (bprmf, nmf, lgcn)
        checkpoint_dir: Base directory for checkpoints
        
    Returns:
        Path to teacher checkpoint
        
    Example:
        >>> path = get_teacher_checkpoint_path('citeulike', 'nmf')
        >>> print(path)
        results/teacher_models/citeulike/nmf_best.pt
    """
    return Path(checkpoint_dir) / dataset_name / f"{model_name}_best.pt"


def print_config_summary(config: RecDistillConfig) -> None:
    """
    Print a summary of RecDistill configuration.
    
    Args:
        config: RecDistillConfig object
    """
    train_cfg = config.distill_student
    
    print("\n" + "=" * 70)
    print("RECDISTILL CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print("\n📊 DATA")
    print(f"  Dataset: {train_cfg.dataset}")
    
    print("\n👨‍🏫 TEACHER")
    print(f"  Model: {train_cfg.teacher.model}")
    print(f"  Embedding Dim: {train_cfg.teacher.embedding_dim}")
    print(f"  Path: {train_cfg.teacher.path or 'Not specified (will be auto-located)'}")
    
    print("\n👨‍🎓 STUDENT")
    print(f"  Backbone: {train_cfg.student.backbone}")
    print(f"  Distiller: {train_cfg.distillation.strategy}")
    print(f"  Embedding Dim: {train_cfg.student.embedding_dim}")
    
    print("\n🔬 DISTILLATION")
    print(f"  Strategy: {train_cfg.distillation.strategy}")
    print(f"  Temperature: {train_cfg.distillation.temperature}")
    if hasattr(train_cfg.distillation, 'lambda_kl'):
        print(f"  Lambda KL: {train_cfg.distillation.lambda_kl}")
    if hasattr(train_cfg.distillation, 'lambda_de'):
        print(f"  Lambda DE: {train_cfg.distillation.lambda_de}")
    
    print("\n⚙️  OPTIMIZATION")
    print(f"  Epochs: {train_cfg.optimization.epochs}")
    print(f"  Batch Size: {train_cfg.optimization.batch_size}")
    print(f"  Learning Rate: {train_cfg.optimization.learning_rate}")
    print(f"  L2 Reg: {train_cfg.optimization.l2_reg}")
    print(f"  Validation Rate: {train_cfg.optimization.validation_rate}")
    
    print("\n🏃 RUNTIME")
    print(f"  Seed: {train_cfg.runtime.seed}")
    print(f"  Device: {train_cfg.runtime.device or 'Auto (cuda if available)'}")
    print(f"  Num Workers: {train_cfg.runtime.num_workers}")
    print(f"  Output Path: {train_cfg.runtime.output_path}")
    if train_cfg.runtime.wandb.get('enabled'):
        print(f"  W&B: Enabled")
    
    print("\n📈 EVALUATION")
    print(f"  K: {train_cfg.evaluation.k}")
    print(f"  Every N epochs: {train_cfg.evaluation.every}")
    print(f"  Validation Only: {train_cfg.evaluation.val_only}")
    print(f"  Selection Metric: {train_cfg.evaluation.selection_metric}")
    
    print("\n⏸️  EARLY STOPPING")
    if train_cfg.early_stopping:
        print(f"  Enabled: {train_cfg.early_stopping.enabled}")
        print(f"  Metric: {train_cfg.early_stopping.metric}")
        print(f"  Patience: {train_cfg.early_stopping.patience}")
        print(f"  Min Delta: {train_cfg.early_stopping.min_delta}")
    else:
        print(f"  Disabled")
    
    print("\n" + "=" * 70 + "\n")


def validate_config(config: RecDistillConfig) -> bool:
    """
    Validate RecDistill configuration for common issues.
    
    Args:
        config: RecDistillConfig object
        
    Returns:
        True if configuration is valid, False otherwise
    """
    train_cfg = config.distill_student
    issues = []
    
    # Check embedding dimensions
    if train_cfg.student.embedding_dim >= train_cfg.teacher.embedding_dim:
        issues.append(
            f"⚠️  Student embedding dim ({train_cfg.student.embedding_dim}) "
            f"should be smaller than teacher ({train_cfg.teacher.embedding_dim})"
        )
    
    # Check optimization parameters
    if train_cfg.optimization.epochs < train_cfg.optimization.validation_rate:
        issues.append(
            f"⚠️  Validation rate ({train_cfg.optimization.validation_rate}) "
            f"should be smaller than total epochs ({train_cfg.optimization.epochs})"
        )
    
    # Check early stopping if enabled
    if train_cfg.early_stopping and train_cfg.early_stopping.enabled:
        if train_cfg.early_stopping.patience > train_cfg.optimization.epochs:
            issues.append(
                f"⚠️  Early stopping patience ({train_cfg.early_stopping.patience}) "
                f"should be smaller than total epochs ({train_cfg.optimization.epochs})"
            )
    
    if issues:
        print("\n⚠️  Configuration warnings:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("Configuration validation passed.")
    return True


def list_example_experiments() -> None:
    """Print available pre-configured experiments."""
    loader = get_config_loader()
    
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)
    
    datasets = loader.list_datasets()
    models = loader.list_models()
    distillers = loader.list_distillers()
    
    print("\nPredefine combinations:")
    for dataset in datasets:
        for distiller in distillers:
            for teacher in models['teacher']:
                print(f"  python train_distiller.py \\")
                print(f"    --dataset {dataset} \\")
                print(f"    --teacher {teacher} \\")
                print(f"    --distiller {distiller}")
    
    print("\n" + "=" * 70 + "\n")
