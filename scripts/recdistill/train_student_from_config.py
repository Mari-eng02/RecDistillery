"""
Launch student training from a YAML configuration file.

Usage:
    python scripts/recdistill/train_student_from_config.py --config path/to/config.yaml
    python scripts/recdistill/train_student_from_config.py --config path/to/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import RecDistillConfig, get_config_loader
from recdistill.config_integration import (
    load_recdistill_config_from_file,
    load_recdistill_experiment,
    normalize_recdistill_config,
    recdistill_config_to_dict,
)
from recdistill.experiment_runner import RecDistillExperimentRunner
from recdistill.paths import DISTILLED_STUDENT_EXT, distilled_student_artifact_path
from recdistill.model_validation import validate_distillation_request, validate_recdistill_config_dict
from recdistill.tracking import resolve_wandb_logger
from recdistill.training import set_seed


def _add_arg(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


_DISTILLER_METHODS = {"DE", "RRD", "UNKD", "HTD", "FTD"}


def _config_value(primary: dict, secondary: dict, key: str, default=None):
    if key in primary:
        return primary[key]
    if key in secondary:
        return secondary[key]
    return default


def _distiller_methods(student_model: Any) -> set[str] | None:
    if student_model is None:
        return None

    normalized = str(student_model).upper().replace("+", "_").replace("-", "_")
    if normalized == "COMPOSITE":
        return set(_DISTILLER_METHODS)

    methods = {part for part in normalized.split("_") if part}
    unknown = methods - _DISTILLER_METHODS
    if unknown:
        raise ValueError(
            f"Unsupported student.model: {student_model}. "
            "Supported methods: DE, RRD, UNKD, HTD, FTD, and composites joined with '+' or '_'."
        )
    if "HTD" in methods and "FTD" in methods:
        raise ValueError("HTD and FTD cannot be active at the same time: the training runner supports one topology distiller.")
    return methods


def _lambda_for_method(
    *,
    method: str,
    active_methods: set[str] | None,
    distill_conf: dict,
    student_conf: dict,
    key: str,
    default: float = 0.0,
) -> float:
    value = float(_config_value(distill_conf, student_conf, key, default))
    if active_methods is not None and method not in active_methods and value != 0.0:
        model = student_conf.get("model")
        raise ValueError(
            f"train_student.student.model is {model!r}, but {key}={value} would activate {method}. "
            f"Use a composite model name that includes {method} or set {key}: 0.0."
        )
    if active_methods is not None and method not in active_methods:
        return 0.0
    return value


def _normalize_train_conf(config: dict) -> dict:
    return config.get("train_student", config)


def _set_dotted(root: dict, dotted_key: str, value: Any) -> None:
    cursor = root
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def _safe_slug(value: str, max_len: int = 80) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.=")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _teacher_model_from_path(teacher_conf: dict) -> str | None:
    raw_path = teacher_conf.get("path")
    if not raw_path:
        return None
    parts = Path(str(raw_path)).parts
    lowered = [part.lower() for part in parts]
    try:
        teachers_idx = lowered.index("teachers")
        return parts[teachers_idx + 2]
    except (ValueError, IndexError):
        return Path(str(raw_path)).stem or None


def _teacher_model_label(teacher_conf: dict) -> str:
    return str(_teacher_model_from_path(teacher_conf) or teacher_conf.get("model") or "teacher")


def _experiment_tuple(train_conf: dict) -> tuple[str, str, str, str, str, str]:
    student_conf = train_conf.get("student", {}) or {}
    teacher_conf = train_conf.get("teacher", {}) or {}
    dataset = str(train_conf.get("dataset") or "dataset")
    distiller = str(student_conf.get("model") or "DE")
    teacher_framework = str(teacher_conf.get("framework") or "recbole")
    teacher = _teacher_model_label(teacher_conf)
    student_framework = str(student_conf.get("framework") or "recbole")
    student = str(student_conf.get("backbone") or "student")
    return (
        _safe_slug(distiller.lower()),
        _safe_slug(teacher_framework.lower()),
        _safe_slug(teacher),
        _safe_slug(student_framework.lower()),
        _safe_slug(student),
        _safe_slug(dataset.lower()),
    )


def _default_results_dir(train_conf: dict, strategy: str) -> Path:
    distiller, teacher_framework, teacher, student_framework, student, dataset = _experiment_tuple(train_conf)
    return Path("results") / "recdistill" / distiller / teacher_framework / teacher / student_framework / student / dataset


def _distilled_path_from_train_conf(train_conf: dict, *, strategy: str) -> Path:
    dataset = str(train_conf.get("dataset", "dataset"))
    teacher_conf = train_conf.get("teacher", {}) or {}
    student_conf = train_conf.get("student", {}) or {}
    path = distilled_student_artifact_path(
        distiller=str(student_conf.get("model") or "DE"),
        teacher_framework=teacher_conf.get("framework"),
        teacher_model=_teacher_model_label(teacher_conf),
        student_framework=student_conf.get("framework"),
        student_model=str(student_conf.get("backbone") or "student"),
        dataset=dataset,
        embedding_dim=int(student_conf.get("embedding_dim", 0)),
        strategy=strategy,
    )
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _search_best_output_path(train_conf: dict) -> Path:
    runtime_conf = train_conf.get("runtime", {}) or {}
    if runtime_conf.get("output_path"):
        return Path(runtime_conf["output_path"])
    path = _distilled_path_from_train_conf(train_conf, strategy="best")
    if not path.stem.endswith("_best"):
        path = path.with_name(f"{path.stem}_best{path.suffix}")
    return path


def _compose_imported_teacher_experiment(
    *,
    dataset_name: str,
    teacher_path: str,
    distiller_strategy: str,
    student_backbone: str,
    student_framework: str,
) -> dict:
    loader = get_config_loader()
    dataset = loader.load_dataset_config(dataset_name)
    student_cfg = loader.load_model_config("student", student_backbone, framework=student_framework)
    template = copy.deepcopy(loader._load_yaml(loader.root / "experiments" / f"recdistill_template_{distiller_strategy}.yaml"))
    distiller_cfg = loader._load_yaml(loader.root / "distillers" / f"{distiller_strategy}.yaml")

    teacher_model = Path(teacher_path).stem
    teacher_framework = "imported"
    parts = Path(teacher_path).parts
    try:
        teachers_index = [part.lower() for part in parts].index("teachers")
        teacher_framework = parts[teachers_index + 1]
        teacher_model = parts[teachers_index + 2]
    except (ValueError, IndexError):
        pass

    template["train_student"]["dataset"] = dataset.name
    template["train_student"]["teacher"].update(
        {
            "framework": teacher_framework,
            "model": teacher_model,
            "embedding_dim": student_cfg.embedding_dim + 1,
            "path": teacher_path,
            "format": "checkpoint",
        }
    )
    template["train_student"]["student"]["framework"] = student_cfg.framework
    template["train_student"]["student"]["backbone"] = student_cfg.backbone
    template["train_student"]["student"]["embedding_dim"] = student_cfg.embedding_dim
    loader._copy_model_specific_fields(student_cfg, template["train_student"]["student"])
    template["train_student"]["distillation"].update(distiller_cfg)
    template["train_student"]["distillation"]["strategy"] = str(
        template["train_student"]["distillation"].get("strategy", distiller_strategy)
    ).upper()
    template["train_student"]["student"]["model"] = template["train_student"]["distillation"]["strategy"]
    runtime = template["train_student"].setdefault("runtime", {})
    runtime["output_path"] = str(
        distilled_student_artifact_path(
            distiller=distiller_strategy,
            teacher_framework=teacher_framework,
            teacher_model=teacher_model,
            student_framework=student_cfg.framework,
            student_model=student_cfg.backbone,
            dataset=dataset.name,
            embedding_dim=student_cfg.embedding_dim,
            strategy="fixed",
        ).relative_to(REPO_ROOT)
    ).replace("\\", "/")
    return recdistill_config_to_dict(normalize_recdistill_config(template))


def _copy_artifact_with_history(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.parent.parent.joinpath("perf").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    source_history = source_path.with_suffix(".history.json")
    if source_history.exists():
        shutil.copy2(source_history, destination_path.with_suffix(".history.json"))


def _tracked_results_dir(train_conf: dict, config_path: Path) -> Path:
    distiller, teacher_framework, teacher, student_framework, student, dataset = _experiment_tuple(train_conf)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = f"{config_path.stem}__{stamp}_{time.time_ns()}"
    return (
        Path("results")
        / "recdistill"
        / distiller
        / teacher_framework
        / teacher
        / student_framework
        / student
        / dataset
        / "tracked"
        / unique
    )


def _expand_range_spec(spec: Any) -> list[Any]:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Grid-search list values cannot be empty.")
        return spec

    if isinstance(spec, dict):
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list) or not values:
                raise ValueError("Grid-search 'values' must be a non-empty list.")
            return values
        if {"start", "stop", "step"}.issubset(spec.keys()):
            start = spec["start"]
            stop = spec["stop"]
            step = spec["step"]
            if step == 0:
                raise ValueError("Grid-search 'step' cannot be zero.")
            values = []
            current = start
            epsilon = abs(step) * 1e-9
            if step > 0:
                while current <= stop + epsilon:
                    values.append(current)
                    current += step
            else:
                while current >= stop - epsilon:
                    values.append(current)
                    current += step
            if not values:
                raise ValueError(f"Grid-search range generated no values: {spec}")
            if all(isinstance(v, int) for v in (start, stop, step)):
                values = [int(v) for v in values]
            return values
    raise ValueError(
        "Each grid-search parameter must be either a list, "
        "a {'values': [...]} object, or a {'start','stop','step'} range object."
    )


def _materialize_grid_overrides(config: dict) -> tuple[list[dict[str, Any]], dict]:
    train_conf = _normalize_train_conf(config)
    optim_conf = train_conf.get("optimization", {}) or {}
    grid_conf = optim_conf.get("grid_search", train_conf.get("grid_search", {})) or {}
    if not grid_conf.get("enabled", False):
        return [{}], grid_conf

    parameters = grid_conf.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "train_student.optimization.grid_search.parameters must be a non-empty mapping when enabled=true."
        )

    keys = list(parameters.keys())
    value_lists = [_expand_range_spec(parameters[key]) for key in keys]
    overrides = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    if not overrides:
        raise ValueError("Grid search produced zero combinations.")
    return overrides, grid_conf


def _build_run_config(
    base_config: dict,
    overrides: dict[str, Any],
    run_idx: int,
    total_runs: int,
    grid_conf: dict,
    strategy: str,
    create_dirs: bool = True,
) -> tuple[dict, str]:
    config = copy.deepcopy(base_config)
    train_conf = _normalize_train_conf(config)

    for dotted_key, value in overrides.items():
        normalized_key = dotted_key
        if normalized_key.startswith("train_student."):
            normalized_key = normalized_key[len("train_student.") :]
        _set_dotted(train_conf, normalized_key, value)

    run_id = f"run_{run_idx:04d}"
    dataset = str(train_conf.get("dataset", "dataset"))
    teacher_model = str(train_conf.get("teacher", {}).get("model", "teacher"))
    student_backbone = str(train_conf.get("student", {}).get("backbone", "student"))
    canonical_output_path = _distilled_path_from_train_conf(train_conf, strategy="fixed")
    default_base_output_dir = canonical_output_path.parent.parent
    if strategy == "grid":
        default_base_output_dir = _distilled_path_from_train_conf(train_conf, strategy="grid").parent.parent
    base_output_dir = Path(grid_conf.get("output_dir") or default_base_output_dir)
    if create_dirs:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        (base_output_dir / "wei").mkdir(parents=True, exist_ok=True)
        (base_output_dir / "perf").mkdir(parents=True, exist_ok=True)

    override_slug = "_".join(f"{key}={value}" for key, value in overrides.items()) or "single"
    override_slug = _safe_slug(override_slug, max_len=90)
    if strategy == "fixed" and not overrides:
        output_path = canonical_output_path
    else:
        output_filename = f"{run_id}_{override_slug}{DISTILLED_STUDENT_EXT}"
        output_path = base_output_dir / "wei" / output_filename
    runtime_conf = train_conf.get("runtime", {}) or {}
    existing_output_path = runtime_conf.get("output_path")
    if strategy == "fixed" and existing_output_path:
        output_path = Path(existing_output_path)
    else:
        _set_dotted(train_conf, "runtime.output_path", str(output_path))

    wandb_conf = train_conf.get("runtime", {}).get("wandb", {})
    if isinstance(wandb_conf, dict) and wandb_conf.get("enabled", False):
        base_name = wandb_conf.get("run_name") or f"{dataset}_{teacher_model}_{student_backbone}"
        wandb_conf["run_name"] = f"{base_name}__{run_id}"

    metadata = {
        "run_id": run_id,
        "run_idx": run_idx,
        "total_runs": total_runs,
        "output_path": str(output_path),
        "base_output_dir": str(base_output_dir),
        "overrides": overrides,
        "override_slug": override_slug,
    }
    return config, metadata


def build_command(config: dict) -> list[str]:
    run_conf = config.get("run", {})
    try:
        validate_recdistill_config_dict(config)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    train_conf = _normalize_train_conf(config)

    script = run_conf.get("script", "scripts/recdistill/train_student.py")
    python_bin = run_conf.get("python", sys.executable)

    teacher_conf = train_conf.get("teacher", {})
    student_conf = train_conf.get("student", {})
    distill_conf = train_conf.get("distillation", {})
    optim_conf = train_conf.get("optimization", {})
    runtime_conf = train_conf.get("runtime", {})
    eval_conf = train_conf.get("evaluation", {})
    early_conf = train_conf.get("early_stopping", {})
    wandb_conf = runtime_conf.get("wandb", {})

    dataset = train_conf.get("dataset")
    if not dataset:
        raise ValueError("Missing required field: train_student.dataset")

    cmd = [str(python_bin), str(script), "--dataset", str(dataset)]

    if teacher_conf.get("path"):
        _add_arg(cmd, "--teacher-path", teacher_conf.get("path"))
    else:
        _add_arg(cmd, "--teacher-model", teacher_conf.get("model"))
        _add_arg(cmd, "--teacher-embedding-dim", teacher_conf.get("embedding_dim"))
        _add_arg(cmd, "--teacher-framework", teacher_conf.get("framework"))
        _add_arg(cmd, "--teacher-format", teacher_conf.get("format"))
        _add_arg(cmd, "--teacher-adapter", teacher_conf.get("adapter"))
        _add_arg(cmd, "--teacher-user-embeddings-path", teacher_conf.get("user_embeddings_path"))
        _add_arg(cmd, "--teacher-item-embeddings-path", teacher_conf.get("item_embeddings_path"))
        _add_arg(cmd, "--teacher-score-matrix-path", teacher_conf.get("score_matrix_path"))
        _add_arg(cmd, "--teacher-topk-items-path", teacher_conf.get("topk_items_path"))
        _add_arg(cmd, "--teacher-topk-scores-path", teacher_conf.get("topk_scores_path"))

    noise_conf = teacher_conf.get("noise", {}) if isinstance(teacher_conf.get("noise", {}), dict) else {}
    _add_arg(cmd, "--teacher-noise-scale", noise_conf.get("scale", teacher_conf.get("noise_scale")))
    _add_arg(cmd, "--teacher-noise-target", noise_conf.get("target", teacher_conf.get("noise_target")))
    _add_arg(cmd, "--teacher-noise-seed", noise_conf.get("seed", teacher_conf.get("noise_seed")))

    student_backbone = student_conf.get("backbone")
    if not student_backbone:
        raise ValueError("Missing required field: train_student.student.backbone")
    supported_student_backbones = {"BPRMF", "BPR", "LINE", "LGCN", "LIGHTGCN", "NGCF", "DGCF", "SGL", "SPECTRALCF", "SPECTRAL_CF", "NMF", "NFM", "NEUMF"}
    if str(student_backbone).upper() not in supported_student_backbones:
        raise ValueError(
            f"Unsupported train_student.student.backbone: {student_backbone}. "
            "Currently supported: BPRMF, LINE, LGCN, NGCF, DGCF, SGL, SpectralCF, NMF."
        )
    _add_arg(cmd, "--student-backbone", student_backbone)
    _add_arg(cmd, "--student-framework", student_conf.get("framework"))

    _add_arg(cmd, "--student-embedding-dim", student_conf.get("embedding_dim"))
    _add_arg(
        cmd,
        "--lightgcn-layers",
        student_conf.get("num_layers", student_conf.get("layers", student_conf.get("lightgcn_layers"))),
    )
    mlp_dims = student_conf.get("mlp_hidden_size", student_conf.get("mlp_dims", student_conf.get("neumf_mlp_dims")))
    if isinstance(mlp_dims, list):
        mlp_dims = ",".join(str(value) for value in mlp_dims)
    _add_arg(cmd, "--neumf-mlp-dims", mlp_dims)
    _add_arg(cmd, "--neumf-dropout", student_conf.get("dropout", student_conf.get("neumf_dropout")))

    _add_arg(cmd, "--epochs", optim_conf.get("epochs"))
    _add_arg(cmd, "--batch-size", optim_conf.get("batch_size"))
    _add_arg(cmd, "--learning-rate", optim_conf.get("learning_rate"))
    _add_arg(cmd, "--l2-reg", optim_conf.get("l2_reg"))

    student_model = student_conf.get("model")
    active_methods = _distiller_methods(student_model)

    _add_arg(
        cmd,
        "--lambda-de",
        _lambda_for_method(
            method="DE",
            active_methods=active_methods,
            distill_conf=distill_conf,
            student_conf=student_conf,
            key="lambda_de",
        ),
    )
    _add_arg(cmd, "--num-experts", distill_conf.get("num_experts", student_conf.get("num_experts")))
    _add_arg(cmd, "--temperature", distill_conf.get("temperature", student_conf.get("temperature")))

    _add_arg(
        cmd,
        "--lambda-rrd",
        _lambda_for_method(
            method="RRD",
            active_methods=active_methods,
            distill_conf=distill_conf,
            student_conf=student_conf,
            key="lambda_rrd",
        ),
    )
    rrd_conf = distill_conf.get("rrd", {}) if isinstance(distill_conf.get("rrd", {}), dict) else {}
    _add_arg(
        cmd,
        "--rrd-interesting-size",
        rrd_conf.get("interesting_size", distill_conf.get("interesting_size", student_conf.get("rrd_interesting_size"))),
    )
    _add_arg(
        cmd,
        "--rrd-uninteresting-size",
        rrd_conf.get("uninteresting_size", distill_conf.get("uninteresting_size", student_conf.get("rrd_uninteresting_size"))),
    )
    _add_arg(
        cmd,
        "--rrd-temperature",
        rrd_conf.get("temperature", distill_conf.get("rrd_temperature", student_conf.get("rrd_temperature"))),
    )
    _add_arg(
        cmd,
        "--rrd-teacher-topk",
        rrd_conf.get("teacher_topk", distill_conf.get("teacher_topk", student_conf.get("rrd_teacher_topk"))),
    )

    unkd_conf = distill_conf.get("unkd", {}) if isinstance(distill_conf.get("unkd", {}), dict) else {}
    _add_arg(
        cmd,
        "--lambda-unkd",
        _lambda_for_method(
            method="UNKD",
            active_methods=active_methods,
            distill_conf=distill_conf,
            student_conf=student_conf,
            key="lambda_unkd",
        ),
    )
    _add_arg(cmd, "--unkd-sample-num", unkd_conf.get("sample_num", student_conf.get("unkd_sample_num")))
    _add_arg(cmd, "--unkd-group-count", unkd_conf.get("group_count", student_conf.get("unkd_group_count")))
    _add_arg(
        cmd,
        "--unkd-popularity-lambda",
        unkd_conf.get("popularity_lambda", student_conf.get("unkd_popularity_lambda")),
    )
    _add_arg(cmd, "--unkd-rank-top-k", unkd_conf.get("rank_top_k", student_conf.get("unkd_rank_top_k")))
    _add_arg(
        cmd,
        "--unkd-rank-temperature",
        unkd_conf.get("rank_temperature", student_conf.get("unkd_rank_temperature")),
    )

    topology_conf = distill_conf.get("topology", {}) if isinstance(distill_conf.get("topology", {}), dict) else {}
    htd_conf = distill_conf.get("htd", {}) if isinstance(distill_conf.get("htd", {}), dict) else {}
    ftd_conf = distill_conf.get("ftd", {}) if isinstance(distill_conf.get("ftd", {}), dict) else {}

    td_type = topology_conf.get("type")
    if td_type is None:
        if active_methods is not None and "FTD" in active_methods:
            td_type = "FTD"
        elif active_methods is not None and "HTD" in active_methods:
            td_type = "HTD"
        else:
            td_type = "HTD"
    td_type = str(td_type).upper()
    _add_arg(cmd, "--td-type", td_type)

    lambda_td = topology_conf.get(
        "lambda_td",
        htd_conf.get("lambda_td", ftd_conf.get("lambda_td", distill_conf.get("lambda_td", student_conf.get("lambda_td", 0.0)))),
    )
    lambda_td = float(lambda_td)
    if active_methods is not None and {"HTD", "FTD"}.isdisjoint(active_methods) and lambda_td != 0.0:
        raise ValueError(
            f"train_student.student.model is {student_model!r}, but lambda_td={lambda_td} would activate topology distillation. "
            "Use HTD/FTD in student.model or set lambda_td: 0.0."
        )
    if active_methods is not None and lambda_td != 0.0 and td_type in {"HTD", "FTD"} and td_type not in active_methods:
        raise ValueError(
            f"train_student.student.model is {student_model!r}, but topology.type is {td_type}. "
            f"Use student.model: {td_type} or set topology.type to a method included in student.model."
        )
    if active_methods is not None and {"HTD", "FTD"}.isdisjoint(active_methods):
        lambda_td = 0.0
    _add_arg(
        cmd,
        "--lambda-td",
        lambda_td,
    )
    _add_arg(cmd, "--htd-alpha", topology_conf.get("alpha", htd_conf.get("alpha", student_conf.get("htd_alpha"))))
    _add_arg(
        cmd,
        "--htd-num-groups",
        topology_conf.get("num_groups", htd_conf.get("num_groups", student_conf.get("htd_num_groups"))),
    )
    _add_arg(
        cmd,
        "--htd-topology-mode",
        topology_conf.get("topology_mode", htd_conf.get("topology_mode", student_conf.get("htd_topology_mode"))),
    )
    _add_arg(
        cmd,
        "--htd-initial-tau",
        topology_conf.get("initial_tau", htd_conf.get("initial_tau", student_conf.get("htd_initial_tau"))),
    )
    _add_arg(
        cmd,
        "--htd-min-tau",
        topology_conf.get("min_tau", htd_conf.get("min_tau", student_conf.get("htd_min_tau"))),
    )
    _add_arg(
        cmd,
        "--htd-decay-epochs",
        topology_conf.get("decay_epochs", htd_conf.get("decay_epochs", student_conf.get("htd_decay_epochs"))),
    )
    _add_arg(
        cmd,
        "--td-entity-sample-size",
        topology_conf.get(
            "entity_sample_size",
            distill_conf.get("td_entity_sample_size", student_conf.get("td_entity_sample_size")),
        ),
    )

    _add_arg(cmd, "--seed", runtime_conf.get("seed"))
    _add_arg(cmd, "--device", runtime_conf.get("device"))
    _add_arg(cmd, "--num-workers", runtime_conf.get("num_workers"))
    _add_arg(cmd, "--output-path", runtime_conf.get("output_path"))
    _add_arg(cmd, "--output-strategy", runtime_conf.get("output_strategy"))
    _add_arg(cmd, "--save-every", runtime_conf.get("save_every"))

    if eval_conf.get("enabled") is False:
        cmd.append("--skip-eval")
    _add_arg(cmd, "--eval-k", eval_conf.get("k"))
    _add_arg(cmd, "--eval-every", eval_conf.get("every"))
    _add_arg(cmd, "--eval-batch-size", eval_conf.get("batch_size"))
    if eval_conf.get("val_only", False):
        cmd.append("--eval-val-only")
    _add_arg(cmd, "--selection-split", eval_conf.get("selection_split"))
    _add_arg(cmd, "--selection-metric", eval_conf.get("selection_metric"))
    if eval_conf.get("assert_no_train_leak", False):
        cmd.append("--assert-no-train-leak")

    if early_conf.get("enabled", False):
        cmd.append("--early-stop")
        _add_arg(cmd, "--early-stop-mode", early_conf.get("mode"))
        _add_arg(cmd, "--early-stop-metric", early_conf.get("metric"))
        _add_arg(cmd, "--early-stop-patience", early_conf.get("patience"))
        _add_arg(cmd, "--early-stop-min-delta", early_conf.get("min_delta"))
        _add_arg(cmd, "--early-stop-warmup", early_conf.get("warmup"))
        if early_conf.get("restore_best", False):
            cmd.append("--early-stop-restore-best")

    if wandb_conf.get("enabled", False):
        cmd.append("--wandb-log")
        _add_arg(cmd, "--wandb-project", wandb_conf.get("project"))
        _add_arg(cmd, "--wandb-entity", wandb_conf.get("entity"))
        _add_arg(cmd, "--wandb-run-name", wandb_conf.get("run_name"))
        tags = wandb_conf.get("tags")
        if isinstance(tags, list) and tags:
            _add_arg(cmd, "--wandb-tags", ",".join(str(tag) for tag in tags))
        _add_arg(cmd, "--wandb-group", wandb_conf.get("group"))
        _add_arg(cmd, "--wandb-notes", wandb_conf.get("notes"))

    extra_args = runtime_conf.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise ValueError("train_student.runtime.extra_args must be a list.")
    cmd.extend(str(arg) for arg in extra_args)

    return cmd


def _read_history_metrics(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        return {}
    last_row = history[-1]
    metrics: dict[str, Any] = {
        "final_epoch": last_row.get("epoch"),
        "final_total_loss": last_row.get("total_loss"),
    }
    for row in reversed(history):
        if "val_ndcg" in row:
            metrics["val_precision"] = row.get("val_precision")
            metrics["val_recall"] = row.get("val_recall")
            metrics["val_ndcg"] = row.get("val_ndcg")
            metrics["val_hr"] = row.get("val_hr")
            metrics["test_precision"] = row.get("test_precision")
            metrics["test_recall"] = row.get("test_recall")
            metrics["test_ndcg"] = row.get("test_ndcg")
            metrics["test_hr"] = row.get("test_hr")
            metrics["last_selection_score"] = row.get("selection_score")
            break
    return metrics


def _read_checkpoint_summary(output_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not output_path.exists():
        return summary

    try:
        import torch

        payload = torch.load(output_path, map_location="cpu")
        if isinstance(payload, dict):
            summary["best_epoch"] = payload.get("best_epoch")
            summary["best_selection_score"] = payload.get("best_selection_score")
            summary["early_stopped"] = payload.get("early_stopped")
            summary["early_stop_reason"] = payload.get("early_stop_reason")
            summary["early_best_epoch"] = payload.get("early_best_epoch")
            summary["early_best_value"] = payload.get("early_best_value")
    except Exception:
        pass
    return summary


def _write_recap_files(records: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run_recap.json"
    tsv_path = output_dir / "run_recap.tsv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    field_order: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in field_order:
                field_order.append(key)

    with tsv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=field_order, delimiter="\t")
        writer.writeheader()
        for row in records:
            serializable_row = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable_row)
    return json_path, tsv_path


def _sort_records_for_recap(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple:
        status = 0 if row.get("status") == "completed" else 1
        selection = row.get("best_selection_score")
        loss = row.get("final_total_loss")
        selection_rank = float(selection) if selection is not None else float("-inf")
        loss_rank = -float(loss) if loss is not None else float("-inf")
        return (status, -selection_rank, -loss_rank)

    return sorted(records, key=sort_key)


def _run_with_framework_runner(config: dict) -> dict[str, Any]:
    """Execute one resolved training config through the reusable framework API."""
    validated_config = RecDistillConfig(**normalize_recdistill_config(config))
    runner = RecDistillExperimentRunner.from_config(validated_config)
    set_seed(int(runner.args.seed))
    runner.wandb_logger = resolve_wandb_logger(runner.args, runner.run_config())
    return runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch RecDistill student training from a validated config.")
    parser.add_argument("--config", help="Config preset or direct YAML/JSON configuration file")
    parser.add_argument("--dataset", help="Dataset name for ConfigLoader composition")
    parser.add_argument("--teacher-path", help="Imported teacher artifact path")
    parser.add_argument("--teacher-model", "--teacher", dest="teacher_model", help="Teacher model for ConfigLoader composition")
    parser.add_argument("--teacher-framework", default=None, choices=["recbole", "elliot", "lenskit"], help="Teacher framework for ConfigLoader composition")
    parser.add_argument("--distiller", help="Distillation strategy for ConfigLoader composition")
    parser.add_argument("--student-backbone", "--student", dest="student_backbone", help="Student backbone for ConfigLoader composition")
    parser.add_argument("--student-framework", default=None, choices=["recbole", "elliot", "lenskit"], help="Student framework for ConfigLoader composition")
    parser.add_argument("--output-strategy", choices=["fixed", "best", "bayesian", "tracked"], help="Output strategy namespace")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved run plan")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Store outputs in a unique tracked results directory and record timing information.",
    )
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw_text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            config = normalize_recdistill_config(json.loads(raw_text))
        else:
            if yaml is None:
                raise ModuleNotFoundError(
                    "PyYAML is not installed. Use a JSON config file or install PyYAML to read YAML."
                )
            config = recdistill_config_to_dict(load_recdistill_config_from_file(config_path))
    else:
        args.student_framework = args.student_framework or "recbole"
        has_imported_teacher = bool(args.teacher_path)
        missing = [
            name
            for name, value in {
                "--dataset": args.dataset,
                "--teacher-path or --teacher-model": args.teacher_path or args.teacher_model,
                "--distiller": args.distiller,
                "--student-backbone": args.student_backbone if has_imported_teacher else (args.student_backbone or args.teacher_model),
            }.items()
            if not value
        ]
        if missing:
            parser.error("--config or --dataset/--distiller plus either --teacher-path or --teacher-model is required")
        if has_imported_teacher:
            if args.teacher_model or args.teacher_framework:
                parser.error("Use only --teacher-path for imported teachers; do not pass --teacher-model or --teacher-framework.")
            try:
                validate_distillation_request(
                    teacher_framework=None,
                    teacher_model=None,
                    student_framework=args.student_framework,
                    student_backbone=args.student_backbone,
                    distiller=args.distiller,
                    validate_teacher=False,
                )
            except ValueError as exc:
                parser.error(str(exc))
            config = _compose_imported_teacher_experiment(
                dataset_name=args.dataset,
                teacher_path=args.teacher_path,
                distiller_strategy=args.distiller,
                student_backbone=args.student_backbone,
                student_framework=args.student_framework,
            )
            teacher_label = Path(args.teacher_path).stem
            teacher_framework_label = "imported"
        else:
            args.teacher_framework = args.teacher_framework or "recbole"
            try:
                validate_distillation_request(
                    teacher_framework=args.teacher_framework,
                    teacher_model=args.teacher_model,
                    student_framework=args.student_framework,
                    student_backbone=args.student_backbone or args.teacher_model,
                    distiller=args.distiller,
                )
            except ValueError as exc:
                parser.error(str(exc))
            config = recdistill_config_to_dict(
                load_recdistill_experiment(
                    dataset_name=args.dataset,
                    teacher_model=args.teacher_model,
                    teacher_framework=args.teacher_framework,
                    distiller_strategy=args.distiller,
                    student_backbone=args.student_backbone,
                    student_framework=args.student_framework,
                )
            )
            teacher_label = args.teacher_model
            teacher_framework_label = args.teacher_framework
        preset_path = get_config_loader().save_generated_preset(
            kind="recdistill",
            family="generated",
            name=f"{args.distiller}_{teacher_framework_label}_{teacher_label}_{args.student_framework}_{args.student_backbone or teacher_label}_{args.dataset}",
            path_parts=[
                args.distiller,
                teacher_framework_label,
                teacher_label,
                args.student_framework,
                args.student_backbone or teacher_label,
                args.dataset,
            ],
            config=config,
        )
        print(f"Generated RecDistill config saved to: {preset_path}")
        config_path = preset_path

    run_conf = config.get("run", {})
    train_conf = _normalize_train_conf(config)
    if args.output_strategy:
        _set_dotted(train_conf, "runtime.output_strategy", args.output_strategy)
    optim_conf = train_conf.get("optimization", {}) or {}
    optuna_conf = (
        optim_conf.get("optuna")
        or train_conf.get("optuna")
        or config.get("optuna")
        or {}
    )

    config_dry_run = bool(run_conf.get("dry_run", False))
    dry_run = args.dry_run or config_dry_run

    if bool(optuna_conf.get("enabled", False)):
        python_bin = run_conf.get("python", sys.executable)
        optuna_script = run_conf.get("optuna_script", "scripts/recdistill/run_optuna.py")
        optuna_cmd = [str(python_bin), str(optuna_script), "--config", str(config_path)]

        printable_cmd = " ".join(shlex.quote(part) for part in optuna_cmd)
        print(f"Resolved Optuna CLI:\n{printable_cmd}\n")
        if dry_run:
            print("Dry-run enabled. Optuna run not executed.")
            return
        subprocess.run(optuna_cmd, check=True)
        return

    overrides_list, grid_conf = _materialize_grid_overrides(config)
    total_runs = len(overrides_list)
    fail_fast = bool(grid_conf.get("fail_fast", False))
    strategy = "grid" if bool(grid_conf.get("enabled", False)) else "fixed"
    if args.track:
        grid_conf = dict(grid_conf)
        grid_conf["output_dir"] = str(_tracked_results_dir(train_conf, config_path))

    records: list[dict[str, Any]] = []
    last_output_dir: Path | None = None

    for run_idx, overrides in enumerate(overrides_list, start=1):
        run_config, metadata = _build_run_config(
            base_config=config,
            overrides=overrides,
            run_idx=run_idx,
            total_runs=total_runs,
            grid_conf=grid_conf,
            strategy=strategy,
            create_dirs=not dry_run,
        )
        try:
            validate_recdistill_config_dict(run_config)
        except ValueError as exc:
            parser.error(str(exc))
        output_path = Path(metadata["output_path"])
        last_output_dir = Path(metadata["base_output_dir"])

        cmd = build_command(run_config)
        printable_cmd = " ".join(shlex.quote(part) for part in cmd)
        print(f"[{run_idx}/{total_runs}] Equivalent CLI:\n{printable_cmd}\n")

        record: dict[str, Any] = {
            "run_id": metadata["run_id"],
            "status": "dry_run" if dry_run else "pending",
            "output_path": str(output_path),
            "training_time_sec": 0.0,
            "avg_epoch_time_sec": 0.0,
            **{f"param.{key}": value for key, value in overrides.items()},
        }

        if dry_run:
            records.append(record)
            continue

        started = time.time()
        exit_code = 0
        error_message = None
        try:
            _run_with_framework_runner(run_config)
            status = "completed"
        except Exception as exc:
            status = "failed"
            exit_code = 1
            error_message = str(exc)

        elapsed = time.time() - started
        record["status"] = status
        record["training_time_sec"] = round(elapsed, 3)
        record["exit_code"] = exit_code
        record["error"] = error_message

        history_metrics = _read_history_metrics(output_path.with_suffix(".history.json"))
        checkpoint_metrics = _read_checkpoint_summary(output_path)
        record.update(history_metrics)
        record.update(checkpoint_metrics)
        final_epoch = record.get("final_epoch") or checkpoint_metrics.get("best_epoch") or 0
        try:
            final_epoch_int = int(final_epoch)
        except (TypeError, ValueError):
            final_epoch_int = 0
        if final_epoch_int > 0:
            record["avg_epoch_time_sec"] = round(elapsed / final_epoch_int, 6)
        record["history_path"] = str(output_path.with_suffix(".history.json"))
        record["final_checkpoint_path"] = str(output_path) if output_path.exists() else None
        record["best_checkpoint_path"] = str(output_path) if output_path.exists() and checkpoint_metrics.get("best_epoch") else None
        records.append(record)

        if status == "failed" and fail_fast:
            print("Grid search stopped due to failure (fail_fast=true).")
            break

    if last_output_dir is None:
        raise RuntimeError("Unable to resolve output directory for recap files.")

    ranked_records = _sort_records_for_recap(records)
    if dry_run:
        print(f"{strategy.capitalize()} dry-run recap:")
        for row in ranked_records[: min(10, len(ranked_records))]:
            print(f"- {row['run_id']} | status={row.get('status')} | output_path={row.get('output_path')}")
        print("\nDry-run enabled. Recap files not written.")
        return

    json_path, tsv_path = _write_recap_files(ranked_records, last_output_dir)

    if strategy == "grid":
        best_completed = next(
            (
                row
                for row in ranked_records
                if row.get("status") == "completed" and row.get("final_checkpoint_path")
            ),
            None,
        )
        if best_completed is not None:
            source_path = Path(best_completed["final_checkpoint_path"])
            best_output_path = _search_best_output_path(train_conf)
            if source_path.exists():
                _copy_artifact_with_history(source_path, best_output_path)
                best_completed["promoted_best_checkpoint_path"] = str(best_output_path)
                best_meta_path = best_output_path.parent.parent / "perf" / "best_grid_run.json"
                best_meta_path.write_text(json.dumps(best_completed, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Promoted best grid artifact: {best_output_path}")

    print(f"{strategy.capitalize()} run recap:")
    for row in ranked_records[: min(10, len(ranked_records))]:
        print(
            f"- {row['run_id']} | status={row.get('status')} "
            f"| best_selection_score={row.get('best_selection_score')} "
            f"| val_ndcg={row.get('val_ndcg')} "
            f"| test_ndcg={row.get('test_ndcg')}"
        )
    print(f"\nSaved recap JSON: {json_path}")
    print(f"Saved recap TSV: {tsv_path}")

    if not dry_run and any(row.get("status") == "failed" for row in records):
        raise RuntimeError("One or more grid-search runs failed. Check recap files for details.")


if __name__ == "__main__":
    main()
