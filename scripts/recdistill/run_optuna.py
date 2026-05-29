"""
Optuna-based Bayesian hyperparameter optimization for RecDistill experiments.

The dispatcher accepts any canonical experiment config:

    python scripts/recdistill/run_optuna.py --config config/experiments/teacher/recbole_lgcn_citeulike_001.yaml
    python scripts/recdistill/run_optuna.py --config config/experiments/student/recbole_lgcn_citeulike_001.yaml
    python scripts/recdistill/run_optuna.py --config config/experiments/recdistill/de_citeulike_001.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import RecDistillConfig, get_config_loader
from recdistill.config_integration import normalize_recdistill_config
from recdistill.experiment_runner import RecDistillExperimentRunner
from recdistill.native_runner import NativeModelTrainingRunner, native_args_from_config
from recdistill.paths import (
    DISTILLED_STUDENT_EXT,
    RESULTS_ROOT,
    STUDENT_EXT,
    TEACHER_EXT,
    best_checkpoint_path,
    experiment_artifact_filename,
    experiment_run_dir,
    normalize_experiment_id,
)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    yaml = None


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed. Use JSON config or install PyYAML.")
        data = yaml.safe_load(raw_text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    data.setdefault("experiment", {})
    if isinstance(data["experiment"], dict):
        data["experiment"]["id"] = normalize_experiment_id(data["experiment"].get("id"), config_path=config_path)
    return data


def _detect_kind(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    kind = str(experiment.get("kind") or "").strip().lower()
    if kind in {"teacher", "student", "recdistill"}:
        return kind
    if "train_teacher" in config:
        return "teacher"
    if "train_student" in config:
        return "student"
    if "distill_student" in config:
        return "recdistill"
    raise ValueError("Cannot detect experiment kind. Set experiment.kind or use train_teacher/train_student/distill_student.")


def _train_root_name(kind: str) -> str:
    return {"teacher": "train_teacher", "student": "train_student", "recdistill": "distill_student"}[kind]


def _artifact_ext(kind: str) -> str:
    return {"teacher": TEACHER_EXT, "student": STUDENT_EXT, "recdistill": DISTILLED_STUDENT_EXT}[kind]


def _existing_run_dir_from_config_path(config_path: Path, kind: str) -> Path | None:
    try:
        resolved = config_path.resolve()
        results_root = RESULTS_ROOT.resolve()
    except OSError:
        return None
    if resolved.parent.name != "config":
        return None
    run_dir = resolved.parent.parent
    if run_dir.parent.name != kind:
        return None
    try:
        run_dir.relative_to(results_root / kind)
    except ValueError:
        return None
    return run_dir


def _train_conf(config: dict[str, Any], kind: str) -> dict[str, Any]:
    root_name = _train_root_name(kind)
    train = config.get(root_name)
    if not isinstance(train, dict):
        raise ValueError(f"Missing or invalid {root_name} section.")
    return train


def _set_dotted(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = root
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def _get_dotted(root: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = root
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _write_records_tsv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in records:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _ensure_wandb_import() -> None:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("wandb is not installed. Install it or disable --wandb-log.") from exc


def _run_trial_wandb(
    *,
    enabled: bool,
    project: str | None,
    entity: str | None,
    group: str | None,
    run_name: str,
    config: dict[str, Any],
):
    if not enabled:
        return None
    _ensure_wandb_import()
    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        config=config,
        reinit=True,
    )


def _prepare_trial_config(
    base_config: dict[str, Any],
    *,
    kind: str,
    trial_seed: int,
    sampled_params: dict[str, Any],
    output_path: Path,
    metric: str,
    val_only: bool,
    disable_training_wandb: bool,
    dataset: str | None,
    model: str | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    train = _train_conf(config, kind)

    if dataset:
        _set_dotted(train, "dataset", dataset)
    if model:
        if kind == "teacher":
            _set_dotted(train, "teacher.model", model)
        elif kind == "student":
            _set_dotted(train, "student.backbone", model)
        else:
            _set_dotted(train, "student.backbone", model)

    _set_dotted(train, "runtime.seed", int(trial_seed))
    _set_dotted(train, "runtime.output_path", str(output_path))
    _set_dotted(train, "evaluation.enabled", True)
    _set_dotted(train, "evaluation.selection_split", "val")
    _set_dotted(train, "evaluation.selection_metric", metric)
    _set_dotted(train, "evaluation.val_only", bool(val_only))

    for dotted_key, value in sampled_params.items():
        _set_dotted(train, dotted_key, value)

    if disable_training_wandb:
        _set_dotted(train, "runtime.wandb.enabled", False)

    return config


def _run_experiment(config: dict[str, Any], *, kind: str, config_path: Path | None) -> dict[str, Any]:
    if kind in {"teacher", "student"}:
        args = native_args_from_config(config, role=kind, config_path=config_path)
        return NativeModelTrainingRunner(args).run()

    resolved = get_config_loader().resolve_config_modules(config)
    validated = RecDistillConfig(**normalize_recdistill_config(resolved))
    return RecDistillExperimentRunner.from_config(validated).run()


def _normalize_trial_result(result: dict[str, Any], output_path: Path) -> dict[str, Any]:
    best_path = result.get("best_path") or result.get("best_checkpoint")
    if not best_path:
        candidate = best_checkpoint_path(output_path)
        best_path = str(candidate) if candidate.exists() else None
    return {
        "output_path": str(output_path),
        "history_path": result.get("history_path") or result.get("history_file"),
        "best_epoch": result.get("best_epoch"),
        "best_selection_score": result.get("best_selection_score"),
        "best_artifact_path": best_path,
        "early_stopped": result.get("early_stopped"),
    }


def _parse_seed_list(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected at least one seed in test_seeds.")
        return [int(v) for v in value]
    values = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected at least one seed in --test-seeds.")
    return [int(v) for v in values]


def _identity_from_config(config: dict[str, Any], kind: str) -> tuple[str, str, str]:
    resolved = get_config_loader().resolve_config_modules(config)
    train = _train_conf(resolved, kind)
    dataset = str(train.get("dataset") or "dataset")
    if kind == "teacher":
        teacher = train.get("teacher", {}) or {}
        framework = str(teacher.get("framework") or "auto")
        model = str(teacher.get("model") or "teacher")
    elif kind == "student":
        student = train.get("student", {}) or {}
        framework = str(student.get("framework") or "auto")
        model = str(student.get("backbone") or "student")
    else:
        student = train.get("student", {}) or {}
        dist = str((train.get("distillation", {}) or {}).get("strategy") or "recdistill")
        framework = str(student.get("framework") or "auto")
        model = f"{dist}_{student.get('backbone') or 'student'}"
    return framework, model, dataset


def _default_search_space(kind: str) -> dict[str, Any]:
    space: dict[str, Any] = {
        "optimization.learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "optimization.l2_reg": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    }
    if kind == "recdistill":
        space.update(
            {
                "distillation.lambda_de": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
                "distillation.num_experts": {"type": "int", "low": 10, "high": 30, "step": 10},
            }
        )
    return space


def _sample_from_spec(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"Search-space list for '{name}' cannot be empty.")
        return trial.suggest_categorical(name, spec)

    if not isinstance(spec, dict):
        raise ValueError(f"Invalid search-space spec for '{name}'. Use list or dict.")

    kind = spec.get("type")
    low = spec.get("low")
    high = spec.get("high")
    log = bool(spec.get("log", False))

    if kind is None:
        if "choices" in spec or "values" in spec:
            kind = "categorical"
        elif isinstance(low, int) and isinstance(high, int):
            kind = "int"
        else:
            kind = "float"

    if kind == "categorical":
        choices = spec.get("choices", spec.get("values"))
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical search-space for '{name}' needs non-empty 'choices' or 'values'.")
        return trial.suggest_categorical(name, choices)

    if kind == "float":
        if low is None or high is None:
            raise ValueError(f"Float search-space for '{name}' needs 'low' and 'high'.")
        step = spec.get("step")
        if step is not None and log:
            raise ValueError(f"Search-space for '{name}': 'step' and 'log' are mutually exclusive.")
        return trial.suggest_float(name, float(low), float(high), log=log, step=step)

    if kind == "int":
        if low is None or high is None:
            raise ValueError(f"Int search-space for '{name}' needs 'low' and 'high'.")
        return trial.suggest_int(name, int(low), int(high), step=int(spec.get("step", 1)), log=log)

    raise ValueError(f"Unsupported search-space type '{kind}' for '{name}'.")


def _sample_hparams(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    return {key: _sample_from_spec(trial, key, spec) for key, spec in search_space.items()}


def _optuna_state_to_str(state: Any) -> str:
    name = getattr(state, "name", None)
    return str(name) if name is not None else str(state)


def _copy_artifact_with_history(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    if source_path.parent.name == "artifacts":
        source_history = source_path.parent.parent / "logs" / f"{source_path.stem}.history.json"
    else:
        source_history = source_path.with_suffix(".history.json")
    if source_history.exists():
        history_dest = destination_path.parent.parent / "logs" / f"{destination_path.stem}.history.json"
        history_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_history, history_dest)


def _delete_if_exists(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian optimization for a RecDistill experiment.")
    parser.add_argument("--config", default=None, help="Teacher, student, or recdistill experiment config")
    parser.add_argument("--base-config", default=None, help="Alias for --config")
    parser.add_argument("--dataset", default=None, help="Optional dataset override")
    parser.add_argument("--backbone", default=None, help="Optional model/backbone override")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--study-name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URI")
    parser.add_argument("--resume-study", action="store_true", help="Resume an existing Optuna study with the same name/storage")
    parser.add_argument("--metric", choices=["precision", "recall", "ndcg", "hr"], default=None, help="Validation metric for objective")
    parser.add_argument("--output-dir", default=None, help="Output directory for trial artifacts and summaries")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Optional study timeout in seconds")
    parser.add_argument("--rerun-best-on-test", action="store_true", help="Rerun best config on test after optimization")
    parser.add_argument("--test-seeds", default=None, help="Comma-separated seeds for best-config test rerun")
    parser.add_argument("--wandb-log", action="store_true", help="Log each Optuna trial as a W&B run")
    parser.add_argument("--wandb-project", default=None, help="W&B project for Optuna-level logs")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team")
    parser.add_argument("--wandb-group", default=None, help="W&B group name")
    parser.add_argument("--keep-training-wandb", action="store_true", help="Do not disable training-level W&B from config")
    args = parser.parse_args()

    config_path_value = args.config or args.base_config
    if config_path_value is None:
        raise ValueError("Provide --config.")
    config_path = Path(config_path_value)
    base_config = load_config(config_path)
    kind = _detect_kind(base_config)
    settings_config = get_config_loader().resolve_config_modules(base_config)
    train = _train_conf(settings_config, kind)
    optim_conf = train.get("optimization", {}) or {}
    distillation_conf = train.get("distillation", {}) if isinstance(train.get("distillation", {}), dict) else {}
    bayesian_conf = (
        optim_conf.get("bayesian")
        or train.get("bayesian")
        or settings_config.get("bayesian")
        or {}
    )
    distiller_bayesian_conf = distillation_conf.get("bayesian", {}) if isinstance(distillation_conf.get("bayesian", {}), dict) else {}
    bayesian_wandb_conf = bayesian_conf.get("wandb", {}) if isinstance(bayesian_conf.get("wandb", {}), dict) else {}

    framework_label, model_label, dataset_label = _identity_from_config(settings_config, kind)
    dataset = args.dataset or dataset_label
    model = args.backbone or model_label
    n_trials = args.n_trials if args.n_trials is not None else bayesian_conf.get("n_trials")
    seed = args.seed if args.seed is not None else int(bayesian_conf.get("seed", _get_dotted(train, "runtime.seed", 42)))
    configured_study_name = args.study_name or bayesian_conf.get("study_name")
    storage = args.storage or bayesian_conf.get("storage", "sqlite:///optuna_studies.db")
    metric = args.metric or bayesian_conf.get("metric", _get_dotted(train, "evaluation.selection_metric", "ndcg"))
    output_dir_arg = args.output_dir or bayesian_conf.get("output_dir")
    timeout_sec = args.timeout_sec if args.timeout_sec is not None else bayesian_conf.get("timeout_sec")
    rerun_best_on_test = bool(args.rerun_best_on_test or bayesian_conf.get("rerun_best_on_test", False))
    test_seeds_raw = args.test_seeds or bayesian_conf.get("test_seeds", "42")
    search_space = {}
    search_space.update(bayesian_conf.get("search_space") or {})
    search_space.update(distiller_bayesian_conf.get("search_space") or {})
    if not search_space:
        search_space = _default_search_space(kind)
    wandb_log = bool(args.wandb_log or bayesian_wandb_conf.get("enabled", False))
    wandb_project = args.wandb_project or bayesian_wandb_conf.get("project")
    wandb_entity = args.wandb_entity or bayesian_wandb_conf.get("entity")
    wandb_group_arg = args.wandb_group or bayesian_wandb_conf.get("group")
    keep_training_wandb = bool(args.keep_training_wandb or bayesian_conf.get("keep_training_wandb", False))

    if n_trials is None:
        raise ValueError(f"Missing n_trials. Set {_train_root_name(kind)}.optimization.bayesian.n_trials or use --n-trials.")
    n_trials = int(n_trials)
    if metric not in {"precision", "recall", "ndcg", "hr"}:
        raise ValueError("metric must be one of: precision, recall, ndcg, hr.")
    if wandb_log and not wandb_project:
        raise ValueError("W&B logging enabled but project missing. Set --wandb-project or optimization.bayesian.wandb.project.")

    set_global_seed(seed)

    experiment_meta = base_config.get("experiment", {}) if isinstance(base_config.get("experiment", {}), dict) else {}
    experiment_id = normalize_experiment_id(experiment_meta.get("id"), config_path=config_path)
    resume_study = bool(args.resume_study or bayesian_conf.get("resume_study", False))
    output_dir = (
        Path(output_dir_arg)
        if output_dir_arg
        else _existing_run_dir_from_config_path(config_path, kind)
        or experiment_run_dir(
            kind,
            experiment_id,
            framework=framework_label,
            model=model,
            dataset=dataset,
        )
    )
    study_name = configured_study_name or f"optuna_{kind}_{dataset}_{model}_{output_dir.name}"
    artifacts_dir = output_dir / "artifacts"
    logs_dir = output_dir / "logs"
    config_dir = output_dir / "config"
    perf_dir = output_dir / "perf"
    for directory in (artifacts_dir, logs_dir, config_dir, perf_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        (config_dir / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("optuna is not installed. Install it with `pip install optuna`.") from exc

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        load_if_exists=resume_study,
    )

    trial_records: list[dict[str, Any]] = []
    ext = _artifact_ext(kind)

    def objective(trial: "optuna.Trial") -> float:
        trial_seed = seed + int(trial.number)
        sampled_params = _sample_hparams(trial, search_space=search_space)
        output_path = artifacts_dir / f"trial_{trial.number:05d}{ext}"
        trial_config = _prepare_trial_config(
            base_config=base_config,
            kind=kind,
            trial_seed=trial_seed,
            sampled_params=sampled_params,
            output_path=output_path,
            metric=metric,
            val_only=True,
            disable_training_wandb=not keep_training_wandb,
            dataset=args.dataset,
            model=args.backbone,
        )

        wandb_group = wandb_group_arg or f"optuna/{study_name}/{kind}"
        wandb_run = _run_trial_wandb(
            enabled=wandb_log,
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            run_name=f"{study_name}_trial_{trial.number:05d}",
            config={
                "study_name": study_name,
                "kind": kind,
                "dataset": dataset,
                "model": model,
                "trial_number": trial.number,
                "seed": trial_seed,
                **sampled_params,
            },
        )

        started = time.time()
        status = "completed"
        error_message = None
        objective_value: float | None = None
        summary: dict[str, Any] = {}
        try:
            result = _run_experiment(trial_config, kind=kind, config_path=None)
            summary = _normalize_trial_result(result, output_path)
            objective_value = summary.get("best_selection_score")
            if objective_value is None:
                raise RuntimeError(f"Missing best_selection_score for trial {trial.number}.")

            trial.set_user_attr("output_path", str(output_path))
            trial.set_user_attr("history_path", summary.get("history_path"))
            trial.set_user_attr("best_artifact_path", summary.get("best_artifact_path"))
            trial.set_user_attr("best_epoch", summary.get("best_epoch"))
            trial.set_user_attr("early_stopped", summary.get("early_stopped"))
            if summary.get("best_artifact_path") != str(output_path):
                _delete_if_exists(str(output_path))
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            elapsed = round(time.time() - started, 3)
            if wandb_run is not None:
                import wandb

                wandb.log({"trial_duration_sec": elapsed})
                if status == "completed" and objective_value is not None:
                    wandb.log({"objective_value": float(objective_value), f"val_{metric}": float(objective_value)})
                else:
                    wandb.log({"trial_failed": 1})
                wandb_run.summary["status"] = status
                if error_message is not None:
                    wandb_run.summary["error"] = error_message
                wandb.finish(exit_code=0 if status == "completed" else 1)

            trial_records.append(
                {
                    "trial_number": int(trial.number),
                    "status": status,
                    "duration_sec": elapsed,
                    "value": float(objective_value) if objective_value is not None else None,
                    "kind": kind,
                    "dataset": dataset,
                    "model": model,
                    "best_epoch": summary.get("best_epoch"),
                    "output_path": None if summary.get("best_artifact_path") != summary.get("output_path") else summary.get("output_path"),
                    "best_artifact_path": summary.get("best_artifact_path"),
                    **sampled_params,
                    "error": error_message,
                }
            )

        return float(objective_value)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        catch=(RuntimeError, FileNotFoundError, ValueError),
    )

    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "state": _optuna_state_to_str(trial.state),
            "value": float(trial.value) if trial.value is not None else None,
            "kind": kind,
            "dataset": dataset,
            "model": model,
        }
        row.update(trial.params)
        for key, value in trial.user_attrs.items():
            row[f"user_attr.{key}"] = value
        trial_rows.append(row)

    trial_rows_sorted = sorted(
        trial_rows,
        key=lambda row: (
            0 if row["state"] == "COMPLETE" else 1,
            -(row["value"] if row["value"] is not None else float("-inf")),
        ),
    )
    (logs_dir / "optuna_trials.json").write_text(json.dumps(trial_rows_sorted, indent=2), encoding="utf-8")
    _write_records_tsv(trial_rows_sorted, logs_dir / "optuna_trials.tsv")
    if trial_records:
        (logs_dir / "optuna_runtime_records.json").write_text(json.dumps(trial_records, indent=2), encoding="utf-8")
        _write_records_tsv(trial_records, logs_dir / "optuna_runtime_records.tsv")

    if not study.best_trials:
        raise RuntimeError("No successful trial found in this Optuna study.")

    best = study.best_trial
    best_summary = {
        "study_name": study_name,
        "kind": kind,
        "dataset": dataset,
        "model": model,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_trial_number": int(best.number),
        "storage": storage,
    }
    best_source = best.user_attrs.get("best_artifact_path") or best.user_attrs.get("output_path")
    if best_source:
        best_source_path = Path(best_source)
        if best_source_path.exists():
            best_output_path = artifacts_dir / experiment_artifact_filename(
                kind=kind,
                experiment_id=experiment_id,
                framework=framework_label,
                model=model,
                dataset=dataset,
                best=True,
            )
            _copy_artifact_with_history(best_source_path, best_output_path)
            best_summary["best_artifact_path"] = str(best_output_path)
            best_summary["source_trial_artifact_path"] = str(best_source_path)
            print(f"Promoted best Optuna artifact: {best_output_path}")
    (logs_dir / "best_trial.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print("\nBest trial summary")
    print(f"- kind: {kind}")
    print(f"- dataset: {dataset}")
    print(f"- model: {model}")
    print(f"- best value ({metric}): {best.value}")
    print(f"- best params: {best.params}")
    print(f"- trial number: {best.number}")
    print(f"- study: {study_name}")
    print(f"- storage: {storage}")

    if not rerun_best_on_test:
        return

    rerun_dir = output_dir / "best_rerun"
    for directory in (rerun_dir / "artifacts", rerun_dir / "logs", rerun_dir / "perf"):
        directory.mkdir(parents=True, exist_ok=True)
    rerun_records: list[dict[str, Any]] = []
    for test_seed in _parse_seed_list(test_seeds_raw):
        run_output_path = rerun_dir / "artifacts" / f"seed_{test_seed}{ext}"
        trial_config = _prepare_trial_config(
            base_config=base_config,
            kind=kind,
            trial_seed=test_seed,
            sampled_params=best.params,
            output_path=run_output_path,
            metric=metric,
            val_only=False,
            disable_training_wandb=False,
            dataset=args.dataset,
            model=args.backbone,
        )
        result = _run_experiment(trial_config, kind=kind, config_path=None)
        summary = _normalize_trial_result(result, run_output_path)
        rerun_records.append(
            {
                "seed": int(test_seed),
                "best_epoch": summary.get("best_epoch"),
                "best_selection_score": summary.get("best_selection_score"),
                "output_path": summary.get("output_path"),
                "best_artifact_path": summary.get("best_artifact_path"),
                "history_path": summary.get("history_path"),
            }
        )

    (rerun_dir / "logs" / "best_rerun_results.json").write_text(json.dumps(rerun_records, indent=2), encoding="utf-8")
    _write_records_tsv(rerun_records, rerun_dir / "logs" / "best_rerun_results.tsv")
    print(f"\nSaved best-config test rerun results: {rerun_dir / 'logs' / 'best_rerun_results.tsv'}")


if __name__ == "__main__":
    main()
