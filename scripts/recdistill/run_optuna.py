"""
Optuna-based Bayesian hyperparameter optimization for student distillation training.

Example:
    python scripts/recdistill/run_optuna.py \
        --config config/presets/recdistill/search/de_template.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.recdistill.train_student_from_config import build_command
from recdistill.config_integration import normalize_recdistill_config
from recdistill.paths import DISTILLED_STUDENT_EXT, distilled_student_artifact_path

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


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed. Use JSON config or install PyYAML.")
        data = yaml.safe_load(raw_text) or {}
    if isinstance(data, dict) and "preset" in data and "config" in data:
        data = data["config"]
    return normalize_recdistill_config(data)


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


def _read_checkpoint_summary(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {}
    import torch

    payload = torch.load(output_path, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    return {
        "best_epoch": payload.get("best_epoch"),
        "best_selection_score": payload.get("best_selection_score"),
        "early_stopped": payload.get("early_stopped"),
    }


def _read_history_row_by_epoch(history_path: Path, epoch: int) -> dict[str, Any]:
    if not history_path.exists():
        return {}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        return {}
    for row in history:
        if int(row.get("epoch", -1)) == int(epoch):
            return row
    return {}


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
    base_config: dict,
    *,
    dataset: str,
    backbone: str,
    trial_seed: int,
    sampled_params: dict[str, Any],
    output_path: Path,
    metric: str,
    val_only: bool,
    disable_training_wandb: bool,
) -> dict:
    config = copy.deepcopy(base_config)
    train_conf = _normalize_train_conf(config)

    _set_dotted(train_conf, "dataset", dataset)
    _set_dotted(train_conf, "teacher.model", backbone)
    _set_dotted(train_conf, "runtime.seed", int(trial_seed))
    _set_dotted(train_conf, "runtime.output_path", str(output_path))
    _set_dotted(train_conf, "evaluation.enabled", True)
    _set_dotted(train_conf, "evaluation.selection_split", "val")
    _set_dotted(train_conf, "evaluation.selection_metric", metric)
    _set_dotted(train_conf, "evaluation.val_only", bool(val_only))

    for dotted_key, value in sampled_params.items():
        _set_dotted(train_conf, dotted_key, value)

    if disable_training_wandb:
        _set_dotted(train_conf, "runtime.wandb.enabled", False)

    return config


def _collect_trial_result(output_path: Path, metric: str) -> dict[str, Any]:
    history_path = output_path.with_suffix(".history.json")
    ckpt = _read_checkpoint_summary(output_path)
    best_epoch = ckpt.get("best_epoch")
    best_row = _read_history_row_by_epoch(history_path, int(best_epoch)) if best_epoch else {}
    result = {
        "output_path": str(output_path),
        "history_path": str(history_path) if history_path.exists() else None,
        "best_epoch": best_epoch,
        "best_selection_score": ckpt.get("best_selection_score"),
        "early_stopped": ckpt.get("early_stopped"),
        "best_val_metric": best_row.get(f"val_{metric}") if best_row else None,
        "best_val_ndcg": best_row.get("val_ndcg") if best_row else None,
        "best_val_recall": best_row.get("val_recall") if best_row else None,
    }
    return result


def _parse_seed_list(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected at least one seed in test_seeds.")
        return [int(v) for v in value]
    text = str(value)
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected at least one seed in --test-seeds.")
    return [int(v) for v in values]


def _safe_slug(value: str, max_len: int = 80) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.=")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _experiment_tuple(train_conf: dict, *, dataset: str, model: str) -> tuple[str, str, str, str, str, str]:
    teacher_conf = train_conf.get("teacher", {}) or {}
    student_conf = train_conf.get("student", {}) or {}
    distiller = str(student_conf.get("model") or "DE")
    teacher_framework = str(teacher_conf.get("framework") or "recbole")
    teacher = str(teacher_conf.get("model") or model)
    student_framework = str(student_conf.get("framework") or "recbole")
    student = str(student_conf.get("backbone") or model)
    return (
        _safe_slug(distiller.lower()),
        _safe_slug(teacher_framework.lower()),
        _safe_slug(teacher),
        _safe_slug(student_framework.lower()),
        _safe_slug(student),
        _safe_slug(dataset.lower()),
    )


def _default_bayesian_dir(train_conf: dict, *, dataset: str, model: str) -> Path:
    distiller, teacher_framework, teacher_name, student_framework, student_name, dataset_name = _experiment_tuple(
        train_conf,
        dataset=dataset,
        model=model,
    )
    return (
        Path("results")
        / "recdistill"
        / distiller
        / teacher_framework
        / teacher_name
        / student_framework
        / student_name
        / dataset_name
        / "bayesian"
    )


def _search_best_output_path(train_conf: dict, *, dataset: str, model: str) -> Path:
    runtime_conf = train_conf.get("runtime", {}) or {}
    if runtime_conf.get("output_path"):
        return Path(runtime_conf["output_path"])

    teacher_conf = train_conf.get("teacher", {}) or {}
    student_conf = train_conf.get("student", {}) or {}
    path = distilled_student_artifact_path(
        distiller=str(student_conf.get("model") or "DE"),
        teacher_framework=teacher_conf.get("framework"),
        teacher_model=str(teacher_conf.get("model") or model),
        student_framework=student_conf.get("framework"),
        student_model=str(student_conf.get("backbone") or model),
        dataset=str(dataset),
        embedding_dim=int(student_conf.get("embedding_dim", 0)),
        strategy="best",
    )
    try:
        path = path.relative_to(REPO_ROOT)
    except ValueError:
        pass
    if not path.stem.endswith("_best"):
        path = path.with_name(f"{path.stem}_best{path.suffix}")
    return path


def _copy_artifact_with_history(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.parent.parent.joinpath("perf").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    source_history = source_path.with_suffix(".history.json")
    if source_history.exists():
        shutil.copy2(source_history, destination_path.with_suffix(".history.json"))


def _default_search_space() -> dict[str, Any]:
    return {
        "optimization.learning_rate": [1e-2, 1e-3, 1e-4],
        "optimization.l2_reg": [1e-2, 1e-3, 1e-4],
        "distillation.lambda_de": [1e-1, 1e-2, 1e-3],
        "distillation.num_experts": [10, 20, 30],
    }


def _sample_from_spec(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"Search-space list for '{name}' cannot be empty.")
        return trial.suggest_categorical(name, spec)

    if not isinstance(spec, dict):
        raise ValueError(
            f"Invalid search-space spec for '{name}'. Use list or dict."
        )

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
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(low), int(high), step=step, log=log)

    raise ValueError(f"Unsupported search-space type '{kind}' for '{name}'.")


def _sample_hparams(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for key, spec in search_space.items():
        sampled[key] = _sample_from_spec(trial, key, spec)
    return sampled


def _optuna_state_to_str(state: Any) -> str:
    name = getattr(state, "name", None)
    return str(name) if name is not None else str(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian optimization for student distillation.")
    parser.add_argument("--config", default=None, help="Single config file")
    parser.add_argument("--base-config", default=None, help="Base train_student config (YAML or JSON)")
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--backbone", default=None, help="Backbone/teacher name (e.g., BPRMF, LGCN, NMF)")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--study-name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URI")
    parser.add_argument("--metric", choices=["ndcg", "recall"], default=None, help="Validation metric for objective")
    parser.add_argument("--output-dir", default=None, help="Output directory for trial artifacts and summaries")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Optional study timeout in seconds")
    parser.add_argument("--rerun-best-on-test", action="store_true", help="Rerun best config on test after optimization")
    parser.add_argument("--test-seeds", default=None, help="Comma-separated seeds for best-config test rerun")
    parser.add_argument("--wandb-log", action="store_true", help="Log each Optuna trial as a W&B run")
    parser.add_argument("--wandb-project", default=None, help="W&B project for Optuna-level logs")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team")
    parser.add_argument("--wandb-group", default=None, help="W&B group name (default: study/dataset/backbone)")
    parser.add_argument("--keep-training-wandb", action="store_true", help="Do not disable training-level W&B from config")
    args = parser.parse_args()

    config_path_value = args.config or args.base_config
    if config_path_value is None:
        raise ValueError("Provide --config (preferred) or --base-config.")
    base_config = load_config(Path(config_path_value))
    train_conf = _normalize_train_conf(base_config)
    optim_conf = train_conf.get("optimization", {}) or {}
    optuna_conf = (
        optim_conf.get("optuna")
        or train_conf.get("optuna")
        or base_config.get("optuna")
        or {}
    )
    optuna_wandb_conf = optuna_conf.get("wandb", {}) if isinstance(optuna_conf.get("wandb", {}), dict) else {}

    dataset = args.dataset or train_conf.get("dataset")
    backbone = args.backbone or train_conf.get("teacher", {}).get("model") or train_conf.get("student", {}).get("backbone")
    n_trials = args.n_trials if args.n_trials is not None else optuna_conf.get("n_trials")
    seed = args.seed if args.seed is not None else int(optuna_conf.get("seed", train_conf.get("runtime", {}).get("seed", 42)))
    study_name = args.study_name or optuna_conf.get("study_name")
    storage = args.storage or optuna_conf.get("storage", "sqlite:///optuna_studies.db")
    metric = args.metric or optuna_conf.get("metric", train_conf.get("evaluation", {}).get("selection_metric", "ndcg"))
    output_dir_arg = args.output_dir or optuna_conf.get("output_dir")
    timeout_sec = args.timeout_sec if args.timeout_sec is not None else optuna_conf.get("timeout_sec")
    rerun_best_on_test = bool(args.rerun_best_on_test or optuna_conf.get("rerun_best_on_test", False))
    test_seeds_raw = args.test_seeds or optuna_conf.get("test_seeds", "42")
    search_space = optuna_conf.get("search_space") or _default_search_space()
    wandb_log = bool(args.wandb_log or optuna_wandb_conf.get("enabled", False))
    wandb_project = args.wandb_project or optuna_wandb_conf.get("project")
    wandb_entity = args.wandb_entity or optuna_wandb_conf.get("entity")
    wandb_group_arg = args.wandb_group or optuna_wandb_conf.get("group")
    keep_training_wandb = bool(args.keep_training_wandb or optuna_conf.get("keep_training_wandb", False))

    if not dataset:
        raise ValueError("Missing dataset. Set train_student.dataset in config or use --dataset.")
    if not backbone:
        raise ValueError("Missing backbone/model. Set train_student.teacher.model or use --backbone.")
    if n_trials is None:
        raise ValueError("Missing n_trials. Set train_student.optimization.optuna.n_trials or use --n-trials.")
    n_trials = int(n_trials)
    if metric not in {"ndcg", "recall"}:
        raise ValueError("metric must be one of: ndcg, recall.")
    if not study_name:
        study_name = f"optuna_{dataset}_{backbone}_{metric}"

    if wandb_log and not wandb_project:
        raise ValueError(
            "W&B logging enabled but project missing. "
            "Set --wandb-project or train_student.optimization.optuna.wandb.project."
        )

    set_global_seed(seed)

    output_dir = Path(output_dir_arg) if output_dir_arg else _default_bayesian_dir(
        train_conf,
        dataset=str(dataset),
        model=str(backbone),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wei_dir = output_dir / "wei"
    wei_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perf").mkdir(parents=True, exist_ok=True)

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
        load_if_exists=True,
    )

    trial_records: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        trial_seed = seed + int(trial.number)
        sampled_params = _sample_hparams(trial, search_space=search_space)

        output_path = wei_dir / f"trial_{trial.number:05d}{DISTILLED_STUDENT_EXT}"
        trial_config = _prepare_trial_config(
            base_config=base_config,
            dataset=dataset,
            backbone=backbone,
            trial_seed=trial_seed,
            sampled_params=sampled_params,
            output_path=output_path,
            metric=metric,
            val_only=True,
            disable_training_wandb=not keep_training_wandb,
        )
        cmd = build_command(trial_config)

        wandb_group = wandb_group_arg or f"optuna/{study_name}/{dataset}/{backbone}"
        wandb_run = _run_trial_wandb(
            enabled=wandb_log,
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            run_name=f"{study_name}_trial_{trial.number:05d}",
            config={
                "study_name": study_name,
                "dataset": dataset,
                "backbone": backbone,
                "trial_number": trial.number,
                "seed": trial_seed,
                **sampled_params,
            },
        )

        started = time.time()
        status = "completed"
        error_message = None
        try:
            subprocess.run(cmd, check=True)
            summary = _collect_trial_result(output_path, metric=metric)
            objective_value = summary.get("best_selection_score")
            if objective_value is None:
                raise RuntimeError(f"Missing best_selection_score for trial {trial.number}.")

            trial.set_user_attr("output_path", str(output_path))
            trial.set_user_attr("history_path", summary.get("history_path"))
            trial.set_user_attr("best_epoch", summary.get("best_epoch"))
            trial.set_user_attr("best_val_ndcg", summary.get("best_val_ndcg"))
            trial.set_user_attr("best_val_recall", summary.get("best_val_recall"))
            trial.set_user_attr("early_stopped", summary.get("early_stopped"))
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            elapsed = round(time.time() - started, 3)
            if wandb_run is not None:
                import wandb

                wandb.log({"trial_duration_sec": elapsed})
                if status == "completed":
                    wandb.log({"objective_value": float(objective_value), f"val_{metric}": float(objective_value)})
                else:
                    wandb.log({"trial_failed": 1})
                wandb_run.summary["status"] = status
                if error_message is not None:
                    wandb_run.summary["error"] = error_message
                wandb.finish(exit_code=0 if status == "completed" else 1)

            record = {
                "trial_number": int(trial.number),
                "status": status,
                "duration_sec": elapsed,
                "value": float(objective_value) if status == "completed" else None,
                "dataset": dataset,
                "backbone": backbone,
                **sampled_params,
                "error": error_message,
            }
            trial_records.append(record)

        return float(objective_value)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        catch=(subprocess.CalledProcessError, RuntimeError, FileNotFoundError),
    )

    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "state": _optuna_state_to_str(trial.state),
            "value": float(trial.value) if trial.value is not None else None,
            "dataset": dataset,
            "backbone": backbone,
        }
        row.update(trial.params)
        for key, value in trial.user_attrs.items():
            row[f"user_attr.{key}"] = value
        trial_rows.append(row)

    trial_rows_sorted = sorted(
        trial_rows,
        key=lambda row: (0 if row["state"] == "COMPLETE" else 1, -(row["value"] if row["value"] is not None else float("-inf"))),
    )
    (output_dir / "optuna_trials.json").write_text(json.dumps(trial_rows_sorted, indent=2), encoding="utf-8")
    _write_records_tsv(trial_rows_sorted, output_dir / "optuna_trials.tsv")

    if trial_records:
        (output_dir / "optuna_runtime_records.json").write_text(json.dumps(trial_records, indent=2), encoding="utf-8")
        _write_records_tsv(trial_records, output_dir / "optuna_runtime_records.tsv")

    if not study.best_trials:
        raise RuntimeError("No successful trial found in this Optuna study.")

    best = study.best_trial
    best_summary = {
        "study_name": study_name,
        "dataset": dataset,
        "backbone": backbone,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_trial_number": int(best.number),
        "storage": storage,
    }
    best_trial_output = best.user_attrs.get("output_path")
    best_trial_output_path = Path(best_trial_output) if best_trial_output else None
    if best_trial_output_path is not None and best_trial_output_path.exists():
        best_output_path = _search_best_output_path(base_config, dataset=str(dataset), model=str(backbone))
        _copy_artifact_with_history(best_trial_output_path, best_output_path)
        best_summary["best_artifact_path"] = str(best_output_path)
        best_summary["source_trial_artifact_path"] = str(best_trial_output_path)
        print(f"Promoted best Optuna artifact: {best_output_path}")
    (output_dir / "best_trial.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print("\nBest trial summary")
    print(f"- dataset: {dataset}")
    print(f"- backbone: {backbone}")
    print(f"- best value ({metric}): {best.value}")
    print(f"- best params: {best.params}")
    print(f"- trial number: {best.number}")
    print(f"- study: {study_name}")
    print(f"- storage: {storage}")

    if not rerun_best_on_test:
        return

    test_seeds = _parse_seed_list(str(test_seeds_raw))
    rerun_records: list[dict[str, Any]] = []
    for test_seed in test_seeds:
        rerun_dir = output_dir / "best_rerun"
        (rerun_dir / "wei").mkdir(parents=True, exist_ok=True)
        (rerun_dir / "perf").mkdir(parents=True, exist_ok=True)
        run_output_path = rerun_dir / "wei" / f"seed_{test_seed}{DISTILLED_STUDENT_EXT}"
        trial_config = _prepare_trial_config(
            base_config=base_config,
            dataset=dataset,
            backbone=backbone,
            trial_seed=test_seed,
            sampled_params=best.params,
            output_path=run_output_path,
            metric=metric,
            val_only=False,
            disable_training_wandb=False,
        )
        cmd = build_command(trial_config)
        subprocess.run(cmd, check=True)

        ckpt = _read_checkpoint_summary(run_output_path)
        best_epoch = int(ckpt.get("best_epoch") or 0)
        best_row = _read_history_row_by_epoch(run_output_path.with_suffix(".history.json"), best_epoch) if best_epoch > 0 else {}
        rerun_records.append(
            {
                "seed": int(test_seed),
                "best_epoch": best_epoch if best_epoch > 0 else None,
                "best_selection_score": ckpt.get("best_selection_score"),
                "val_ndcg": best_row.get("val_ndcg"),
                "val_recall": best_row.get("val_recall"),
                "test_ndcg": best_row.get("test_ndcg"),
                "test_recall": best_row.get("test_recall"),
                "test_precision": best_row.get("test_precision"),
                "test_hr": best_row.get("test_hr"),
                "output_path": str(run_output_path),
            }
        )

    rerun_dir = output_dir / "best_rerun"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    (rerun_dir / "best_rerun_results.json").write_text(json.dumps(rerun_records, indent=2), encoding="utf-8")
    _write_records_tsv(rerun_records, rerun_dir / "best_rerun_results.tsv")
    print(f"\nSaved best-config test rerun results: {rerun_dir / 'best_rerun_results.tsv'}")


if __name__ == "__main__":
    main()
