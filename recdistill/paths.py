from __future__ import annotations

from pathlib import Path
from typing import Any

from recdistill.registry import distiller_slug, model_slug


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = str(REPO_ROOT)
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "config"
PRESETS_ROOT = CONFIG_ROOT / "presets"
TEACHER_EXT = ".teacher"
STUDENT_EXT = ".student"
DISTILLED_STUDENT_EXT = ".distilled_student"

CITEULIKE = "citeulike"
BOOKCROSSING = "bookcrossing"
AMAZONDM = "amazon_dm"
AMAZONCD = "amazon_cd"

LGCN = "LGCN"
BPRMF = "BPRMF"
NMF = "NMF"

_DATASET_FILENAME_BY_TYPE = {
    "raw": Path("data") / "dataset.tsv",
    "processed": Path("dataset.tsv"),
    "train": Path("train.tsv"),
    "val": Path("val.tsv"),
    "test": Path("test.tsv"),
}


class PathManager:
    DISTILLERS = ["de", "rrd", "unkd", "htd", "ftd", "de_rrd", "de_unkd", "rrd_unkd", "de_rrd_unkd"]
    BACKBONES = ["bprmf", "lgcn", "nmf"]
    DATASETS = ["amazon_cd", "bookcrossing", "citeulike"]


def _create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_directory(dataset_name: str, create_if_not_exists: bool = True) -> str:
    dataset_dir = DATA_ROOT / dataset_name
    if not dataset_dir.exists():
        if not create_if_not_exists:
            raise FileNotFoundError(
                f"Directory at {dataset_dir} not found. Please, check that dataset directory exists"
            )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory at '{dataset_dir}'")
    return str(dataset_dir.resolve())


def dataset_filepath(dataset_name: str, type: str = "raw", exists: bool = True) -> str:
    if type not in _DATASET_FILENAME_BY_TYPE:
        raise AssertionError(f"Incorrect dataset type. Dataset type found {type}.")
    path = Path(dataset_directory(dataset_name)) / _DATASET_FILENAME_BY_TYPE[type]
    if exists and not path.exists():
        raise FileNotFoundError(f"File at {path} not found. Please, check your files")
    return str(path.resolve())


def _relative_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_teacher_checkpoint(
    *,
    dataset: str,
    teacher_model: str | None,
    teacher_embedding_dim: int | None,
    teacher_framework: str | None = None,
    teacher_path: str | Path | None = None,
) -> Path:
    if teacher_path is not None:
        return Path(teacher_path)

    if teacher_model is None or teacher_embedding_dim is None:
        raise ValueError(
            "When teacher_path is not set, both teacher_model and teacher_embedding_dim are required."
        )
    model = model_slug(teacher_model).upper()
    dataset_slug = str(dataset).lower()
    base = RESULTS_ROOT / dataset_slug / "teacher"
    framework_slug = str(teacher_framework).strip().lower() if teacher_framework else ""
    if framework_slug:
        base = base / framework_slug
        file_name = f"{framework_slug}_{model}_{dataset_slug}_{int(teacher_embedding_dim)}{TEACHER_EXT}"
    else:
        file_name = f"{model}_{dataset_slug}_{int(teacher_embedding_dim)}{TEACHER_EXT}"
    return base / model / "best" / "wei" / file_name


def resolve_student_checkpoint(
    *,
    dataset: str,
    distiller: str,
    teacher_model: str | None,
    student_backbone: str | None,
    student_framework: str | None = None,
    student_embedding_dim: int,
    output_path: str | Path | None = None,
) -> Path:
    if output_path is not None:
        return Path(output_path)

    raw_distiller = str(distiller).strip().lower()
    is_plain = raw_distiller in {"none", "plain", "no", "false", "0"}
    teacher_slug = model_slug(teacher_model) if teacher_model is not None else "teacher"
    student_slug = model_slug(student_backbone) if student_backbone is not None else "student"
    distiller_name = "plain" if is_plain else distiller_slug(distiller)
    dataset_slug = str(dataset).lower()
    framework_slug = str(student_framework).strip().lower() if student_framework else ""
    extension = STUDENT_EXT if is_plain else DISTILLED_STUDENT_EXT
    file_name = (
        f"{framework_slug}_{student_slug}_{dataset_slug}_{int(student_embedding_dim)}{extension}"
        if framework_slug
        else f"{student_slug}_{dataset_slug}_{int(student_embedding_dim)}{extension}"
    )
    return (
        RESULTS_ROOT
        / "recdistill"
        / distiller_name
        / teacher_slug
        / student_slug
        / dataset_slug
        / "fixed"
        / "checkpoints"
        / file_name
    )


def history_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".history.json")


def best_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.suffix in {STUDENT_EXT, DISTILLED_STUDENT_EXT, TEACHER_EXT}:
        return path.with_name(f"{path.stem}.best{path.suffix}")
    return path.with_suffix(".best.pt")


def early_stop_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.suffix in {STUDENT_EXT, DISTILLED_STUDENT_EXT, TEACHER_EXT}:
        return path.with_name(f"{path.stem}.earlystop_best{path.suffix}")
    return path.with_suffix(".earlystop_best.pt")


def recommendation_path(
    *,
    dataset: str,
    distiller: str,
    teacher_model: str | None,
    student_embedding_dim: int,
    student_backbone: str | None = None,
    filename: str | None = None,
) -> Path:
    teacher_slug = model_slug(teacher_model) if teacher_model is not None else "teacher"
    student_slug = model_slug(student_backbone) if student_backbone is not None else "student"
    base = (
        RESULTS_ROOT
        / "recdistill"
        / distiller_slug(distiller)
        / teacher_slug
        / student_slug
        / str(dataset).lower()
        / "recs"
    )
    return base / (filename or f"student_{int(student_embedding_dim)}.tsv")


def performance_path(
    *,
    dataset: str,
    distiller: str,
    teacher_model: str | None,
    student_backbone: str | None = None,
    phase: str = "fixed",
    filename: str = "metrics.json",
) -> Path:
    teacher_slug = model_slug(teacher_model) if teacher_model is not None else "teacher"
    student_slug = model_slug(student_backbone) if student_backbone is not None else "student"
    return (
        RESULTS_ROOT
        / "recdistill"
        / distiller_slug(distiller)
        / teacher_slug
        / student_slug
        / str(dataset).lower()
        / phase
        / "perf"
        / filename
    )


def teacher_weights_path(
    model: str,
    dataset: str,
    embedding_dim: int,
    phase: str = "best",
    framework: str | None = None,
) -> str:
    """Compatibility helper for legacy scripts, backed by the framework path module."""
    model_name = model_slug(model).upper()
    dataset_slug = str(dataset).lower()
    base = RESULTS_ROOT / dataset_slug / "teacher"
    framework_slug = str(framework).strip().lower() if framework else ""
    if framework_slug:
        base = base / framework_slug
        file_name = f"{framework_slug}_{model_name}_{dataset_slug}_{int(embedding_dim)}{TEACHER_EXT}"
    else:
        file_name = f"{model_name}_{dataset_slug}_{int(embedding_dim)}{TEACHER_EXT}"
    path = base / model_name / phase / "wei" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def student_weights_path(
    distiller: str,
    teacher: str,
    dataset: str,
    embedding_dim: int,
    phase: str = "fixed",
    student: str | None = None,
) -> str:
    """Compatibility helper for exported distilled student payloads."""
    distiller_name = distiller_slug(distiller)
    teacher_name = model_slug(teacher).upper()
    student_name = model_slug(student or teacher).upper()
    dataset_slug = str(dataset).lower()
    path = (
        RESULTS_ROOT
        / "recdistill"
        / distiller_name
        / teacher_name.lower()
        / student_name.lower()
        / dataset_slug
        / phase
        / "wei"
        / f"{distiller_name}_{teacher_name}_to_{student_name}_{dataset_slug}_{int(embedding_dim)}{STUDENT_EXT}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_student_checkpoint_from_args(args: Any, distiller_name: str) -> Path:
    return resolve_student_checkpoint(
        dataset=args.dataset,
        distiller=distiller_name,
        teacher_model=args.teacher_model,
        student_backbone=args.student_backbone,
        student_framework=getattr(args, "student_framework", None),
        student_embedding_dim=args.student_embedding_dim,
        output_path=args.output_path,
    )


def resolve_teacher_checkpoint_from_args(args: Any) -> Path:
    return resolve_teacher_checkpoint(
        dataset=args.dataset,
        teacher_model=args.teacher_model,
        teacher_embedding_dim=args.teacher_embedding_dim,
        teacher_framework=getattr(args, "teacher_framework", None),
        teacher_path=args.teacher_path,
    )
