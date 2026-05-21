from __future__ import annotations

from pathlib import Path
from typing import Any

from recdistill.registry import canonical_model_name, distiller_slug, model_slug


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


def _framework_slug(framework: str | None, default: str = "recbole") -> str:
    raw = str(framework or default).strip().lower()
    return default if raw in {"", "auto"} else raw


def _dataset_slug(dataset: str) -> str:
    return str(dataset).strip().lower()


def _model_label(model: str) -> str:
    raw = str(model).strip()
    try:
        return canonical_model_name(raw)
    except ValueError:
        cleaned = raw.replace(" ", "_").replace("/", "_").replace("\\", "_")
        if not cleaned:
            raise ValueError("Model name cannot be empty.")
        return cleaned


def teacher_artifact_path(
    *,
    framework: str | None,
    model: str,
    dataset: str,
    embedding_dim: int,
) -> Path:
    framework_slug = _framework_slug(framework)
    model_name = _model_label(model)
    dataset_slug = _dataset_slug(dataset)
    file_name = f"{framework_slug}_{model_name}_{dataset_slug}_{int(embedding_dim)}{TEACHER_EXT}"
    return RESULTS_ROOT / "teachers" / framework_slug / model_name / dataset_slug / "best" / "wei" / file_name


def imported_teacher_artifact_path(
    *,
    framework: str | None,
    model: str,
    dataset: str,
    embedding_dim: int,
) -> Path:
    framework_slug = _framework_slug(framework)
    model_name = _model_label(model)
    dataset_slug = _dataset_slug(dataset)
    file_name = f"{framework_slug}_{model_name}_{dataset_slug}_{int(embedding_dim)}{TEACHER_EXT}"
    return RESULTS_ROOT / "teachers" / framework_slug / model_name / dataset_slug / "wei" / file_name


def student_artifact_path(
    *,
    framework: str | None,
    model: str,
    dataset: str,
    embedding_dim: int,
) -> Path:
    framework_slug = _framework_slug(framework)
    model_name = _model_label(model)
    dataset_slug = _dataset_slug(dataset)
    file_name = f"{framework_slug}_{model_name}_{dataset_slug}_{int(embedding_dim)}{STUDENT_EXT}"
    return RESULTS_ROOT / "students" / framework_slug / model_name / dataset_slug / "best" / "wei" / file_name


def distilled_student_artifact_path(
    *,
    distiller: str,
    teacher_framework: str | None,
    teacher_model: str,
    student_framework: str | None,
    student_model: str,
    dataset: str,
    embedding_dim: int,
    strategy: str = "best",
) -> Path:
    distiller_name = distiller_slug(distiller)
    teacher_framework_slug = _framework_slug(teacher_framework)
    student_framework_slug = _framework_slug(student_framework)
    teacher_model_name = _model_label(teacher_model)
    student_model_name = _model_label(student_model)
    dataset_slug = _dataset_slug(dataset)
    file_name = (
        f"{teacher_framework_slug}_{teacher_model_name}_"
        f"{student_framework_slug}_{student_model_name}_"
        f"{dataset_slug}_{int(embedding_dim)}{DISTILLED_STUDENT_EXT}"
    )
    return (
        RESULTS_ROOT
        / "recdistill"
        / distiller_name
        / teacher_framework_slug
        / teacher_model_name
        / student_framework_slug
        / student_model_name
        / dataset_slug
        / str(strategy).strip().lower()
        / "wei"
        / file_name
    )


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
    trained_path = teacher_artifact_path(
        framework=teacher_framework,
        model=teacher_model,
        dataset=dataset,
        embedding_dim=teacher_embedding_dim,
    )
    imported_path = imported_teacher_artifact_path(
        framework=teacher_framework,
        model=teacher_model,
        dataset=dataset,
        embedding_dim=teacher_embedding_dim,
    )
    return imported_path if imported_path.exists() else trained_path


def resolve_student_checkpoint(
    *,
    dataset: str,
    distiller: str,
    teacher_model: str | None,
    student_backbone: str | None,
    student_embedding_dim: int,
    teacher_framework: str | None = None,
    student_framework: str | None = None,
    output_path: str | Path | None = None,
    strategy: str = "best",
) -> Path:
    if output_path is not None:
        return Path(output_path)

    raw_distiller = str(distiller).strip().lower()
    is_plain = raw_distiller in {"none", "plain", "no", "false", "0"}
    if is_plain:
        if student_backbone is None:
            raise ValueError("student_backbone is required for plain student path resolution.")
        return student_artifact_path(
            framework=student_framework,
            model=student_backbone,
            dataset=dataset,
            embedding_dim=student_embedding_dim,
        )

    if teacher_model is None or student_backbone is None:
        raise ValueError("teacher_model and student_backbone are required for distilled student path resolution.")
    return distilled_student_artifact_path(
        distiller=distiller,
        teacher_framework=teacher_framework,
        teacher_model=teacher_model,
        student_framework=student_framework,
        student_model=student_backbone,
        dataset=dataset,
        embedding_dim=student_embedding_dim,
        strategy=strategy,
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
    phase: str = "best",
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
    path = teacher_artifact_path(
        framework=framework,
        model=model,
        dataset=dataset,
        embedding_dim=embedding_dim,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def student_weights_path(
    distiller: str,
    teacher: str,
    dataset: str,
    embedding_dim: int,
    phase: str = "best",
    student: str | None = None,
) -> str:
    """Compatibility helper for exported distilled student payloads."""
    path = distilled_student_artifact_path(
        distiller=distiller,
        teacher_framework=None,
        teacher_model=teacher,
        student_framework=None,
        student_model=student or teacher,
        dataset=dataset,
        embedding_dim=embedding_dim,
        strategy=phase,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_student_checkpoint_from_args(args: Any, distiller_name: str) -> Path:
    return resolve_student_checkpoint(
        dataset=args.dataset,
        distiller=distiller_name,
        teacher_framework=getattr(args, "teacher_framework", None),
        teacher_model=args.teacher_model,
        student_backbone=args.student_backbone,
        student_framework=getattr(args, "student_framework", None),
        student_embedding_dim=args.student_embedding_dim,
        output_path=args.output_path,
        strategy=getattr(args, "output_strategy", "best"),
    )


def resolve_teacher_checkpoint_from_args(args: Any) -> Path:
    return resolve_teacher_checkpoint(
        dataset=args.dataset,
        teacher_model=args.teacher_model,
        teacher_embedding_dim=args.teacher_embedding_dim,
        teacher_framework=getattr(args, "teacher_framework", None),
        teacher_path=args.teacher_path,
    )
