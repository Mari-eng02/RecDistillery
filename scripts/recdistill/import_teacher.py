"""Import a teacher artifact into the local .teacher format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.teachers import TeacherSource, load_teacher, save_teacher_state
from recdistill.teachers.registry import available_teacher_adapters
from recdistill.data.datarec_loader import load_interaction_dataset
from recdistill.paths import (
    REPO_ROOT,
    experiment_artifact_filename,
    experiment_run_dir,
    new_experiment_id,
    normalize_experiment_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a supported teacher artifact into .teacher format.")
    parser.add_argument("--input", default=None, help="Input teacher artifact")
    parser.add_argument("--output", default=None, help="Optional explicit output .teacher path")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment ID for canonical imported-teacher outputs")
    parser.add_argument("--framework", default="auto", help="Source framework or adapter family")
    parser.add_argument("--format", default="auto", help="Source artifact format")
    parser.add_argument("--model-name", default=None, help="Teacher model name used in metadata and canonical output path")
    parser.add_argument("--dataset", default=None, help="Dataset name used in the canonical output path")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Embedding dimension used in the canonical output filename; inferred for embedding teachers")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadata key=value pair; values are parsed as JSON when possible. Can be repeated",
    )
    parser.add_argument("--list-adapters", action="store_true", help="List registered teacher adapters and exit")
    return parser


def _resolve_import_identity(args: argparse.Namespace, state) -> tuple[str, str, str, int | None]:
    framework = args.framework if args.framework != "auto" else state.metadata.get("framework")
    framework = str(framework or "external")

    model_name = args.model_name or state.metadata.get("model_name")
    if not model_name:
        raise ValueError("--model-name is required when the source teacher does not provide model_name metadata.")
    model_name = str(model_name)

    dataset = args.dataset or state.metadata.get("dataset")
    if not dataset:
        raise ValueError("--dataset is required when the source teacher does not provide dataset metadata.")
    dataset = str(dataset)

    embedding_dim = args.embedding_dim
    if embedding_dim is not None and state.has_embeddings and int(embedding_dim) != int(state.embedding_dim):
        raise ValueError(
            f"--embedding-dim={embedding_dim} does not match the imported embeddings "
            f"dimension ({state.embedding_dim})."
        )
    if embedding_dim is None:
        if state.has_embeddings:
            embedding_dim = int(state.embedding_dim)
        elif state.metadata.get("embedding_dim") is not None:
            embedding_dim = int(state.metadata["embedding_dim"])
        else:
            embedding_dim = None

    return framework, model_name, dataset, int(embedding_dim) if embedding_dim is not None else None


def _resolve_output_path(args: argparse.Namespace, *, framework: str, model_name: str, dataset: str, experiment_id: str) -> Path:
    if args.output is not None:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        return output_path.resolve()

    filename = experiment_artifact_filename(
        kind="teacher",
        experiment_id=experiment_id,
        framework=framework,
        model=model_name,
        dataset=dataset,
        best=True,
    )
    return experiment_run_dir(
        "teacher",
        experiment_id,
        framework=framework,
        model=model_name,
        dataset=dataset,
    ) / "artifacts" / filename


def _run_dir_from_output_path(output_path: Path) -> Path:
    return output_path.parent.parent if output_path.parent.name == "artifacts" else output_path.parent


def _write_import_config(
    *,
    output_path: Path,
    experiment_id: str,
    framework: str,
    model_name: str,
    dataset: str,
    embedding_dim: int | None,
) -> Path:
    run_dir = _run_dir_from_output_path(output_path)
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    name = f"{_path_slug(framework)}_{_path_slug(model_name)}_{_path_slug(dataset)}_{experiment_id}"
    config_path = config_dir / f"{name}.yaml"
    config = {
        "experiment": {
            "id": str(experiment_id),
            "name": f"{_path_slug(framework)}_{_path_slug(model_name)}_{_path_slug(dataset)}",
            "kind": "teacher",
        },
        "train_teacher": {
            "dataset": dataset,
            "teacher": {
                "framework": framework,
                "model": model_name,
                "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
                "path": str(output_path),
                "format": "checkpoint",
            },
        },
    }
    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False, allow_unicode=False)
    return config_path


def _write_import_summary(
    *,
    output_path: Path,
    input_path: str,
    framework: str,
    model_name: str,
    dataset: str,
    embedding_dim: int | None,
    state,
) -> Path:
    logs_dir = _run_dir_from_output_path(output_path) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "import_summary.json"
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "framework": framework,
        "model": model_name,
        "dataset": dataset,
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "num_users": int(state.num_users),
        "num_items": int(state.num_items),
        "has_embeddings": bool(state.has_embeddings),
        "has_scorer": bool(state.scorer is not None),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _path_slug(value: object) -> str:
    return str(value).strip().replace(" ", "_").replace("-", "_").replace("+", "_").replace("/", "_").replace("\\", "_")


def parse_metadata(rows: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for row in rows:
        key, sep, value = row.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid metadata entry {row!r}. Expected key=value.")
        metadata[key] = _parse_metadata_value(value)
    return metadata


def _metadata_with_dataset_shape(metadata: dict[str, object], dataset: str | None) -> dict[str, object]:
    enriched = dict(metadata)
    if not dataset:
        return enriched
    enriched.setdefault("dataset", dataset)
    if "num_users" in enriched and "num_items" in enriched:
        return enriched
    dataset_info = load_interaction_dataset(dataset)
    enriched.setdefault("num_users", int(dataset_info.num_users))
    enriched.setdefault("num_items", int(dataset_info.num_items))
    return enriched


def _parse_metadata_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_adapters:
        for adapter in available_teacher_adapters():
            print(adapter)
        return

    if args.input is None:
        parser.error("--input is required")

    metadata = _metadata_with_dataset_shape(parse_metadata(args.metadata), args.dataset)

    source = TeacherSource(
        path=Path(args.input),
        framework=args.framework,
        format=args.format,
        model_name=args.model_name,
        metadata=metadata,
    )
    state = load_teacher(source, device="cpu")
    try:
        framework, model_name, dataset, embedding_dim = _resolve_import_identity(args, state)
        if args.framework == "external":
            framework = "external"
        experiment_id = normalize_experiment_id(args.experiment_id or new_experiment_id())
        output_path = _resolve_output_path(
            args,
            framework=framework,
            model_name=model_name,
            dataset=dataset,
            experiment_id=experiment_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = save_teacher_state(
        output_path,
        state,
        framework=framework,
        model_name=model_name,
        metadata={
            "dataset": dataset,
            "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
            "imported_from": str(args.input),
            "import_source_format": args.format,
            "import_source_framework": args.framework,
        },
    )
    config_path = _write_import_config(
        output_path=output_path,
        experiment_id=experiment_id,
        framework=framework,
        model_name=model_name,
        dataset=dataset,
        embedding_dim=embedding_dim,
    )
    summary_path = _write_import_summary(
        output_path=output_path,
        input_path=args.input,
        framework=framework,
        model_name=model_name,
        dataset=dataset,
        embedding_dim=embedding_dim,
        state=state,
    )

    print(f"Imported teacher: {output_path}")
    print(f"Generated config: {config_path}")
    print(f"Import summary: {summary_path}")
    print(f"Format: {payload['format_version']}")
    print(f"Users/items: {state.num_users}/{state.num_items}")
    print(f"Embedding dim: {state.embedding_dim if state.has_embeddings else 'none'}")
    print(f"Has scorer: {state.scorer is not None}")


if __name__ == "__main__":
    main()
