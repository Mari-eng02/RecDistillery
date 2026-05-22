"""Import a teacher artifact into the local .teacher format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.teachers import TeacherSource, load_teacher, save_teacher_state
from recdistill.teachers.registry import available_teacher_adapters
from recdistill.paths import REPO_ROOT, imported_teacher_artifact_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a supported teacher artifact into .teacher format.")
    parser.add_argument("--input", default=None, help="Input teacher artifact")
    parser.add_argument("--output", default=None, help="Optional explicit output .teacher path; must follow the results/teachers canonical layout")
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


def _resolve_import_identity(args: argparse.Namespace, state) -> tuple[str, str, str, int]:
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
            raise ValueError("--embedding-dim is required for non-embedding teachers.")

    return framework, model_name, dataset, int(embedding_dim)


def _resolve_output_path(args: argparse.Namespace, *, framework: str, model_name: str, dataset: str, embedding_dim: int) -> Path:
    canonical = imported_teacher_artifact_path(
        framework=framework,
        model=model_name,
        dataset=dataset,
        embedding_dim=embedding_dim,
    )
    if args.output is None:
        return canonical

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = output_path.resolve()
    canonical = canonical.resolve()
    if output_path != canonical:
        raise ValueError(
            "--output must follow the canonical results/teachers layout exactly. "
            f"Expected: {canonical}"
        )
    return output_path


def parse_metadata(rows: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for row in rows:
        key, sep, value = row.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid metadata entry {row!r}. Expected key=value.")
        metadata[key] = _parse_metadata_value(value)
    return metadata


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

    metadata = parse_metadata(args.metadata)

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
        output_path = _resolve_output_path(
            args,
            framework=framework,
            model_name=model_name,
            dataset=dataset,
            embedding_dim=embedding_dim,
        )
    except ValueError as exc:
        parser.error(str(exc))

    payload = save_teacher_state(
        output_path,
        state,
        framework=framework,
        model_name=model_name,
        metadata={
            "dataset": dataset,
            "embedding_dim": embedding_dim,
            "imported_from": str(args.input),
            "import_source_format": args.format,
            "import_source_framework": args.framework,
        },
    )

    print(f"Imported teacher: {output_path}")
    print(f"Format: {payload['format_version']}")
    print(f"Users/items: {state.num_users}/{state.num_items}")
    print(f"Embedding dim: {state.embedding_dim if state.has_embeddings else 'none'}")
    print(f"Has scorer: {state.scorer is not None}")


if __name__ == "__main__":
    main()
