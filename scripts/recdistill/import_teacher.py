"""
Import an external teacher into RecDistill's native .teacher format.

Examples:
    python scripts/recdistill/import_teacher.py --input elliot_teacher.teacher --framework elliot --output imported.teacher
    python scripts/recdistill/import_teacher.py --input teacher.teacher --format auto --output imported.teacher
    python scripts/recdistill/import_teacher.py --input embeddings.npz --format embeddings_npz --output imported.teacher
    python scripts/recdistill/import_teacher.py --user-embeddings user.npy --item-embeddings item.npy --output imported.teacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.teachers import TeacherSource, load_teacher, save_teacher_state
from recdistill.teachers.registry import available_teacher_adapters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import any supported teacher into RecDistill .teacher format.")
    parser.add_argument("--input", default=None, help="Input teacher artifact (.teacher, .pt, .npz, etc.)")
    parser.add_argument("--output", default=None, help="Output RecDistill native .teacher file")
    parser.add_argument("--framework", default="auto", help="Source framework or adapter family")
    parser.add_argument("--format", default="auto", help="Source artifact format")
    parser.add_argument("--model-name", default=None, help="Optional teacher model name metadata")
    parser.add_argument("--adapter", default=None, help="Custom TeacherAdapter import path")
    parser.add_argument("--user-embeddings", default=None, help="User embedding .npy file for generic numpy import")
    parser.add_argument("--item-embeddings", default=None, help="Item embedding .npy file for generic numpy import")
    parser.add_argument("--score-matrix", default=None, help="Dense user-item score matrix .npy file")
    parser.add_argument("--topk-items", default=None, help="Precomputed top-k item ids .npy file shaped [users, k]")
    parser.add_argument("--topk-scores", default=None, help="Optional scores for --topk-items, shaped [users, k]")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadata key=value pair; values are parsed as JSON when possible. Can be repeated",
    )
    parser.add_argument("--public-to-local-user-id", default=None, help="JSON/NPY mapping from raw/public user ids to local embedding rows")
    parser.add_argument("--public-to-local-item-id", default=None, help="JSON/NPY mapping from raw/public item ids to local embedding rows")
    parser.add_argument("--local-to-public-user-id", default=None, help="JSON/NPY mapping from local user rows to raw/public user ids")
    parser.add_argument("--local-to-public-item-id", default=None, help="JSON/NPY mapping from local item rows to raw/public item ids")
    parser.add_argument("--list-adapters", action="store_true", help="List registered teacher adapters and exit")
    return parser


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


def load_mapping(path: str | None) -> object | None:
    if path is None:
        return None
    mapping_path = Path(path)
    suffix = mapping_path.suffix.lower()
    if suffix == ".json":
        return json.loads(mapping_path.read_text(encoding="utf-8"))
    if suffix == ".npy":
        value = np.load(mapping_path, allow_pickle=True)
        if value.shape == ():
            return value.item()
        return value.tolist()
    if suffix == ".npz":
        payload = np.load(mapping_path, allow_pickle=True)
        if len(payload.files) != 1:
            raise ValueError(f"Mapping npz must contain exactly one array: {mapping_path}")
        value = payload[payload.files[0]]
        if value.shape == ():
            return value.item()
        return value.tolist()
    raise ValueError(f"Unsupported mapping file extension for {mapping_path}. Use .json, .npy, or .npz.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_adapters:
        for adapter in available_teacher_adapters():
            print(adapter)
        return

    if args.output is None:
        parser.error("--output is required unless --list-adapters is used")

    if args.input is None and not (args.user_embeddings and args.item_embeddings) and not args.score_matrix and not args.topk_items:
        parser.error("--input, both --user-embeddings/--item-embeddings, --score-matrix, or --topk-items are required")

    metadata = parse_metadata(args.metadata)
    for key, value in {
        "public_to_local_user_id": load_mapping(args.public_to_local_user_id),
        "public_to_local_item_id": load_mapping(args.public_to_local_item_id),
        "local_to_public_user_id": load_mapping(args.local_to_public_user_id),
        "local_to_public_item_id": load_mapping(args.local_to_public_item_id),
    }.items():
        if value is not None:
            metadata[key] = value

    source = TeacherSource(
        path=Path(args.input) if args.input else None,
        framework=args.framework,
        format=args.format,
        model_name=args.model_name,
        adapter=args.adapter,
        user_embeddings_path=Path(args.user_embeddings) if args.user_embeddings else None,
        item_embeddings_path=Path(args.item_embeddings) if args.item_embeddings else None,
        score_matrix_path=Path(args.score_matrix) if args.score_matrix else None,
        topk_items_path=Path(args.topk_items) if args.topk_items else None,
        topk_scores_path=Path(args.topk_scores) if args.topk_scores else None,
        metadata=metadata,
    )
    state = load_teacher(source, device="cpu")
    payload = save_teacher_state(
        args.output,
        state,
        framework=args.framework if args.framework != "auto" else state.metadata.get("framework"),
        model_name=args.model_name or state.metadata.get("model_name"),
        metadata={
            "imported_from": str(args.input) if args.input else None,
            "import_source_format": args.format,
            "import_source_framework": args.framework,
        },
    )

    print(f"Imported teacher: {args.output}")
    print(f"Format: {payload['format_version']}")
    print(f"Users/items: {state.num_users}/{state.num_items}")
    print(f"Embedding dim: {state.embedding_dim if state.has_embeddings else 'none'}")
    print(f"Has scorer: {state.scorer is not None}")


if __name__ == "__main__":
    main()
