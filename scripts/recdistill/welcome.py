"""
Print the RecDistill training compatibility matrix.

Use this script before launching experiments to see which framework/model
combinations can be trained by the RecDistill PyTorch loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recdistill.supported_models import (  # noqa: E402
    REPORTED_TOTAL_TORCH_COMPATIBLE_MODELS,
    torch_compatible_by_framework,
    torch_compatible_summary_rows,
    trainable_by_framework,
)


FRAMEWORK_DISPLAY_NAMES = {
    "recbole": "RecBole",
    "elliot": "Elliot",
    "lenskit": "Lenskit",
    "total": "Totale",
}

BOLD = "\033[1m"
RESET = "\033[0m"


def _bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def _print_summary() -> None:
    print(_bold("Imported torch-compatible model definitions"))
    print("----------------------------------------------------------")
    print("Framework   Total imported   Torch-compatible   Percentage")
    for row in torch_compatible_summary_rows():
        framework_name = FRAMEWORK_DISPLAY_NAMES.get(row["framework"], row["framework"])
        print(
            f"{framework_name:<11} "
            f"{row['total_imported']:>14}   "
            f"{row['torch_compatible']:>16}   "
            f"{row['percentage']:>10}"
        )
    print()


def _print_torch_compatible_models() -> None:
    print(_bold(f"Torch-compatible imported models ({REPORTED_TOTAL_TORCH_COMPATIBLE_MODELS})"))
    print("------------------------------------------------------------------------------------------------")
    grouped = torch_compatible_by_framework()
    for framework in ("recbole", "elliot", "lenskit"):
        models = grouped.get(framework, [])
        print(f"{FRAMEWORK_DISPLAY_NAMES[framework]} ({len(models)}):")
        line: list[str] = []
        current_len = 0
        for model in models:
            token = model.name
            projected = current_len + len(token) + (2 if line else 0)
            if line and projected > 92:
                print("  " + ", ".join(line))
                line = [token]
                current_len = len(token)
            else:
                line.append(token)
                current_len = projected
        if line:
            print("  " + ", ".join(line))
        print()


def _print_adapter_ready(verbose: bool) -> None:
    print(_bold("Adapter-backed models currently wired to the RecDistill unified loop"))
    print("--------------------------------------------------------------------")
    print("These are the combinations currently supported by the existing")
    print("FrameworkBackbone adapters for teacher/student training.")
    print()

    for framework, backbones in trainable_by_framework().items():
        print(f"{FRAMEWORK_DISPLAY_NAMES.get(framework, framework)}:")
        for backbone in backbones:
            aliases = ", ".join(backbone.aliases)
            print(f"  - {backbone.model} (aliases: {aliases})")
            if verbose:
                print(f"    adapter: {backbone.adapter}")
                print(f"    implementation: {backbone.implementation}")
                if backbone.notes:
                    print(f"    notes: {backbone.notes}")
        print()


def _print_import_note() -> None:
    print(_bold("External teacher import"))
    print("-------------------------------------------------------------------")
    print("A teacher does not need to be torch-based if it is already trained.")
    print("Convert it to .teacher with one of:")
    print("  - user/item embeddings")
    print("  - dense score matrix")
    print("  - top-k item and score arrays")
    print()
    print("Then, import it with the following entry point:")
    print("  python scripts/recdistill/import_teacher.py --help")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show RecDistill trainable model compatibility.")
    parser.add_argument("--verbose", action="store_true", help="Show adapter classes and implementation paths")
    args = parser.parse_args()

    print()
    print(_bold("Welcome to RecDistillery framework!"))
    print("=====================================")
    print()
    _print_summary()
    _print_torch_compatible_models()
    _print_adapter_ready(verbose=args.verbose)
    _print_import_note()
    print()


if __name__ == "__main__":
    main()
