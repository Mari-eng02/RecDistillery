"""
Convert teacher models from CUDA to CPU format.

This script loads CUDA-saved teacher models and re-saves them as CPU-compatible.
Run once to prepare all teacher files for cross-platform use.

Usage:
    python scripts/recdistill/convert_teachers_to_cpu.py [--pattern "results/**/*.teacher"]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.serialization as torch_ser


def convert_teacher_file(teacher_path: Path, backup: bool = True) -> bool:
    """
    Load a CUDA-saved teacher and re-save it as CPU-compatible.
    
    Args:
        teacher_path: Path to .teacher file
        backup: Whether to create a .backup before converting
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    if not teacher_path.exists():
        print(f"  ✗ File not found: {teacher_path}")
        return False
    
    # Create backup
    if backup:
        backup_path = teacher_path.with_suffix(".teacher.backup")
        if not backup_path.exists():
            backup_path.write_bytes(teacher_path.read_bytes())
    
    try:
        # Load with monkey patch to allow CUDA references
        original_validate = torch_ser._validate_device
        
        def allow_cuda_on_cpu(location, backend_name):
            try:
                return original_validate(location, backend_name)
            except RuntimeError as e:
                if 'cuda' in str(location).lower() and 'cuda.is_available' in str(e):
                    return torch.device('cpu')
                raise
        
        torch_ser._validate_device = allow_cuda_on_cpu
        
        try:
            with teacher_path.open("rb") as fp:
                payload = pickle.load(fp, encoding='latin1')
        finally:
            torch_ser._validate_device = original_validate
        
        # Ensure all tensors are on CPU
        def move_to_cpu(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: move_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_cpu(v) for v in obj)
            return obj
        
        payload = move_to_cpu(payload)
        
        # Re-save on CPU
        with teacher_path.open("wb") as fp:
            torch.save(payload, fp)
        
        print(f"  ✓ Converted: {teacher_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting {teacher_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CUDA teacher files to CPU format")
    parser.add_argument(
        "--pattern",
        default="results/**/*.teacher",
        help="Glob pattern to find teacher files (default: results/**/*.teacher)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create .backup files"
    )
    args = parser.parse_args()
    
    # Find all teacher files
    teacher_files = list(Path(".").glob(args.pattern))
    
    if not teacher_files:
        print(f"No teacher files found matching pattern: {args.pattern}")
        return
    
    print(f"\n{'='*70}")
    print(f"Converting {len(teacher_files)} teacher file(s) to CPU format")
    print(f"{'='*70}\n")
    
    converted = 0
    failed = 0
    
    for teacher_path in sorted(teacher_files):
        if convert_teacher_file(teacher_path, backup=not args.no_backup):
            converted += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Conversion complete: {converted} converted, {failed} failed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
