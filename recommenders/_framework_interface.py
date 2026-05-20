from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable


@dataclass(frozen=True)
class ModelEntry:
    framework: str
    name: str
    module: str
    class_name: str
    category: str
    path: str


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    return ""


def _module_name(package: str, root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join([package, *rel.parts])


def _category(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else rel.with_suffix("").name


class FrameworkModelInterface:
    def __init__(
        self,
        *,
        framework: str,
        package: str,
        root: Path,
        base_names: Iterable[str] = (),
        include_suffixes: Iterable[str] = (),
        include_names: Iterable[str] = (),
        exclude_names: Iterable[str] = (),
        exclude_suffixes: Iterable[str] = (),
    ):
        self.framework = framework
        self.package = package
        self.root = root
        self.base_names = set(base_names)
        self.include_suffixes = tuple(include_suffixes)
        self.include_names = set(include_names)
        self.exclude_names = set(exclude_names)
        self.exclude_suffixes = tuple(exclude_suffixes)
        self._models: dict[str, ModelEntry] | None = None

    def list_models(self, category: str | None = None) -> list[ModelEntry]:
        models = sorted(self._registry().values(), key=lambda entry: (entry.category, entry.name.lower()))
        if category is not None:
            models = [entry for entry in models if entry.category == category]
        return models

    def model_names(self, category: str | None = None) -> list[str]:
        return [entry.name for entry in self.list_models(category)]

    def categories(self) -> list[str]:
        return sorted({entry.category for entry in self._registry().values()})

    def models_by_category(self) -> dict[str, list[ModelEntry]]:
        grouped: dict[str, list[ModelEntry]] = {}
        for entry in self.list_models():
            grouped.setdefault(entry.category, []).append(entry)
        return grouped

    def get_model_entry(self, name: str) -> ModelEntry:
        key = name.lower()
        try:
            return self._registry()[key]
        except KeyError as exc:
            available = ", ".join(self.model_names())
            raise KeyError(f"Unknown {self.framework} model '{name}'. Available models: {available}") from exc

    def load_model(self, name: str):
        entry = self.get_model_entry(name)
        module = importlib.import_module(entry.module)
        return getattr(module, entry.class_name)

    def load_module(self, name: str) -> ModuleType:
        return importlib.import_module(self.get_model_entry(name).module)

    def is_model_importable(self, name: str) -> bool:
        try:
            self.load_model(name)
        except Exception:
            return False
        return True

    def _registry(self) -> dict[str, ModelEntry]:
        if self._models is None:
            self._models = self._scan()
        return self._models

    def _scan(self) -> dict[str, ModelEntry]:
        models: dict[str, ModelEntry] = {}
        for path in self.root.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue
            module = _module_name(self.package, self.root, path)
            category = _category(self.root, path)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and self._include_class(node):
                    key = node.name.lower()
                    models.setdefault(
                        key,
                        ModelEntry(
                            framework=self.framework,
                            name=node.name,
                            module=module,
                            class_name=node.name,
                            category=category,
                            path=str(path),
                        ),
                    )
        return models

    def _include_class(self, node: ast.ClassDef) -> bool:
        name = node.name
        if name in self.exclude_names or name.endswith(self.exclude_suffixes):
            return False
        if name in self.include_names or name.endswith(self.include_suffixes):
            return True
        return bool(self.base_names.intersection({_base_name(base) for base in node.bases}))
