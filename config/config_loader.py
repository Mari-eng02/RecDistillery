"""
Configuration loader with validation.
Centralizes loading and validation of all configuration files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import yaml
from pydantic import ValidationError

from recdistill.paths import distilled_student_artifact_path

from config.schemas import (
    ConfigPreset,
    RecDistillConfig,
    DataConfig,
    ModelConfig,
)


class ConfigLoader:
    """Unified configuration loader for RecDistill."""

    def __init__(self, config_root: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_root: Root directory for config files. Defaults to ./config
        """
        if config_root is None:
            config_root = Path(__file__).parent
        
        self.root = Path(config_root)
        self._cache: Dict[str, Any] = {}

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML file with caching."""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.root / filepath
        
        if str(filepath) in self._cache:
            return self._cache[str(filepath)]
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = yaml.safe_load(f)
        
        self._cache[str(filepath)] = data
        return data

    def _resolve_config_path(self, path: Union[str, Path]) -> Path:
        config_path = Path(path)
        return config_path if config_path.is_absolute() else self.root / config_path

    def load_module_config(self, module_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load a module config and apply optional overrides."""
        data = copy.deepcopy(self._load_yaml(self._resolve_config_path(module_path)) or {})
        if overrides:
            data = _deep_merge(data, overrides)
        return data

    def resolve_config_modules(self, config: Any) -> Any:
        """Resolve `default: path/to/module.yaml` sections recursively.

        Paths may be absolute or relative to the config root. Sibling keys in
        the same mapping override the loaded defaults.
        """
        if isinstance(config, list):
            return [self.resolve_config_modules(item) for item in config]
        if not isinstance(config, dict):
            return config

        if "default" in config:
            default_path = config.get("default")
            if default_path is None:
                base: Dict[str, Any] = {}
            elif isinstance(default_path, (str, Path)):
                base = self.resolve_config_modules(self.load_module_config(default_path))
            else:
                raise TypeError("Config module `default` must be a path string.")
            overrides = {
                key: self.resolve_config_modules(value)
                for key, value in config.items()
                if key != "default"
            }
            return _deep_merge(base, overrides)

        return {key: self.resolve_config_modules(value) for key, value in config.items()}

    def load_recdistill_config(self, config_path: Union[str, Path]) -> RecDistillConfig:
        """
        Load and validate RecDistill configuration.
        
        Args:
            config_path: Path to RecDistill config YAML file
            
        Returns:
            Validated RecDistillConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        data = self._load_yaml(config_path)
        if isinstance(data, dict) and "preset" in data and "config" in data:
            data = data["config"]
        data = self.resolve_config_modules(data)
        from recdistill.config_integration import normalize_recdistill_config

        data = normalize_recdistill_config(data)
        try:
            return RecDistillConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid RecDistill config in {config_path}:\n{e}")

    def load_preset(self, preset_path: Union[str, Path]) -> ConfigPreset:
        """
        Load a wrapped experiment preset.

        Presets preserve provenance metadata under `preset` and keep the actual
        training payload under `config`.
        """
        data = self._load_yaml(preset_path)
        try:
            return ConfigPreset(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid config preset in {preset_path}:\n{e}")

    def load_recdistill_preset(self, preset_path: Union[str, Path]) -> RecDistillConfig:
        """Load a RecDistill preset and validate its inner config."""
        from recdistill.config_integration import normalize_recdistill_config

        preset = self.load_preset(preset_path)
        if preset.preset.kind.lower() != "recdistill":
            raise ValueError(f"Preset is not a RecDistill preset: {preset_path}")
        try:
            return RecDistillConfig(**normalize_recdistill_config(preset.config))
        except ValidationError as e:
            raise ValueError(f"Invalid RecDistill config inside preset {preset_path}:\n{e}")

    def load_dataset_config(self, dataset_name: str) -> DataConfig:
        """
        Load dataset configuration.
        
        Args:
            dataset_name: Name of dataset (amazon_cd, bookcrossing, citeulike)
            
        Returns:
            Validated DataConfig object
        """
        config_path = self.root / "dataset" / f"{dataset_name}.yaml"
        data = self._load_yaml(config_path)
        try:
            return DataConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid dataset config for {dataset_name}:\n{e}")

    def load_model_config(self, model_type: str, model_name: str, framework: str | None = None) -> ModelConfig:
        """
        Load model configuration.
        
        Args:
            model_type: Type of model ('teacher' or 'student')
            model_name: Name of model (bprmf, nmf, lgcn, etc.)
            
        Returns:
            Validated ModelConfig object
        """
        model_slug = str(model_name).strip().lower()
        framework_slug = str(framework).strip().lower() if framework else None
        if framework_slug:
            config_path = self.root / model_type / framework_slug / f"{model_slug}.yaml"
        else:
            config_path = self.root / model_type / f"{model_slug}.yaml"
            if not config_path.exists():
                default_path = self.root / model_type / "recbole" / f"{model_slug}.yaml"
                if default_path.exists():
                    config_path = default_path
                else:
                    matches = sorted((self.root / model_type).glob(f"*/{model_slug}.yaml"))
                    if len(matches) == 1:
                        config_path = matches[0]
                    elif len(matches) > 1:
                        choices = ", ".join(path.parent.name for path in matches)
                        raise ValueError(
                            f"Ambiguous model config for {model_type}/{model_name}. "
                            f"Specify framework. Available frameworks: {choices}."
                        )
        data = self._load_yaml(config_path)
        try:
            return ModelConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid model config for {model_type}/{model_name}:\n{e}")

    def compose_teacher_training(
        self,
        dataset_name: str,
        model_name: str,
        framework: str = "recbole",
        model_type: str = "teacher",
    ) -> Dict[str, Any]:
        """
        Compose a generic teacher training configuration.
        
        Args:
            dataset_name: Dataset name
            model_name: Model name
            framework: Framework implementation to use
            model_type: Model family to load ('teacher' or 'student')
            
        Returns:
            Complete teacher training configuration dictionary
        """
        self.load_dataset_config(dataset_name)
        model = self.load_model_config(model_type, model_name, framework=framework)
        
        config_path = self.root / "composites" / "teacher_template.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))
        train = template.setdefault("train_teacher", {})

        train["dataset"] = dataset_name
        train["teacher"] = {"default": _module_path("teacher", model.framework, model.model)}
        return template

    def compose_student_training(
        self,
        dataset_name: str,
        model_name: str,
        framework: str = "recbole",
        model_type: str = "student",
    ) -> Dict[str, Any]:
        """
        Compose a generic plain-student training configuration.
        """
        self.load_dataset_config(dataset_name)
        model = self.load_model_config(model_type, model_name, framework=framework)

        config_path = self.root / "composites" / "student_template.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))
        train = template.setdefault("train_student", {})

        train["dataset"] = dataset_name
        train["student"] = {"default": _module_path("student", model.framework, model.backbone)}
        return template

    def save_generated_experiment(
        self,
        *,
        kind: str,
        name: str,
        config: Dict[str, Any],
        path_parts: list[str],
    ) -> Path:
        """Persist an on-the-fly composed config as an experiment file."""
        safe_parts = [_slug(part) for part in path_parts if str(part).strip()]
        experiment_path = self.root / "experiments" / kind / Path(*safe_parts) / f"{_slug(name)}.yaml"
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        with experiment_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp, sort_keys=False, allow_unicode=False)
        return experiment_path

    def save_generated_preset(
        self,
        *,
        kind: str,
        family: str,
        name: str,
        config: Dict[str, Any],
        path_parts: list[str],
    ) -> Path:
        """Backward-compatible alias for saving generated experiment configs."""
        del family
        return self.save_generated_experiment(
            kind=kind,
            name=name,
            config=config,
            path_parts=path_parts,
        )

    def compose_recdistill_experiment(
        self,
        dataset_name: str,
        teacher_model: str,
        distiller_strategy: str,
        student_backbone: str = None,
        teacher_framework: str = "recbole",
        student_framework: str = "recbole",
    ) -> Dict[str, Any]:
        """
        Compose a complete RecDistill experiment configuration.
        
        Args:
            dataset_name: Dataset name
            teacher_model: Teacher model (bprmf, nmf, lgcn)
            distiller_strategy: Distillation strategy (de, htd, ftd, unkd)
            student_backbone: Student backbone (default: same as teacher)
            teacher_framework: Framework used to resolve the teacher model config
            student_framework: Framework used to resolve the student backbone config
            
        Returns:
            Complete distillation configuration dictionary
        """
        if student_backbone is None:
            student_backbone = teacher_model
        
        dataset = self.load_dataset_config(dataset_name)
        teacher_cfg = self.load_model_config("teacher", teacher_model, framework=teacher_framework)
        student_cfg = self.load_model_config("student", student_backbone, framework=student_framework)
        from recdistill.model_validation import validate_distillation_request

        validate_distillation_request(
            teacher_framework=teacher_cfg.framework,
            teacher_model=teacher_cfg.model,
            student_framework=student_cfg.framework,
            student_backbone=student_cfg.backbone,
            distiller=distiller_strategy,
        )
        
        config_path = self.root / "composites" / "recdistill_template.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))
        
        # Merge configs
        train = template["distill_student"]
        train["dataset"] = dataset_name
        train["teacher"] = {"default": _module_path("teacher", teacher_cfg.framework, teacher_cfg.model)}
        train["student"] = {"default": _module_path("student", student_cfg.framework, student_cfg.backbone)}
        train["distillation"] = {
            "default": f"distillation/{_slug(distiller_strategy)}.yaml",
            "strategy": str(distiller_strategy).replace("-", "_").upper(),
        }

        runtime = train.setdefault("runtime", {})
        runtime["output_path"] = str(
            distilled_student_artifact_path(
                distiller=distiller_strategy,
                teacher_framework=teacher_cfg.framework,
                teacher_model=teacher_cfg.model,
                student_framework=student_cfg.framework,
                student_model=student_cfg.backbone,
                dataset=dataset_name,
                embedding_dim=student_cfg.embedding_dim,
                strategy="fixed",
            ).relative_to(Path(__file__).resolve().parents[1])
        ).replace("\\", "/")
        
        return template

    @staticmethod
    def _copy_model_specific_fields(model: ModelConfig, target: Dict[str, Any]) -> None:
        for key in ("num_layers", "lightgcn_layers", "mlp_hidden_size", "mlp_dims"):
            value = getattr(model, key, None)
            if value is not None:
                target[key] = value

    @staticmethod
    def _remove_none_values(target: Dict[str, Any]) -> None:
        for key in list(target):
            if target[key] is None:
                del target[key]

    @staticmethod
    def _elliot_model_key(backbone: str) -> str:
        mapping = {
            "BPRMF": "torch.BPRMF",
            "BPR": "torch.BPRMF",
            "LGCN": "torch.LightGCN",
            "LIGHTGCN": "torch.LightGCN",
            "NGCF": "torch.NGCF",
            "DGCF": "torch.DGCF",
            "SGL": "torch.SGL",
            "ULTRAGCN": "torch.UltraGCN",
            "ULTRA_GCN": "torch.UltraGCN",
            "NMF": "NeuMFTorch",
            "NFM": "NeuMFTorch",
            "NEUMF": "NeuMFTorch",
        }
        normalized = backbone.upper()
        if normalized not in mapping:
            raise ValueError(f"Unsupported Elliot model backbone: {backbone}")
        return mapping[normalized]

    @staticmethod
    def _elliot_model_label(backbone: str) -> str:
        normalized = backbone.upper()
        if normalized in {"LIGHTGCN"}:
            return "LGCN"
        if normalized in {"ULTRA_GCN"}:
            return "ULTRAGCN"
        if normalized in {"NFM", "NEUMF"}:
            return "NMF"
        return normalized

    def clear_cache(self):
        """Clear configuration cache."""
        self._cache.clear()

    def list_datasets(self) -> list[str]:
        """List available datasets."""
        datasets_dir = self.root / "dataset"
        return [f.stem for f in datasets_dir.glob("*.yaml")]

    def list_models(self, model_type: str = None) -> Dict[str, list[str]]:
        """List available models by type."""
        if model_type:
            type_dir = self.root / model_type
            return {
                model_type: [
                    path.relative_to(type_dir).with_suffix("").as_posix()
                    for path in sorted(type_dir.rglob("*.yaml"))
                ]
            }
        else:
            result = {}
            for model_type_name in ("teacher", "student"):
                type_dir = self.root / model_type_name
                if type_dir.exists():
                    result[model_type_name] = [
                        path.relative_to(type_dir).with_suffix("").as_posix()
                        for path in sorted(type_dir.rglob("*.yaml"))
                    ]
            return result

    def list_distillers(self) -> list[str]:
        """List available distiller strategies."""
        distillers_dir = self.root / "distillation"
        return [f.stem for f in distillers_dir.glob("*.yaml")]

    def list_presets(self, kind: Optional[str] = None) -> list[str]:
        """List experiment files relative to config/experiments."""
        return self.list_experiments(kind)

    def list_experiments(self, kind: Optional[str] = None) -> list[str]:
        """List experiment files relative to config/experiments."""
        experiments_dir = self.root / "experiments"
        if not experiments_dir.exists():
            return []
        files = sorted(path for path in experiments_dir.rglob("*.yaml") if path.is_file())
        if kind is not None:
            prefix = kind.lower()
            files = [
                path
                for path in files
                if path.relative_to(experiments_dir).parts[0].lower() == prefix
            ]
        return [path.relative_to(experiments_dir).as_posix() for path in files]


# Global config loader instance
_global_loader: Optional[ConfigLoader] = None


def get_config_loader(config_root: Optional[Path] = None) -> ConfigLoader:
    """Get or create global config loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(config_root)
    return _global_loader


def reset_config_loader():
    """Reset global config loader instance."""
    global _global_loader
    _global_loader = None


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _slug(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("+", "_")


def _module_path(kind: str, framework: str, model: str) -> str:
    return f"{kind}/{_slug(framework)}/{_slug(model)}.yaml"
