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
    ElliotConfig,
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

    def load_elliot_config(self, config_path: Union[str, Path]) -> ElliotConfig:
        """
        Load and validate Elliot configuration.
        
        Args:
            config_path: Path to Elliot config YAML file
            
        Returns:
            Validated ElliotConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        data = self._load_yaml(config_path)
        if isinstance(data, dict) and "preset" in data and "config" in data:
            data = data["config"]
        try:
            return ElliotConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid Elliot config in {config_path}:\n{e}")

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
        Elliot or RecDistill payload under `config`.
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

    def load_elliot_preset(self, preset_path: Union[str, Path]) -> ElliotConfig:
        """Load an Elliot preset and validate its inner config."""
        preset = self.load_preset(preset_path)
        if preset.preset.kind.lower() != "elliot":
            raise ValueError(f"Preset is not an Elliot preset: {preset_path}")
        try:
            return ElliotConfig(**preset.config)
        except ValidationError as e:
            raise ValueError(f"Invalid Elliot config inside preset {preset_path}:\n{e}")

    def load_dataset_config(self, dataset_name: str) -> DataConfig:
        """
        Load dataset configuration.
        
        Args:
            dataset_name: Name of dataset (amazon_cd, bookcrossing, citeulike)
            
        Returns:
            Validated DataConfig object
        """
        config_path = self.root / "datasets" / f"{dataset_name}.yaml"
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
            config_path = self.root / "models" / model_type / framework_slug / f"{model_slug}.yaml"
        else:
            config_path = self.root / "models" / model_type / f"{model_slug}.yaml"
            if not config_path.exists():
                default_path = self.root / "models" / model_type / "recbole" / f"{model_slug}.yaml"
                if default_path.exists():
                    config_path = default_path
                else:
                    matches = sorted((self.root / "models" / model_type).glob(f"*/{model_slug}.yaml"))
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
        
        config_path = self.root / "experiments" / "teacher_template.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))

        template["dataset"] = dataset_name
        template["teacher"]["framework"] = model.framework
        template["teacher"]["model"] = model.model
        template["teacher"]["embedding_dim"] = model.embedding_dim
        template["teacher"]["learning_rate"] = model.learning_rate
        template["teacher"]["l2_reg"] = model.l2_reg
        template["teacher"]["dropout"] = model.dropout
        self._copy_model_specific_fields(model, template["teacher"])
        self._remove_none_values(template["teacher"])
        template["optimization"]["learning_rate"] = model.learning_rate
        template["optimization"]["l2_reg"] = model.l2_reg
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

        config_path = self.root / "experiments" / "student_template.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))

        template["dataset"] = dataset_name
        template["student"]["framework"] = model.framework
        template["student"]["backbone"] = model.backbone
        template["student"]["embedding_dim"] = model.embedding_dim
        template["student"]["learning_rate"] = model.learning_rate
        template["student"]["l2_reg"] = model.l2_reg
        template["student"]["dropout"] = model.dropout
        self._copy_model_specific_fields(model, template["student"])
        self._remove_none_values(template["student"])
        template["optimization"]["learning_rate"] = model.learning_rate
        template["optimization"]["l2_reg"] = model.l2_reg
        return template

    def save_generated_preset(
        self,
        *,
        kind: str,
        family: str,
        name: str,
        config: Dict[str, Any],
        path_parts: list[str],
    ) -> Path:
        """Persist an on-the-fly composed config as a reusable preset."""
        safe_parts = [_slug(part) for part in path_parts if str(part).strip()]
        preset_path = self.root / "presets" / kind / family / Path(*safe_parts) / f"{_slug(name)}.yaml"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preset": {
                "schema_version": 1,
                "kind": kind,
                "family": family,
                "name": name,
                "generated": True,
            },
            "config": config,
        }
        with preset_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=False)
        return preset_path

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
        
        config_path = self.root / "experiments" / f"recdistill_template_{distiller_strategy}.yaml"
        template = copy.deepcopy(self._load_yaml(config_path))
        distiller_cfg = self._load_yaml(self.root / "distillers" / f"{distiller_strategy}.yaml")
        
        # Merge configs
        template["train_student"]["dataset"] = dataset_name
        template["train_student"]["teacher"]["framework"] = teacher_cfg.framework
        template["train_student"]["teacher"]["model"] = teacher_cfg.model
        template["train_student"]["teacher"]["embedding_dim"] = teacher_cfg.embedding_dim
        template["train_student"]["student"]["framework"] = student_cfg.framework
        template["train_student"]["student"]["backbone"] = student_cfg.backbone
        template["train_student"]["student"]["embedding_dim"] = student_cfg.embedding_dim
        self._copy_model_specific_fields(student_cfg, template["train_student"]["student"])
        template["train_student"]["distillation"].update(distiller_cfg)
        template["train_student"]["distillation"]["strategy"] = str(
            template["train_student"]["distillation"].get("strategy", distiller_strategy)
        ).upper()
        template["train_student"]["student"]["model"] = template["train_student"]["distillation"]["strategy"]

        runtime = template["train_student"].setdefault("runtime", {})
        runtime["output_path"] = str(
            distilled_student_artifact_path(
                distiller=distiller_strategy,
                teacher_framework=teacher_cfg.framework,
                teacher_model=teacher_cfg.model,
                student_framework=student_cfg.framework,
                student_model=student_cfg.backbone,
                dataset=dataset_name,
                embedding_dim=student_cfg.embedding_dim,
            ).relative_to(Path(__file__).resolve().parents[1])
        ).replace("\\", "/")
        
        from recdistill.config_integration import normalize_recdistill_config

        return RecDistillConfig(**normalize_recdistill_config(template)).model_dump()

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
            "BPRMF": "external.BPRMF",
            "LGCN": "external.LightGCN",
            "LIGHTGCN": "external.LightGCN",
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
        if normalized in {"NFM", "NEUMF"}:
            return "NMF"
        return normalized

    def clear_cache(self):
        """Clear configuration cache."""
        self._cache.clear()

    def list_datasets(self) -> list[str]:
        """List available datasets."""
        datasets_dir = self.root / "datasets"
        return [f.stem for f in datasets_dir.glob("*.yaml")]

    def list_models(self, model_type: str = None) -> Dict[str, list[str]]:
        """List available models by type."""
        models_dir = self.root / "models"
        if model_type:
            type_dir = models_dir / model_type
            return {
                model_type: [
                    path.relative_to(type_dir).with_suffix("").as_posix()
                    for path in sorted(type_dir.rglob("*.yaml"))
                ]
            }
        else:
            result = {}
            for type_dir in models_dir.iterdir():
                if type_dir.is_dir():
                    result[type_dir.name] = [
                        path.relative_to(type_dir).with_suffix("").as_posix()
                        for path in sorted(type_dir.rglob("*.yaml"))
                    ]
            return result

    def list_distillers(self) -> list[str]:
        """List available distiller strategies."""
        distillers_dir = self.root / "distillers"
        return [f.stem for f in distillers_dir.glob("*.yaml")]

    def list_presets(self, kind: Optional[str] = None) -> list[str]:
        """List preset files relative to config/presets."""
        presets_dir = self.root / "presets"
        if not presets_dir.exists():
            return []
        files = sorted(path for path in presets_dir.rglob("*.yaml") if path.is_file())
        if kind is not None:
            prefix = kind.lower()
            files = [
                path
                for path in files
                if path.relative_to(presets_dir).parts[0].lower() == prefix
            ]
        return [path.relative_to(presets_dir).as_posix() for path in files]


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


def _slug(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("+", "_")
