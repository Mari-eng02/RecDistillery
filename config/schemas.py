"""
Pydantic schemas for configuration validation.
Centralizes all configuration structure definitions for Elliot and RecDistill.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from recdistill.registry import canonical_model_name, parse_distiller_methods


class DataConfig(BaseModel):
    """Dataset configuration."""
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="Dataset name (amazon_cd, bookcrossing, citeulike)")
    train_path: str = Field(..., description="Path to training data")
    test_path: str = Field(..., description="Path to test data")
    validation_path: Optional[str] = Field(default=None, description="Path to validation data")
    side_information: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("train_path", "test_path", "validation_path")
    @classmethod
    def validate_existing_dataset_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if ".." in Path(value).parts:
            return value
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {value}")
        return value


class ModelConfig(BaseModel):
    """Model configuration (common for Teacher/Student)."""
    model_config = ConfigDict(extra="allow")
    
    framework: str = Field(default="recbole", description="Framework implementation (recbole, elliot, lenskit)")
    backbone: Optional[str] = Field(default=None, description="Student backbone (BPRMF, NMF, NGCF, LGCN, etc.)")
    model: Optional[str] = Field(default=None, description="Teacher model (BPRMF, NMF, NGCF, LGCN, etc.)")
    embedding_dim: int = Field(..., description="Embedding dimension")
    learning_rate: float = Field(default=0.001)
    l2_reg: float = Field(default=0.0001)
    dropout: float = Field(default=0.0)
    factors: Optional[int] = Field(default=None)  # For legacy compatibility

    @model_validator(mode="after")
    def normalize_model_name(self) -> "ModelConfig":
        name = self.backbone or self.model
        if name is None:
            raise ValueError("Model config must define either backbone or model.")
        normalized = canonical_model_name(name)
        self.backbone = normalized
        self.model = normalized
        self.framework = str(self.framework).strip().lower()
        return self

    @field_validator("backbone", "model")
    @classmethod
    def normalize_optional_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return canonical_model_name(value)


class OptimizationConfig(BaseModel):
    """Training optimization parameters."""
    epochs: int = Field(default=100)
    batch_size: int = Field(default=512)
    learning_rate: float = Field(default=0.001)
    l2_reg: float = Field(default=0.0001)
    validation_rate: int = Field(default=10)
    validation_metric: str = Field(default="nDCGRendle2020@20")
    bayesian: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    cutoffs: List[int] = Field(default=[10, 20, 50])
    k: int = Field(default=20)
    every: int = Field(default=5)
    batch_size: int = Field(default=1024)
    simple_metrics: List[str] = Field(default=["nDCGRendle2020", "Recall", "Precision", "HR"])
    val_only: bool = Field(default=True)
    selection_split: str = Field(default="val")
    selection_metric: str = Field(default="ndcg")
    enabled: bool = Field(default=True)
    assert_no_train_leak: bool = Field(default=True)
    relevance_threshold: int = Field(default=0)

    @field_validator("selection_split")
    @classmethod
    def validate_selection_split(cls, value: str) -> str:
        normalized = str(value).lower()
        if normalized not in {"val", "test"}:
            raise ValueError("selection_split must be either 'val' or 'test'.")
        return normalized

    @field_validator("selection_metric")
    @classmethod
    def validate_selection_metric(cls, value: str) -> str:
        normalized = str(value).lower()
        if normalized not in {"precision", "recall", "ndcg", "hr"}:
            raise ValueError("selection_metric must be one of precision, recall, ndcg, hr.")
        return normalized

    @model_validator(mode="after")
    def validate_selection_policy(self) -> "EvaluationConfig":
        if self.val_only and self.selection_split == "test":
            raise ValueError("evaluation.val_only=true is incompatible with selection_split='test'.")
        return self


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    enabled: bool = Field(default=True)
    mode: str = Field(default="val_metric")  # val_metric or check
    metric: str = Field(default="ndcg")
    patience: int = Field(default=10)
    min_delta: float = Field(default=0.0)


class RuntimeConfig(BaseModel):
    """Runtime configuration."""
    seed: int = Field(default=42)
    device: Optional[str] = Field(default=None)
    num_workers: int = Field(default=4)
    output_path: Optional[str] = Field(default=None)
    output_strategy: str = Field(default="fixed")
    save_every: int = Field(default=0)
    gpu: int = Field(default=0)
    backend: str = Field(default="pytorch")
    wandb: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    extra_args: List[str] = Field(default_factory=list)


class DistillerConfig(BaseModel):
    """Knowledge distillation configuration."""
    model_config = ConfigDict(extra="allow")
    
    strategy: str = Field(..., description="Distillation strategy (DE, HTD, FTD, UnKD, etc.)")
    temperature: float = Field(default=3.0)
    lambda_kl: float = Field(default=0.5)
    lambda_de: float = Field(default=0.1)
    # Distillation Experts specific
    num_experts: int = Field(default=20)
    # UnKD specific
    unkd: Optional[Dict[str, Any]] = Field(default=None)
    # Add more distiller-specific configs as needed

    @field_validator("strategy")
    @classmethod
    def normalize_strategy(cls, value: str) -> str:
        methods = parse_distiller_methods(value)
        if not methods:
            raise ValueError("distillation.strategy must contain at least one distiller.")
        if "HTD" in methods and "FTD" in methods:
            raise ValueError("HTD and FTD cannot be active at the same time.")
        return "_".join(methods)


class TeacherConfig(BaseModel):
    """Teacher model configuration."""
    model_config = ConfigDict(extra="allow")

    model: str = Field(...)
    embedding_dim: Optional[int] = Field(default=None)
    path: Optional[str] = Field(default=None)
    framework: str = Field(default="auto")
    format: str = Field(default="auto")

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        try:
            return canonical_model_name(value)
        except ValueError:
            return str(value)


class StudentConfig(BaseModel):
    """Student model configuration."""
    model_config = ConfigDict(extra="allow")
    
    framework: str = Field(default="recbole")
    backbone: str = Field(...)
    embedding_dim: int = Field(...)
    lambda_de: float = Field(default=0.1)
    num_experts: int = Field(default=20)
    temperature: float = Field(default=0.1)

    @field_validator("backbone")
    @classmethod
    def normalize_backbone(cls, value: str) -> str:
        return canonical_model_name(value)


# ============================================================================
# RECDISTILL CONFIGS
# ============================================================================


class DistillStudentTrainingConfig(BaseModel):
    """RecDistill student training configuration."""
    model_config = ConfigDict(extra="allow")
    
    dataset: str = Field(...)
    teacher: TeacherConfig = Field(...)
    student: StudentConfig = Field(...)
    optimization: OptimizationConfig = Field(...)
    distillation: DistillerConfig = Field(...)
    runtime: RuntimeConfig = Field(...)
    evaluation: EvaluationConfig = Field(...)
    early_stopping: Optional[EarlyStoppingConfig] = Field(default_factory=EarlyStoppingConfig)

    @model_validator(mode="after")
    def validate_cross_config_contracts(self) -> "DistillStudentTrainingConfig":
        if self.teacher.embedding_dim is not None and self.teacher.embedding_dim <= self.student.embedding_dim:
            raise ValueError("teacher.embedding_dim must be greater than student.embedding_dim.")

        strategy_methods = set(parse_distiller_methods(self.distillation.strategy))

        active_methods = strategy_methods
        lambda_by_method = {
            "DE": float(getattr(self.distillation, "lambda_de", 0.0)),
            "RRD": float(getattr(self.distillation, "lambda_rrd", 0.0)),
            "UNKD": float(getattr(self.distillation, "lambda_unkd", 0.0)),
        }
        topology = getattr(self.distillation, "topology", {}) or {}
        lambda_td = float(topology.get("lambda_td", getattr(self.distillation, "lambda_td", 0.0)))
        if lambda_td > 0.0 and {"HTD", "FTD"}.isdisjoint(active_methods):
            raise ValueError("lambda_td > 0 requires HTD or FTD in distillation.strategy.")
        for method, value in lambda_by_method.items():
            if value > 0.0 and method not in active_methods:
                raise ValueError(f"lambda for {method} is > 0 but {method} is not active.")
        return self


class RecDistillConfig(BaseModel):
    """Root RecDistill configuration."""
    distill_student: DistillStudentTrainingConfig = Field(...)


class PresetMetadata(BaseModel):
    """Metadata for experiment presets."""
    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(default=1)
    kind: str = Field(..., description="Preset kind, for example recdistill, teacher, student, or raw")
    family: str = Field(..., description="Logical preset family")


class ConfigPreset(BaseModel):
    """Wrapped preset preserving config plus metadata."""
    model_config = ConfigDict(extra="allow")

    preset: PresetMetadata = Field(...)
    config: Dict[str, Any] = Field(...)
