from pathlib import Path

from recommenders._framework_interface import FrameworkModelInterface, ModelEntry


models = FrameworkModelInterface(
    framework="lenskit",
    package=__package__,
    root=Path(__file__).resolve().parent,
    base_names=("Component", "Trainable", "UsesTrainer"),
    include_suffixes=("Scorer", "Selector", "Ranker", "Score"),
    include_names=("ALS", "BPR"),
    exclude_names=("BaseRec", "ALSBase", "FlexMFScorerBase", "UsesTrainer", "Placeholder"),
    exclude_suffixes=("Config", "Trainer", "Base", "Model", "Params", "Data", "Epoch", "Batch"),
)

list_models = models.list_models
model_names = models.model_names
available_models = models.model_names
categories = models.categories
models_by_category = models.models_by_category
get_model_entry = models.get_model_entry
load_model = models.load_model
load_module = models.load_module
is_model_importable = models.is_model_importable

__all__ = [
    "ModelEntry",
    "models",
    "list_models",
    "model_names",
    "available_models",
    "categories",
    "models_by_category",
    "get_model_entry",
    "load_model",
    "load_module",
    "is_model_importable",
]
