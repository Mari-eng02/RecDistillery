from pathlib import Path

from recommenders._framework_interface import FrameworkModelInterface, ModelEntry


models = FrameworkModelInterface(
    framework="recbole",
    package=f"{__package__}.model",
    root=Path(__file__).resolve().parent / "model",
    base_names=("GeneralRecommender", "SequentialRecommender", "KnowledgeRecommender", "ContextRecommender"),
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
