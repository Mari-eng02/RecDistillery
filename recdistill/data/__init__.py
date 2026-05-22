__all__ = [
    "InteractionBatch",
    "RRDAuxBatch",
    "InteractionDataset",
    "load_train_dataset",
    "load_eval_split",
    "load_interaction_dataset",
]


def __getattr__(name: str):
    if name == "InteractionBatch":
        from recdistill.data.batch import InteractionBatch

        return InteractionBatch
    if name == "RRDAuxBatch":
        from recdistill.data.batch import RRDAuxBatch

        return RRDAuxBatch
    if name == "InteractionDataset":
        from recdistill.data.interactions import InteractionDataset

        return InteractionDataset
    if name == "load_train_dataset":
        from recdistill.data.datarec_loader import load_train_dataset

        return load_train_dataset
    if name == "load_eval_split":
        from recdistill.data.datarec_loader import load_eval_split

        return load_eval_split
    if name == "load_interaction_dataset":
        from recdistill.data.datarec_loader import load_interaction_dataset

        return load_interaction_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
