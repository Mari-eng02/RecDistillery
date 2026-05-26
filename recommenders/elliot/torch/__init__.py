__all__ = [
    "BPRMFModel",
    "DGCFModel",
    "LightGCNModel",
    "NGCFModel",
    "SGLModel",
    "UltraGCNModel",
]


def __getattr__(name: str):
    if name == "BPRMFModel":
        from recommenders.elliot.torch.bprmf import BPRMFModel

        return BPRMFModel
    if name == "DGCFModel":
        from recommenders.elliot.torch.dgcf import DGCFModel

        return DGCFModel
    if name == "LightGCNModel":
        from recommenders.elliot.torch.lightgcn import LightGCNModel

        return LightGCNModel
    if name == "NGCFModel":
        from recommenders.elliot.torch.ngcf import NGCFModel

        return NGCFModel
    if name == "SGLModel":
        from recommenders.elliot.torch.sgl import SGLModel

        return SGLModel
    if name == "UltraGCNModel":
        from recommenders.elliot.torch.ultragcn import UltraGCNModel

        return UltraGCNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
