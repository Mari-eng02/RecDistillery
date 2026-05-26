from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainableBackbone:
    framework: str
    model: str
    aliases: tuple[str, ...]
    adapter: str
    implementation: str
    notes: str = ""


@dataclass(frozen=True)
class UnsupportedBackbone:
    framework: str
    model: str
    reason: str
    recommended_path: str


@dataclass(frozen=True)
class TorchCompatibleModel:
    framework: str
    name: str
    note: str = ""


REPORTED_TOTAL_MODELS: dict[str, int] = {
    "recbole": 91,
    "elliot": 64,
    "lenskit": 23,
}


REPORTED_TORCH_COMPATIBLE_COUNTS: dict[str, int] = {
    "recbole": 91,
    "elliot": 7,
    "lenskit": 5,
}


REPORTED_TORCH_COMPATIBLE_PERCENTAGES: dict[str, str] = {
    "recbole": "100%",
    "elliot": "10.8%",
    "lenskit": "21.7%",
    "total": "57.9%",
}


REPORTED_TOTAL_IMPORTED_MODELS = 178
REPORTED_TOTAL_TORCH_COMPATIBLE_MODELS = 103


RECBOLE_TORCH_COMPATIBLE_MODELS: tuple[str, ...] = (
    "AFM",
    "AutoInt",
    "DCN",
    "DCNV2",
    "DeepFM",
    "DSSM",
    "EulerNet",
    "FFM",
    "FiGNN",
    "FM",
    "FNN",
    "FwFM",
    "KD_DAGFM",
    "LR",
    "NFM",
    "PNN",
    "WideDeep",
    "xDeepFM",
    "ADMMSLIM",
    "AsymKNN",
    "BPR",
    "CDAE",
    "ConvNCF",
    "DGCF",
    "DiffRec",
    "DMF",
    "EASE",
    "ENMF",
    "FISM",
    "GCMC",
    "ItemKNN",
    "LightGCN",
    "LINE",
    "MacridVAE",
    "MultiDAE",
    "MultiVAE",
    "NAIS",
    "NCEPLRec",
    "NCL",
    "NeuMF",
    "NGCF",
    "NNCF",
    "Pop",
    "RaCT",
    "Random",
    "RecVAE",
    "SGL",
    "SimpleX",
    "SLIMElastic",
    "SpectralCF",
    "CFKG",
    "CKE",
    "KGAT",
    "KGCN",
    "KGIN",
    "KGNNLS",
    "KTUP",
    "MCCLK",
    "MKR",
    "RippleNet",
    "BERT4Rec",
    "Caser",
    "CORE",
    "DIEN",
    "DIN",
    "FDSA",
    "FEARec",
    "FOSSIL",
    "FPMC",
    "GCSAN",
    "GRU4Rec",
    "GRU4RecCPR",
    "GRU4RecF",
    "GRU4RecKG",
    "HGN",
    "HRM",
    "KSR",
    "LightSANs",
    "NARM",
    "NextItNet",
    "NPE",
    "RepeatNet",
    "S3Rec",
    "SASRec",
    "SASRecCPR",
    "SASRecF",
    "SHAN",
    "SINE",
    "SRGNN",
    "STAMP",
    "TransRec",
)


ELLIOT_TORCH_COMPATIBLE_MODELS: tuple[str, ...] = (
    "BPRMF",
    "DGCF",
    "LightGCN",
    "NGCF",
    "NeuMFTorch",
    "SGL",
    "UltraGCN",
)


LENSKIT_TORCH_COMPATIBLE_MODELS: tuple[str, ...] = (
    "BPR",
    "EASEScorer",
    "FlexMFExplicitScorer",
    "FlexMFImplicitScorer",
    "LightGCNScorer",
)


TORCH_COMPATIBLE_IMPORTED_MODELS: tuple[TorchCompatibleModel, ...] = tuple(
    TorchCompatibleModel("recbole", name) for name in RECBOLE_TORCH_COMPATIBLE_MODELS
) + tuple(
    TorchCompatibleModel(
        "elliot",
        name,
        "Torch implementation of Elliot NeuMF.",
    )
    for name in ELLIOT_TORCH_COMPATIBLE_MODELS
) + tuple(
    TorchCompatibleModel("lenskit", name) for name in LENSKIT_TORCH_COMPATIBLE_MODELS
)


TRAINABLE_BACKBONES: tuple[TrainableBackbone, ...] = (
    TrainableBackbone(
        framework="recbole",
        model="BPRMF",
        aliases=("BPR", "BPRMF"),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.bpr.BPR",
    ),
    TrainableBackbone(
        framework="recbole",
        model="LINE",
        aliases=("LINE",),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.line.LINE",
    ),
    TrainableBackbone(
        framework="recbole",
        model="LGCN",
        aliases=("LGCN", "LightGCN"),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.lightgcn.LightGCN",
    ),
    TrainableBackbone(
        framework="recbole",
        model="NGCF",
        aliases=("NGCF",),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.ngcf.NGCF",
    ),
    TrainableBackbone(
        framework="recbole",
        model="DGCF",
        aliases=("DGCF",),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.dgcf.DGCF",
    ),
    TrainableBackbone(
        framework="recbole",
        model="SGL",
        aliases=("SGL",),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.sgl.SGL",
    ),
    TrainableBackbone(
        framework="recbole",
        model="SPECTRALCF",
        aliases=("SpectralCF", "SPECTRALCF"),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.spectralcf.SpectralCF",
    ),
    TrainableBackbone(
        framework="recbole",
        model="NMF",
        aliases=("NMF", "NeuMF"),
        adapter="RecBoleBackboneAdapter",
        implementation="recommenders.recbole.model.general_recommender.neumf.NeuMF",
    ),
    TrainableBackbone(
        framework="elliot",
        model="BPRMF",
        aliases=("BPR", "BPRMF"),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.bprmf.BPRMFModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="NMF",
        aliases=("NMF", "NeuMF"),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.neural.NeuMF.neural_matrix_factorization_torch_model.NeuralMatrixFactorizationTorchModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="LGCN",
        aliases=("LGCN", "LightGCN"),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.lightgcn.LightGCNModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="NGCF",
        aliases=("NGCF",),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.ngcf.NGCFModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="DGCF",
        aliases=("DGCF",),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.dgcf.DGCFModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="SGL",
        aliases=("SGL",),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.sgl.SGLModel",
    ),
    TrainableBackbone(
        framework="elliot",
        model="ULTRAGCN",
        aliases=("UltraGCN", "ULTRAGCN"),
        adapter="ElliotBackboneAdapter",
        implementation="recommenders.elliot.torch.ultragcn.UltraGCNModel",
    ),
    TrainableBackbone(
        framework="lenskit",
        model="BPRMF",
        aliases=("BPRMF",),
        adapter="LensKitBackboneAdapter",
        implementation="recommenders.lenskit.flexmf._model.FlexMFModel",
        notes="LensKit FlexMF configured as matrix factorization without biases.",
    ),
    TrainableBackbone(
        framework="lenskit",
        model="LGCN",
        aliases=("LGCN", "LightGCN"),
        adapter="LensKitBackboneAdapter",
        implementation="recommenders.lenskit.graphs.lightgcn.LightGCN",
    ),
)


UNSUPPORTED_KNOWN_BACKBONES: tuple[UnsupportedBackbone, ...] = (
    UnsupportedBackbone(
        framework="lenskit",
        model="NMF",
        reason="The imported LensKit models do not include a native NeuMF/NMF implementation.",
        recommended_path="Use RecBole/Elliot NeuMF, or import an external teacher with import_teacher.py.",
    ),
)


def torch_compatible_by_framework() -> dict[str, list[TorchCompatibleModel]]:
    grouped: dict[str, list[TorchCompatibleModel]] = {}
    for model in TORCH_COMPATIBLE_IMPORTED_MODELS:
        grouped.setdefault(model.framework, []).append(model)
    return grouped


def torch_compatible_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for framework in ("recbole", "elliot", "lenskit"):
        rows.append(
            {
                "framework": framework,
                "total_imported": str(REPORTED_TOTAL_MODELS[framework]),
                "torch_compatible": str(REPORTED_TORCH_COMPATIBLE_COUNTS[framework]),
                "percentage": REPORTED_TORCH_COMPATIBLE_PERCENTAGES[framework],
            }
        )
    rows.append(
        {
            "framework": "total",
            "total_imported": str(REPORTED_TOTAL_IMPORTED_MODELS),
            "torch_compatible": str(REPORTED_TOTAL_TORCH_COMPATIBLE_MODELS),
            "percentage": REPORTED_TORCH_COMPATIBLE_PERCENTAGES["total"],
        }
    )
    return rows


def trainable_by_framework() -> dict[str, list[TrainableBackbone]]:
    grouped: dict[str, list[TrainableBackbone]] = {}
    for backbone in TRAINABLE_BACKBONES:
        grouped.setdefault(backbone.framework, []).append(backbone)
    return grouped


def unsupported_by_framework() -> dict[str, list[UnsupportedBackbone]]:
    grouped: dict[str, list[UnsupportedBackbone]] = {}
    for backbone in UNSUPPORTED_KNOWN_BACKBONES:
        grouped.setdefault(backbone.framework, []).append(backbone)
    return grouped
