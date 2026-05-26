from __future__ import annotations


MODEL_ALIASES = {
    "bprmf": "BPRMF",
    "bpr": "BPRMF",
    "line": "LINE",
    "lgcn": "LGCN",
    "lightgcn": "LGCN",
    "ngcf": "NGCF",
    "dgcf": "DGCF",
    "sgl": "SGL",
    "ultragcn": "ULTRAGCN",
    "ultra_gcn": "ULTRAGCN",
    "spectralcf": "SPECTRALCF",
    "spectral_cf": "SPECTRALCF",
    "nmf": "NMF",
    "nfm": "NMF",
    "neumf": "NMF",
    "neumftorch": "NMF",
}

DISTILLER_ALIASES = {
    "de": "DE",
    "distillation_experts": "DE",
    "rrd": "RRD",
    "relaxed_ranking_distillation": "RRD",
    "unkd": "UNKD",
    "unkd_distillation": "UNKD",
    "htd": "HTD",
    "hierarchical_topology_distillation": "HTD",
    "ftd": "FTD",
    "full_topology_distillation": "FTD",
}

SUPPORTED_BACKBONES = frozenset({"BPRMF", "LINE", "LGCN", "NGCF", "DGCF", "SGL", "ULTRAGCN", "SPECTRALCF", "NMF"})
SUPPORTED_DISTILLERS = frozenset({"DE", "RRD", "UNKD", "HTD", "FTD"})


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace("+", "_")


def canonical_model_name(value: str) -> str:
    key = _normalize_key(value)
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unsupported model/backbone '{value}'. "
            f"Supported aliases: {', '.join(sorted(MODEL_ALIASES))}."
        )
    return MODEL_ALIASES[key]


def canonical_distiller_name(value: str) -> str:
    key = _normalize_key(value)
    if key not in DISTILLER_ALIASES:
        raise ValueError(
            f"Unsupported distiller '{value}'. "
            f"Supported aliases: {', '.join(sorted(DISTILLER_ALIASES))}."
        )
    return DISTILLER_ALIASES[key]


def parse_distiller_methods(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = str(value).strip()
    if _normalize_key(raw) in DISTILLER_ALIASES:
        return (canonical_distiller_name(raw),)
    normalized = raw.replace("-", "_").replace("+", "_")
    if not normalized:
        return ()
    return tuple(canonical_distiller_name(part) for part in normalized.split("_") if part)


def distiller_slug(value: str | None) -> str:
    methods = parse_distiller_methods(value)
    if not methods:
        return "none"
    return "_".join(method.lower() for method in methods)


def model_slug(value: str) -> str:
    return canonical_model_name(value).lower()
