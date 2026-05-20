from __future__ import annotations

from recdistill.registry import canonical_model_name, parse_distiller_methods
from recdistill.supported_models import (
    TORCH_COMPATIBLE_IMPORTED_MODELS,
    TRAINABLE_BACKBONES,
    UNSUPPORTED_KNOWN_BACKBONES,
)

EMBEDDING_DISTILLERS = frozenset({"DE", "HTD", "FTD"})
SCORING_DISTILLERS = frozenset({"RRD", "UNKD"})
TOPOLOGY_DISTILLERS = frozenset({"HTD", "FTD"})

# DE builds its projection experts from the configured student.embedding_dim.
# These adapters expose layer-concatenated embeddings, so their effective
# representation size is embedding_dim * (num_layers + 1).
DE_FIXED_DIM_UNSAFE_STUDENTS = frozenset(
    {
        ("recbole", "ngcf"),
        ("recbole", "spectralcf"),
    }
)


def validate_trainable_model(framework: str, model: str, *, role: str = "model") -> str:
    """Return the canonical model name if the framework/model pair is trainable by RecDistill."""
    framework_key = _framework_key(framework)
    if model is None or not str(model).strip():
        raise ValueError(f"Missing {role}. Available {framework_key} models: {_available_models(framework_key)}.")
    raw_model = str(model).strip()
    model_key = _model_key(raw_model)

    for backbone in TRAINABLE_BACKBONES:
        if backbone.framework == framework_key and model_key in {_model_key(backbone.model), *map(_model_key, backbone.aliases)}:
            return backbone.model

    canonical = _canonical_or_raw(raw_model)
    unsupported = _unsupported_entry(framework_key, canonical)
    if unsupported is not None:
        raise ValueError(
            f"{role} '{raw_model}' cannot be trained with the RecDistill PyTorch loop for framework '{framework_key}'. "
            f"Reason: {unsupported.reason} Recommended path: {unsupported.recommended_path}"
        )

    other_frameworks = sorted(
        {
            backbone.framework
            for backbone in TRAINABLE_BACKBONES
            if model_key in {_model_key(backbone.model), *map(_model_key, backbone.aliases)}
        }
    )
    if other_frameworks:
        raise ValueError(
            f"{role} '{raw_model}' is not adapter-backed for framework '{framework_key}'. "
            f"It is currently available for: {', '.join(other_frameworks)}. "
            f"Available {framework_key} models: {_available_models(framework_key)}."
        )

    if _is_torch_compatible_import(framework_key, raw_model):
        raise ValueError(
            f"{role} '{raw_model}' from framework '{framework_key}' is torch-compatible in the imported definitions, "
            "but no RecDistill training adapter is currently wired for it. "
            f"Available adapter-backed {framework_key} models: {_available_models(framework_key)}."
        )

    raise ValueError(
        f"{role} '{raw_model}' is not available as a RecDistill PyTorch-trainable model for framework '{framework_key}'. "
        f"Available adapter-backed {framework_key} models: {_available_models(framework_key)}."
    )


def validate_distillation_request(
    *,
    teacher_framework: str | None,
    teacher_model: str | None,
    student_framework: str,
    student_backbone: str,
    distiller: str,
    validate_teacher: bool = True,
) -> tuple[str | None, str]:
    """Validate a composed distillation run and return canonical teacher/student names."""
    if distiller is None or not str(distiller).strip():
        raise ValueError("Distillation strategy is required. Choose one of: DE, RRD, UnKD, HTD, FTD.")
    canonical_teacher = None
    if validate_teacher and teacher_model:
        canonical_teacher = validate_trainable_model(
            teacher_framework or "recbole",
            teacher_model,
            role="teacher model",
        )
    canonical_student = validate_trainable_model(
        student_framework,
        student_backbone,
        role="student backbone",
    )

    try:
        methods = set(parse_distiller_methods(distiller))
    except ValueError as exc:
        raise ValueError(f"Unsupported distillation strategy '{distiller}'. {exc}") from exc
    if not methods:
        raise ValueError("Distillation strategy must contain at least one method.")
    if TOPOLOGY_DISTILLERS.issubset(methods):
        raise ValueError("Incompatible distillation strategy: HTD and FTD cannot be active at the same time.")
    _validate_student_backbone_distiller(
        student_framework=student_framework,
        student_backbone=canonical_student,
        methods=methods,
    )

    # Adapter capability constraints. All currently trainable backbones expose embeddings and pair scoring,
    # but keep this check explicit so new adapters fail early with a useful message.
    if methods & EMBEDDING_DISTILLERS and canonical_student is None:
        raise ValueError("DE/HTD/FTD require an embedding-based student backbone.")
    if methods & SCORING_DISTILLERS and canonical_student is None:
        raise ValueError("RRD/UnKD require a student backbone that can score user-item pairs.")
    return canonical_teacher, canonical_student


def validate_recdistill_config_dict(config: dict) -> None:
    train_conf = config.get("train_student", config)
    teacher_conf = train_conf.get("teacher", {}) or {}
    student_conf = train_conf.get("student", {}) or {}
    distill_conf = train_conf.get("distillation", {}) or {}
    strategy = distill_conf.get("strategy") or student_conf.get("model")
    validate_distillation_request(
        teacher_framework=teacher_conf.get("framework", "recbole"),
        teacher_model=teacher_conf.get("model"),
        student_framework=student_conf.get("framework", "recbole"),
        student_backbone=student_conf.get("backbone"),
        distiller=strategy,
        validate_teacher=not bool(teacher_conf.get("path")),
    )
    validate_teacher_representation_request(
        teacher_conf=teacher_conf,
        distiller=strategy,
    )


def validate_teacher_representation_request(*, teacher_conf: dict, distiller: str | None) -> None:
    try:
        methods = set(parse_distiller_methods(distiller))
    except ValueError:
        return
    if not methods:
        return

    representations = _declared_teacher_representations(teacher_conf)
    _validate_teacher_representation_shape(teacher_conf, representations)

    if len(representations) > 1:
        raise ValueError(
            "Ambiguous teacher representation: the config declares multiple teacher formats "
            f"({', '.join(sorted(representations))}). Choose exactly one of embeddings, scores, or top-k/ranking."
        )

    if not (methods & EMBEDDING_DISTILLERS):
        return

    representation = next(iter(representations), None)
    if representation in {"scores", "topk", "ranking"}:
        raise ValueError(
            f"Incompatible teacher representation/distiller: {', '.join(sorted(methods & EMBEDDING_DISTILLERS))} "
            f"requires an embedding-based teacher, but the config declares a {representation}-based teacher. "
            "Use RRD or UnKD with score/ranking teachers, or provide/import user and item embeddings."
        )


def validate_loaded_teacher_for_distillation(teacher_state, distiller: str | None) -> None:
    methods = set(parse_distiller_methods(distiller))
    if methods & EMBEDDING_DISTILLERS and not bool(getattr(teacher_state, "has_embeddings", False)):
        available = "score/ranking scorer" if getattr(teacher_state, "scorer", None) is not None else "no embedding representation"
        raise ValueError(
            f"Incompatible loaded teacher/distiller: {', '.join(sorted(methods & EMBEDDING_DISTILLERS))} "
            f"requires user/item embeddings, but the loaded teacher provides {available}. "
            "Use RRD or UnKD with score/ranking teachers, or import an embedding-based .teacher."
        )


def _framework_key(framework: str | None) -> str:
    key = str(framework or "").strip().lower()
    if key not in {"recbole", "elliot", "lenskit"}:
        raise ValueError(f"Unsupported framework '{framework}'. Choose from: recbole, elliot, lenskit.")
    return key


def _declared_teacher_representations(teacher_conf: dict) -> set[str]:
    representations: set[str] = set()
    if teacher_conf.get("user_embeddings_path") or teacher_conf.get("item_embeddings_path"):
        representations.add("embeddings")
    if teacher_conf.get("score_matrix_path"):
        representations.add("scores")
    if teacher_conf.get("topk_items_path") or teacher_conf.get("topk_scores_path"):
        representations.add("topk")
    fmt = str(teacher_conf.get("format") or "").strip().lower()
    if fmt in {"scores", "score", "score_matrix", "precomputed_scores", "numpy_scores"}:
        representations.add("scores")
    elif fmt in {"topk", "ranking", "rankings", "precomputed_topk", "numpy_topk"}:
        representations.add("topk")
    elif fmt in {"embeddings", "embedding", "numpy_embeddings", "embeddings_npz"}:
        representations.add("embeddings")
    representation = str(teacher_conf.get("representation") or "").strip().lower()
    if representation in {"scores", "score", "score_matrix"}:
        representations.add("scores")
    elif representation in {"topk", "ranking", "rankings"}:
        representations.add("topk")
    elif representation in {"embeddings", "embedding"}:
        representations.add("embeddings")
    return representations


def _validate_teacher_representation_shape(teacher_conf: dict, representations: set[str]) -> None:
    has_user_embeddings = bool(teacher_conf.get("user_embeddings_path"))
    has_item_embeddings = bool(teacher_conf.get("item_embeddings_path"))
    if has_user_embeddings != has_item_embeddings:
        raise ValueError(
            "Invalid teacher embedding representation: provide both user_embeddings_path and item_embeddings_path, "
            "or neither."
        )

    has_topk_items = bool(teacher_conf.get("topk_items_path"))
    has_topk_scores = bool(teacher_conf.get("topk_scores_path"))
    if has_topk_scores and not has_topk_items:
        raise ValueError("Invalid teacher top-k representation: topk_scores_path requires topk_items_path.")


def _canonical_or_raw(model: str) -> str:
    try:
        return canonical_model_name(model)
    except ValueError:
        return str(model).strip()


def _unsupported_entry(framework: str, canonical_model: str):
    model_key = _model_key(canonical_model)
    for entry in UNSUPPORTED_KNOWN_BACKBONES:
        if entry.framework == framework and _model_key(entry.model) == model_key:
            return entry
    return None


def _is_torch_compatible_import(framework: str, model: str) -> bool:
    model_key = _model_key(model)
    canonical_key = _model_key(_canonical_or_raw(model))
    return any(
        imported.framework == framework
        and _model_key(imported.name) in {model_key, canonical_key}
        for imported in TORCH_COMPATIBLE_IMPORTED_MODELS
    )


def _available_models(framework: str) -> str:
    names = sorted(backbone.model for backbone in TRAINABLE_BACKBONES if backbone.framework == framework)
    return ", ".join(names) if names else "none"


def _validate_student_backbone_distiller(
    *,
    student_framework: str,
    student_backbone: str,
    methods: set[str],
) -> None:
    framework_key = _framework_key(student_framework)
    backbone_key = _model_key(student_backbone)

    if "DE" in methods and (framework_key, backbone_key) in DE_FIXED_DIM_UNSAFE_STUDENTS:
        raise ValueError(
            f"Incompatible student backbone/distiller: {student_framework} {student_backbone} cannot currently be used with DE. "
            "This adapter returns layer-concatenated embeddings whose effective dimension differs from student.embedding_dim, "
            "while DE builds fixed-size projection experts from student.embedding_dim. "
            "Use RRD, UnKD, HTD, or FTD for this backbone, or choose BPRMF, LINE, LGCN, DGCF, SGL, or NMF for DE."
        )


def _model_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace("+", "_")
