from __future__ import annotations

from typing import Any

from recdistill.data.interactions import InteractionDataset
from recdistill.distillers import CompositeDistiller, DEDistiller, FTDistiller, HTDistiller, RRDDistiller
from recdistill.distillers.base import Distiller
from recdistill.framework_backbone import build_framework_backbone_adapter
from recdistill.registry import canonical_model_name
from recdistill.distillers.unkd import UnKDDistiller
from recdistill.samplers import RRDSampler, TeacherTopKProvider
from recdistill.teachers.state import TeacherState


def normalize_backbone_name(backbone: str) -> str:
    return canonical_model_name(backbone)


def parse_mlp_dims(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(part) for part in value)


def build_student_model(
    *,
    backbone: str,
    dataset: InteractionDataset,
    embedding_dim: int,
    l2_reg: float = 0.0,
    lightgcn_layers: int = 2,
    neumf_mlp_dims: str | list[int] | tuple[int, ...] = "64,32,16,8",
    neumf_dropout: float = 0.0,
    framework: str = "recbole",
    graph_builder=None,
):
    backbone = normalize_backbone_name(backbone)
    del graph_builder
    return build_framework_backbone_adapter(
        framework=framework,
        backbone=backbone,
        dataset=dataset,
        embedding_dim=int(embedding_dim),
        l2_reg=float(l2_reg),
        lightgcn_layers=int(lightgcn_layers),
        neumf_mlp_dims=parse_mlp_dims(neumf_mlp_dims),
        neumf_dropout=float(neumf_dropout),
    )


def build_distiller_from_args(args: Any, teacher_state: TeacherState, student_dim: int) -> Distiller | None:
    active_distillers: list[Distiller] = []
    if float(getattr(args, "lambda_de", 0.0)) > 0:
        if not teacher_state.has_embeddings:
            raise ValueError("DE distillation requires an embedding-based teacher.")
        active_distillers.append(
            DEDistiller(
                teacher_dim=teacher_state.embedding_dim,
                student_dim=int(student_dim),
                num_experts=int(args.num_experts),
                lambda_de=float(args.lambda_de),
                temperature=float(args.temperature),
            )
        )

    if float(getattr(args, "lambda_rrd", 0.0)) > 0:
        sampler = RRDSampler(
            interesting_size=int(args.rrd_interesting_size),
            uninteresting_size=int(args.rrd_uninteresting_size),
            temperature=float(args.rrd_temperature),
            topk_provider=TeacherTopKProvider(top_k=int(args.rrd_teacher_topk)),
        )
        active_distillers.append(RRDDistiller(sampler=sampler, lambda_rrd=float(args.lambda_rrd)))

    if float(getattr(args, "lambda_unkd", 0.0)) > 0:
        active_distillers.append(
            UnKDDistiller(
                lambda_unkd=float(args.lambda_unkd),
                sample_num=int(args.unkd_sample_num),
                group_count=int(args.unkd_group_count),
                popularity_lambda=float(args.unkd_popularity_lambda),
                rank_top_k=int(args.unkd_rank_top_k),
                rank_temperature=float(args.unkd_rank_temperature),
            )
        )

    if float(getattr(args, "lambda_td", 0.0)) > 0:
        if not teacher_state.has_embeddings:
            raise ValueError("HTD/FTD topology distillation requires an embedding-based teacher.")
        topology_type = str(args.td_type).upper()
        if topology_type == "FTD":
            active_distillers.append(
                FTDistiller(
                    lambda_td=float(args.lambda_td),
                    entity_sample_size=int(args.td_entity_sample_size),
                )
            )
        else:
            active_distillers.append(
                HTDistiller(
                    lambda_td=float(args.lambda_td),
                    alpha=float(args.htd_alpha),
                    num_groups=int(args.htd_num_groups),
                    topology_mode=str(args.htd_topology_mode),
                    initial_tau=float(args.htd_initial_tau),
                    min_tau=float(args.htd_min_tau),
                    decay_epochs=int(args.htd_decay_epochs),
                    entity_sample_size=int(args.td_entity_sample_size),
                )
            )

    if not active_distillers:
        return None
    if len(active_distillers) == 1:
        return active_distillers[0]
    return CompositeDistiller(active_distillers)


def build_student_from_config(
    student_config: Any,
    dataset: InteractionDataset,
    *,
    optimization_config: Any | None = None,
    graph_builder=None,
):
    return build_student_model(
        backbone=student_config.backbone,
        dataset=dataset,
        embedding_dim=student_config.embedding_dim,
        l2_reg=getattr(optimization_config, "l2_reg", 0.0) if optimization_config is not None else 0.0,
        lightgcn_layers=getattr(student_config, "num_layers", getattr(student_config, "lightgcn_layers", 2)),
        neumf_mlp_dims=getattr(student_config, "mlp_hidden_size", getattr(student_config, "mlp_dims", "64,32,16,8")),
        neumf_dropout=getattr(student_config, "dropout", 0.0),
        framework=getattr(student_config, "framework", "recbole"),
        graph_builder=graph_builder,
    )


def build_distiller_from_config(
    distillation_config: Any,
    *,
    teacher_state: TeacherState,
    student_dim: int,
) -> Distiller | None:
    class _Args:
        pass

    args = _Args()
    args.lambda_de = getattr(distillation_config, "lambda_de", 0.0)
    args.num_experts = getattr(distillation_config, "num_experts", 10)
    args.temperature = getattr(distillation_config, "temperature", 1.0)
    args.lambda_rrd = getattr(distillation_config, "lambda_rrd", 0.0)
    args.rrd_interesting_size = _nested_get(distillation_config, "rrd", "interesting_size", 10)
    args.rrd_uninteresting_size = _nested_get(distillation_config, "rrd", "uninteresting_size", 50)
    args.rrd_temperature = _nested_get(distillation_config, "rrd", "temperature", 1.0)
    args.rrd_teacher_topk = _nested_get(distillation_config, "rrd", "teacher_topk", 500)
    args.lambda_unkd = getattr(distillation_config, "lambda_unkd", 0.0)
    args.unkd_sample_num = _nested_get(distillation_config, "unkd", "sample_num", 30)
    args.unkd_group_count = _nested_get(distillation_config, "unkd", "group_count", 2)
    args.unkd_popularity_lambda = _nested_get(distillation_config, "unkd", "popularity_lambda", 1.0)
    args.unkd_rank_top_k = _nested_get(distillation_config, "unkd", "rank_top_k", 1000)
    args.unkd_rank_temperature = _nested_get(distillation_config, "unkd", "rank_temperature", 20.0)
    topology = getattr(distillation_config, "topology", None) or {}
    args.lambda_td = _mapping_get(topology, "lambda_td", getattr(distillation_config, "lambda_td", 0.0))
    args.td_type = _mapping_get(topology, "type", getattr(distillation_config, "strategy", "HTD"))
    args.td_entity_sample_size = _mapping_get(topology, "entity_sample_size", 0)
    args.htd_alpha = _mapping_get(topology, "alpha", 0.5)
    args.htd_num_groups = _mapping_get(topology, "num_groups", 40)
    args.htd_topology_mode = _mapping_get(topology, "topology_mode", "group_pe")
    args.htd_initial_tau = _mapping_get(topology, "initial_tau", 1.0)
    args.htd_min_tau = _mapping_get(topology, "min_tau", 1e-10)
    args.htd_decay_epochs = _mapping_get(topology, "decay_epochs", 100)
    return build_distiller_from_args(args, teacher_state=teacher_state, student_dim=student_dim)


def _nested_get(config: Any, section: str, key: str, default: Any) -> Any:
    return _mapping_get(getattr(config, section, None), key, default)


def _mapping_get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default) if value is not None else default
