from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from config import get_config_loader
from recdistill.checkpointing import save_student_checkpoint
from recdistill.data.datarec_loader import load_eval_split, load_interaction_dataset
from recdistill.evaluation import evaluate_student
from recdistill.factories import build_student_model, normalize_backbone_name, parse_mlp_dims
from recdistill.paths import RESULTS_ROOT, STUDENT_EXT, TEACHER_EXT
from recdistill.teachers.serialization import save_teacher_state
from recdistill.teachers.state import PrecomputedScoresScorer, TeacherState
from recdistill.tracking import utc_now_iso
from recdistill.training import build_lightgcn_graph, build_train_loader, set_seed
from recdistill.trainers import DistillationTrainer


@dataclass
class NativeTrainingArgs:
    role: str
    dataset: str
    backbone: str
    embedding_dim: int
    framework: str = "recbole"
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.001
    l2_reg: float = 0.0001
    dropout: float = 0.0
    lightgcn_layers: int = 2
    neumf_mlp_dims: str = "64,32,16,8"
    seed: int = 42
    device: str | None = None
    num_workers: int = 0
    output_path: str | None = None
    save_every: int = 0
    skip_eval: bool = False
    eval_k: int = 20
    eval_every: int = 5
    eval_batch_size: int = 1024
    eval_val_only: bool = True
    selection_split: str = "val"
    selection_metric: str = "ndcg"
    assert_no_train_leak: bool = True


def native_args_from_model_config(
    *,
    role: str,
    dataset: str,
    backbone: str,
    overrides: dict[str, Any] | None = None,
) -> NativeTrainingArgs:
    loader = get_config_loader()
    framework = None
    if overrides:
        framework = overrides.get("framework")
    model_cfg = loader.load_model_config(role, backbone, framework=framework)
    args = NativeTrainingArgs(
        role=role,
        dataset=dataset,
        backbone=model_cfg.backbone,
        framework=getattr(model_cfg, "framework", "recbole"),
        embedding_dim=int(model_cfg.embedding_dim),
        learning_rate=float(model_cfg.learning_rate),
        l2_reg=float(model_cfg.l2_reg),
        dropout=float(model_cfg.dropout),
        lightgcn_layers=int(getattr(model_cfg, "num_layers", getattr(model_cfg, "lightgcn_layers", 2))),
        neumf_mlp_dims=_stringify_mlp_dims(getattr(model_cfg, "mlp_hidden_size", getattr(model_cfg, "mlp_dims", "64,32,16,8"))),
    )
    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(args, key):
                setattr(args, key, value)
    return args


def native_args_to_config(args: NativeTrainingArgs) -> dict[str, Any]:
    role = _normalize_role(args.role)
    model_section_name = "teacher" if role == "teacher" else "student"
    model_key = "model" if role == "teacher" else "backbone"
    model_conf = {
        "framework": args.framework,
        model_key: args.backbone,
        "embedding_dim": int(args.embedding_dim),
        "learning_rate": float(args.learning_rate),
        "l2_reg": float(args.l2_reg),
        "dropout": float(args.dropout),
    }
    if args.backbone in {"LGCN", "NGCF", "DGCF", "SGL", "SPECTRALCF"}:
        model_conf["num_layers"] = int(args.lightgcn_layers)
    if args.backbone == "NMF":
        model_conf["mlp_hidden_size"] = _parse_mlp_dims_for_config(args.neumf_mlp_dims)

    return {
        "dataset": args.dataset,
        model_section_name: model_conf,
        "optimization": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "l2_reg": float(args.l2_reg),
        },
        "runtime": {
            "seed": int(args.seed),
            "device": args.device,
            "num_workers": int(args.num_workers),
            "output_path": args.output_path,
            "save_every": int(args.save_every),
        },
        "evaluation": {
            "enabled": not bool(args.skip_eval),
            "k": int(args.eval_k),
            "every": int(args.eval_every),
            "batch_size": int(args.eval_batch_size),
            "val_only": bool(args.eval_val_only),
            "selection_split": args.selection_split,
            "selection_metric": args.selection_metric,
            "assert_no_train_leak": bool(args.assert_no_train_leak),
        },
    }


def native_args_from_config_file(
    path: str | Path,
    *,
    role: str,
    fallback_dataset: str | None = None,
    fallback_backbone: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> NativeTrainingArgs:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise ModuleNotFoundError("PyYAML is required to read YAML training configs.") from exc
        raw = yaml.safe_load(raw_text) or {}

    config = raw.get("config", raw) if isinstance(raw, dict) else {}
    train = config.get("train_student", config) if isinstance(config, dict) else {}
    model_section_name = "teacher" if _normalize_role(role) == "teacher" else "student"
    model_conf = train.get(model_section_name, train) if isinstance(train, dict) else {}
    optim_conf = train.get("optimization", {}) if isinstance(train, dict) else {}
    runtime_conf = train.get("runtime", {}) if isinstance(train, dict) else {}
    eval_conf = train.get("evaluation", {}) if isinstance(train, dict) else {}

    dataset = train.get("dataset") or config.get("dataset") or fallback_dataset
    backbone = model_conf.get("backbone") or model_conf.get("model") or fallback_backbone
    if dataset is None:
        raise ValueError(f"Missing dataset in {config_path}.")
    if backbone is None:
        raise ValueError(f"Missing {model_section_name} backbone/model in {config_path}.")

    def _override_or_config(key: str, source: dict[str, Any], default: Any = None) -> Any:
        if overrides and key in overrides and overrides[key] is not None:
            return overrides[key]
        value = source.get(key, default)
        return default if value is None else value

    embedding_dim = _override_or_config("embedding_dim", model_conf)
    if embedding_dim is None:
        raise ValueError(f"Missing {model_section_name}.embedding_dim in {config_path}.")
    lightgcn_layers = _first_not_none(
        overrides.get("lightgcn_layers") if overrides else None,
        model_conf.get("lightgcn_layers"),
        model_conf.get("num_layers"),
        model_conf.get("layers"),
        2,
    )
    mlp_dims = _first_not_none(
        overrides.get("neumf_mlp_dims") if overrides else None,
        model_conf.get("neumf_mlp_dims"),
        model_conf.get("mlp_hidden_size"),
        model_conf.get("mlp_dims"),
        "64,32,16,8",
    )

    args = NativeTrainingArgs(
        role=role,
        dataset=str(dataset),
        backbone=str(backbone),
        framework=str(_override_or_config("framework", model_conf, "recbole")),
        embedding_dim=int(embedding_dim),
        epochs=int(_override_or_config("epochs", optim_conf, 100)),
        batch_size=int(_override_or_config("batch_size", optim_conf, 512)),
        learning_rate=float(_override_or_config("learning_rate", optim_conf, model_conf.get("learning_rate", 0.001))),
        l2_reg=float(_override_or_config("l2_reg", optim_conf, model_conf.get("l2_reg", 0.0001))),
        dropout=float(_override_or_config("dropout", model_conf, 0.0)),
        lightgcn_layers=int(lightgcn_layers),
        neumf_mlp_dims=_stringify_mlp_dims(mlp_dims),
        seed=int(runtime_conf.get("seed", 42)),
        device=runtime_conf.get("device"),
        num_workers=int(runtime_conf.get("num_workers", 0)),
        output_path=runtime_conf.get("output_path"),
        save_every=int(runtime_conf.get("save_every", 0)),
        skip_eval=not bool(eval_conf.get("enabled", True)),
        eval_k=int(eval_conf.get("k", 20)),
        eval_every=int(eval_conf.get("every", 5)),
        eval_batch_size=int(eval_conf.get("batch_size", 1024)),
        eval_val_only=bool(eval_conf.get("val_only", True)),
        selection_split=str(eval_conf.get("selection_split", "val")),
        selection_metric=str(eval_conf.get("selection_metric", "ndcg")),
        assert_no_train_leak=bool(eval_conf.get("assert_no_train_leak", True)),
    )
    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(args, key):
                setattr(args, key, value)
    return args


class NativeModelTrainingRunner:
    def __init__(self, args: NativeTrainingArgs):
        self.args = args
        self.role = _normalize_role(args.role)
        self.device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = normalize_backbone_name(args.backbone)
        self.output_path = Path(args.output_path) if args.output_path else self._default_output_path()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.dataset = None
        self.val_dict: dict[int, set[int]] = {}
        self.test_dict: dict[int, set[int]] = {}
        self.model = None
        self.optimizer = None
        self.trainer = None

    def run_config(self) -> dict[str, Any]:
        payload = asdict(self.args)
        payload["role"] = self.role
        payload["backbone"] = self.backbone
        payload["framework"] = self.args.framework
        payload["output_path"] = str(self.output_path)
        return payload

    def prepare(self) -> None:
        print("\n" + "=" * 80)
        print(f"{self.role.capitalize()} Training")
        print("=" * 80)
        print(f"Dataset: {self.args.dataset}")
        print(f"Backbone: {self.backbone}")
        print(f"Framework: {self.args.framework}")
        print(f"Embedding dim: {self.args.embedding_dim}")
        print(f"Device: {self.device}")
        print("=" * 80 + "\n")

        self.dataset = load_interaction_dataset(self.args.dataset)
        print(f"Train interactions: {len(self.dataset.interactions)}")
        print(f"Users/items: {self.dataset.num_users}/{self.dataset.num_items}")

        if not self.args.skip_eval:
            self.val_dict, dropped_val = load_eval_split(
                dataset_name=self.args.dataset,
                split_name="val",
                teacher_num_users=self.dataset.num_users,
                teacher_num_items=self.dataset.num_items,
            )
            self.test_dict, dropped_test = load_eval_split(
                dataset_name=self.args.dataset,
                split_name="test",
                teacher_num_users=self.dataset.num_users,
                teacher_num_items=self.dataset.num_items,
            )
            print(f"Validation interactions: {sum(len(v) for v in self.val_dict.values())} (dropped: {dropped_val})")
            print(f"Test interactions: {sum(len(v) for v in self.test_dict.values())} (dropped: {dropped_test})")

        train_loader = build_train_loader(
            self.dataset,
            batch_size=int(self.args.batch_size),
            num_workers=int(self.args.num_workers),
        )
        self.model = build_student_model(
            backbone=self.backbone,
            dataset=self.dataset,
            embedding_dim=int(self.args.embedding_dim),
            l2_reg=float(self.args.l2_reg),
            lightgcn_layers=int(self.args.lightgcn_layers),
            neumf_mlp_dims=parse_mlp_dims(self.args.neumf_mlp_dims),
            neumf_dropout=float(self.args.dropout),
            framework=self.args.framework,
            graph_builder=build_lightgcn_graph,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate))
        self.trainer = DistillationTrainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=train_loader,
            distiller=None,
            device=self.device,
            dataset=self.dataset,
        )

    def run(self) -> dict[str, Any]:
        set_seed(int(self.args.seed))
        self.prepare()
        return self.train()

    def train(self) -> dict[str, Any]:
        assert self.dataset is not None
        assert self.model is not None
        assert self.optimizer is not None
        assert self.trainer is not None

        history: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_epoch = 0
        best_path = _with_role_suffix(self.output_path, f".best{TEACHER_EXT if self.role == 'teacher' else STUDENT_EXT}")

        for epoch in range(1, int(self.args.epochs) + 1):
            metrics = self.trainer.train_epoch()
            row: dict[str, Any] = {"epoch": epoch, **metrics}
            current_eval = None
            if not self.args.skip_eval and self.args.eval_every > 0 and epoch % int(self.args.eval_every) == 0:
                current_eval = evaluate_student(
                    model=self.model,
                    train_seen=self.dataset.train_dict,
                    val_gt=self.val_dict,
                    test_gt=self.test_dict,
                    top_k=int(self.args.eval_k),
                    batch_size=int(self.args.eval_batch_size),
                    device=self.device,
                    eval_val_only=bool(self.args.eval_val_only),
                )
                self._update_eval_row(row, current_eval)
                leaked_users_test = int(current_eval.get("leaked_users_test", 0))
                if self.args.assert_no_train_leak and (current_eval["leaked_users_val"] > 0 or leaked_users_test > 0):
                    raise RuntimeError(
                        "Train-item leakage detected in recommendations. "
                        f"val_leaks={current_eval['leaked_users_val']} test_leaks={leaked_users_test}"
                    )

                selected_metrics = current_eval[self.args.selection_split]
                selected_score = float(selected_metrics[self.args.selection_metric])
                row["selection_score"] = selected_score
                if selected_score > best_score:
                    best_score = selected_score
                    best_epoch = epoch
                    self._save_artifact(best_path, epoch, history + [row], best_epoch, best_score)

            history.append(row)
            self._print_epoch(epoch, metrics, current_eval)

            if self.args.save_every > 0 and epoch % int(self.args.save_every) == 0:
                periodic = _with_role_suffix(
                    self.output_path,
                    f".ep{epoch}{TEACHER_EXT if self.role == 'teacher' else STUDENT_EXT}",
                )
                self._save_artifact(periodic, epoch, history, best_epoch, best_score if best_epoch > 0 else None)

        final_epoch = int(history[-1]["epoch"]) if history else 0
        self._save_artifact(self.output_path, final_epoch, history, best_epoch, best_score if best_epoch > 0 else None)
        history_path = self.output_path.with_suffix(".history.json")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        result = {
            "status": "completed",
            "ended_at_utc": utc_now_iso(),
            "role": self.role,
            "dataset": self.args.dataset,
            "backbone": self.backbone,
            "output_path": str(self.output_path),
            "history_path": str(history_path),
            "best_epoch": best_epoch,
            "best_selection_score": best_score if best_epoch > 0 else None,
            "best_path": str(best_path) if best_epoch > 0 else None,
        }
        print("\nTraining complete.")
        print(f"{self.role.capitalize()} artifact: {self.output_path}")
        if best_epoch > 0:
            print(f"Best artifact: {best_path} (epoch={best_epoch}, score={best_score:.6f})")
        print(f"History JSON: {history_path}\n")
        return result

    def _save_artifact(
        self,
        path: Path,
        epoch: int,
        history: list[dict[str, Any]],
        best_epoch: int,
        best_score: float | None,
    ) -> None:
        assert self.dataset is not None
        assert self.model is not None
        assert self.optimizer is not None

        if self.role == "teacher":
            scorer = self._build_teacher_scorer()
            state = TeacherState(
                user_embeddings=self.model.get_all_user_embeddings().detach().cpu(),
                item_embeddings=self.model.get_all_item_embeddings().detach().cpu(),
                scorer=scorer,
                metadata={
                    "source": "recdistill_native_training",
                    "dataset": self.args.dataset,
                    "model_name": self.backbone,
                    "score_representation": "exact_scorer" if scorer is not None else "embedding_dot_product",
                    "embedding_dim": int(self.args.embedding_dim),
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_selection_score": best_score,
                    "config": self.run_config(),
                    "history": history,
                },
            )
            save_teacher_state(path, state, framework="recdistill", model_name=self.backbone)
            return

        save_student_checkpoint(
            path,
            {
                "epoch": int(epoch),
                "student_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": history,
                "config": self.run_config(),
                "num_users": int(self.dataset.num_users),
                "num_items": int(self.dataset.num_items),
                "best_epoch": int(best_epoch),
                "best_selection_score": best_score,
            },
        )

    def _build_teacher_scorer(self) -> PrecomputedScoresScorer | None:
        assert self.dataset is not None
        assert self.model is not None
        if self.backbone != "NMF" or not hasattr(self.model, "score_items_for_user"):
            return None

        self.model.eval()
        rows: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, int(self.dataset.num_users), int(self.args.eval_batch_size)):
                stop = min(start + int(self.args.eval_batch_size), int(self.dataset.num_users))
                batch_rows = [
                    self.model.score_items_for_user(user=user, num_items=int(self.dataset.num_items)).detach().cpu()
                    for user in range(start, stop)
                ]
                rows.append(torch.stack(batch_rows, dim=0))
        return PrecomputedScoresScorer(scores=torch.cat(rows, dim=0))

    def _default_output_path(self) -> Path:
        model = self.backbone.upper()
        dataset = str(self.args.dataset).lower()
        framework = str(self.args.framework).strip().lower()
        dim = int(self.args.embedding_dim)
        if self.role == "teacher":
            file_name = f"{framework}_{model}_{dataset}_{dim}{TEACHER_EXT}" if framework else f"{model}_{dataset}_{dim}{TEACHER_EXT}"
            return RESULTS_ROOT / dataset / "teacher" / framework / model / "best" / "wei" / file_name
        file_name = f"{framework}_{model}_{dataset}_{dim}{STUDENT_EXT}" if framework else f"{model}_{dataset}_{dim}{STUDENT_EXT}"
        return RESULTS_ROOT / dataset / "student" / framework / model / "best" / "wei" / file_name

    def _update_eval_row(self, row: dict[str, Any], current_eval: dict[str, Any]) -> None:
        row["val_precision"] = float(current_eval["val"]["precision"])
        row["val_recall"] = float(current_eval["val"]["recall"])
        row["val_ndcg"] = float(current_eval["val"]["ndcg"])
        row["val_hr"] = float(current_eval["val"]["hr"])
        row["leaked_users_val"] = int(current_eval["leaked_users_val"])
        if not self.args.eval_val_only:
            row["test_precision"] = float(current_eval["test"]["precision"])
            row["test_recall"] = float(current_eval["test"]["recall"])
            row["test_ndcg"] = float(current_eval["test"]["ndcg"])
            row["test_hr"] = float(current_eval["test"]["hr"])
            row["leaked_users_test"] = int(current_eval["leaked_users_test"])

    def _print_epoch(self, epoch: int, metrics: dict[str, float], current_eval: dict[str, Any] | None) -> None:
        print(
            f"Epoch {epoch:03d}/{int(self.args.epochs):03d} | "
            f"base={metrics['base_loss']:.6f} total={metrics['total_loss']:.6f}"
        )
        if current_eval is None:
            return
        print(
            f"  Val@{self.args.eval_k}: "
            f"P={current_eval['val']['precision']:.4f} "
            f"R={current_eval['val']['recall']:.4f} "
            f"NDCG={current_eval['val']['ndcg']:.4f} "
            f"HR={current_eval['val']['hr']:.4f} "
            f"| leaks={current_eval['leaked_users_val']}"
        )


def _normalize_role(role: str) -> str:
    normalized = str(role).strip().lower()
    if normalized not in {"teacher", "student"}:
        raise ValueError("role must be either 'teacher' or 'student'.")
    return normalized


def _with_role_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}")


def _stringify_mlp_dims(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return str(value)


def _parse_mlp_dims_for_config(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    return [int(value)]


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
