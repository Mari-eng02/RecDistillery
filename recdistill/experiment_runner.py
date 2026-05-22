from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from config import RecDistillConfig
from recdistill.checkpointing import load_student_checkpoint, save_student_checkpoint
from recdistill.data.datarec_loader import load_eval_split, load_train_dataset, resolve_teacher_dataset_mappings
from recdistill.evaluation import evaluate_embeddings, evaluate_student
from recdistill.factories import build_distiller_from_args, build_student_model, normalize_backbone_name
from recdistill.paths import resolve_student_checkpoint_from_args, resolve_teacher_checkpoint_from_args
from recdistill.paths import DISTILLED_STUDENT_EXT
from recdistill.teachers import TeacherSource, inject_static_noise, load_teacher
from recdistill.model_validation import validate_loaded_teacher_for_distillation
from recdistill.tracking import utc_now_iso
from recdistill.training import build_lightgcn_graph, build_train_loader, prepare_distiller_trainable_modules
from recdistill.trainers import DistillationTrainer


class RecDistillExperimentRunner:
    def __init__(self, args: Any, wandb_logger=None):
        self.config = args if isinstance(args, RecDistillConfig) else None
        self.args = runner_args_from_config(args) if isinstance(args, RecDistillConfig) else args
        self.wandb_logger = wandb_logger
        args = self.args
        self.device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student_backbone = normalize_backbone_name(args.student_backbone)
        self.teacher_source = _teacher_source_from_args(args)
        self.teacher_path = self.teacher_source.path
        self.output_path = _distilled_student_path(
            resolve_student_checkpoint_from_args(args, distiller_name=self.resolve_distiller_name())
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        strategy_dir = self.output_path.parent.parent if self.output_path.parent.name == "wei" else self.output_path.parent
        (strategy_dir / "perf").mkdir(parents=True, exist_ok=True)

        self.teacher_state = None
        self.dataset = None
        self.val_dict: dict[int, set[int]] = {}
        self.test_dict: dict[int, set[int]] = {}
        self.model = None
        self.distiller = None
        self.optimizer = None
        self.trainer = None

    @classmethod
    def from_config(cls, config: RecDistillConfig, wandb_logger=None) -> "RecDistillExperimentRunner":
        return cls(config, wandb_logger=wandb_logger)

    def resolve_distiller_name(self) -> str:
        names = []
        if float(getattr(self.args, "lambda_de", 0.0)) > 0:
            names.append("DE")
        if float(getattr(self.args, "lambda_rrd", 0.0)) > 0:
            names.append("RRD")
        if float(getattr(self.args, "lambda_unkd", 0.0)) > 0:
            names.append("UnKD")
        if float(getattr(self.args, "lambda_td", 0.0)) > 0:
            names.append(str(getattr(self.args, "td_type", "TD")).upper())
        return "-".join(names) if names else "NONE"

    def run_config(self) -> dict[str, Any]:
        return {
            "config_source": "RecDistillConfig" if self.config is not None else "args",
            "dataset": self.args.dataset,
            "teacher_model": self.args.teacher_model,
            "teacher_path": str(self.teacher_path) if self.teacher_path is not None else None,
            "teacher_framework": getattr(self.args, "teacher_framework", "auto"),
            "teacher_format": getattr(self.args, "teacher_format", "auto"),
            "teacher_score_matrix_path": getattr(self.args, "teacher_score_matrix_path", None),
            "teacher_topk_items_path": getattr(self.args, "teacher_topk_items_path", None),
            "teacher_topk_scores_path": getattr(self.args, "teacher_topk_scores_path", None),
            "teacher_embedding_dim": self.args.teacher_embedding_dim,
            "teacher_noise_scale": self.args.teacher_noise_scale,
            "teacher_noise_target": self.args.teacher_noise_target,
            "teacher_noise_seed": self.args.teacher_noise_seed,
            "student_backbone": self.student_backbone,
            "student_framework": self.args.student_framework,
            "student_embedding_dim": self.args.student_embedding_dim,
            "lightgcn_layers": self.args.lightgcn_layers,
            "neumf_mlp_dims": self.args.neumf_mlp_dims,
            "neumf_dropout": self.args.neumf_dropout,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "learning_rate": self.args.learning_rate,
            "l2_reg": self.args.l2_reg,
            "lambda_de": self.args.lambda_de,
            "num_experts": self.args.num_experts,
            "temperature": self.args.temperature,
            "lambda_rrd": self.args.lambda_rrd,
            "rrd_interesting_size": self.args.rrd_interesting_size,
            "rrd_uninteresting_size": self.args.rrd_uninteresting_size,
            "rrd_temperature": self.args.rrd_temperature,
            "rrd_teacher_topk": self.args.rrd_teacher_topk,
            "lambda_unkd": self.args.lambda_unkd,
            "unkd_sample_num": self.args.unkd_sample_num,
            "unkd_group_count": self.args.unkd_group_count,
            "unkd_popularity_lambda": self.args.unkd_popularity_lambda,
            "unkd_rank_top_k": self.args.unkd_rank_top_k,
            "unkd_rank_temperature": self.args.unkd_rank_temperature,
            "lambda_td": self.args.lambda_td,
            "td_type": self.args.td_type,
            "htd_alpha": self.args.htd_alpha,
            "htd_num_groups": self.args.htd_num_groups,
            "htd_topology_mode": self.args.htd_topology_mode,
            "htd_initial_tau": self.args.htd_initial_tau,
            "htd_min_tau": self.args.htd_min_tau,
            "htd_decay_epochs": self.args.htd_decay_epochs,
            "td_entity_sample_size": self.args.td_entity_sample_size,
            "seed": self.args.seed,
            "eval_enabled": not self.args.skip_eval,
            "eval_k": self.args.eval_k,
            "eval_every": self.args.eval_every,
            "selection_split": self.args.selection_split,
            "selection_metric": self.args.selection_metric,
            "assert_no_train_leak": self.args.assert_no_train_leak,
        }

    def prepare(self) -> None:
        print("\n" + "=" * 80)
        print("Student Distillation Training")
        print("=" * 80)
        print(f"Dataset: {self.args.dataset}")
        print(f"Teacher source: {self.teacher_path or self.teacher_source.format}")
        print(f"Student backbone: {self.student_backbone}")
        print(f"Student framework: {self.args.student_framework}")
        print(f"Device: {self.device}")
        print("=" * 80 + "\n")

        teacher_state = load_teacher(self.teacher_source, device="cpu")
        validate_loaded_teacher_for_distillation(teacher_state, self.resolve_distiller_name())
        print(f"Teacher users/items: {teacher_state.num_users}/{teacher_state.num_items}")
        print(f"Teacher embedding dim: {teacher_state.embedding_dim if teacher_state.has_embeddings else 'none'}")
        print(f"Teacher exact scorer: {teacher_state.scorer is not None}")
        teacher_embedding_source = teacher_state.metadata.get("embedding_source")
        if teacher_embedding_source:
            print(f"Teacher embedding source: {teacher_embedding_source}")
        teacher_representation = teacher_state.metadata.get("representation")
        if teacher_representation:
            print(f"Teacher embedding representation: {teacher_representation}")

        if float(self.args.teacher_noise_scale) > 0.0:
            if not teacher_state.has_embeddings:
                raise ValueError("Teacher noise injection requires an embedding-based teacher.")
            teacher_state, noise_info = inject_static_noise(
                teacher_state=teacher_state,
                noise_scale=float(self.args.teacher_noise_scale),
                target=str(self.args.teacher_noise_target),
                seed=self.args.teacher_noise_seed,
            )
            print(
                "Teacher noise injection: "
                f"base_std={noise_info['base_std']:.6f} "
                f"alpha={noise_info['noise_scale']} "
                f"noise_std={noise_info['scaled_noise_std']:.6f} "
                f"target={noise_info['noise_target']}"
            )

        self.teacher_state = teacher_state
        user_mapping, item_mapping, mapping_source = resolve_teacher_dataset_mappings(
            teacher_state.metadata,
            dataset_name=self.args.dataset,
        )
        print(f"Dataset mapping source: {mapping_source}")
        self.dataset, dropped = load_train_dataset(
            dataset_name=self.args.dataset,
            teacher_num_users=teacher_state.num_users,
            teacher_num_items=teacher_state.num_items,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        print(f"Train interactions: {len(self.dataset.interactions)}")
        print(f"Dropped interactions (out of teacher range): {dropped}")

        if not self.args.skip_eval:
            self.val_dict, dropped_val = load_eval_split(
                dataset_name=self.args.dataset,
                split_name="val",
                teacher_num_users=teacher_state.num_users,
                teacher_num_items=teacher_state.num_items,
                user_mapping=user_mapping,
                item_mapping=item_mapping,
            )
            self.test_dict, dropped_test = load_eval_split(
                dataset_name=self.args.dataset,
                split_name="test",
                teacher_num_users=teacher_state.num_users,
                teacher_num_items=teacher_state.num_items,
                user_mapping=user_mapping,
                item_mapping=item_mapping,
            )
            print(f"Validation interactions: {sum(len(v) for v in self.val_dict.values())} (dropped: {dropped_val})")
            print(f"Test interactions: {sum(len(v) for v in self.test_dict.values())} (dropped: {dropped_test})")

        train_loader = build_train_loader(self.dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        self.model = build_student_model(
            backbone=self.student_backbone,
            dataset=self.dataset,
            embedding_dim=self.args.student_embedding_dim,
            l2_reg=self.args.l2_reg,
            lightgcn_layers=self.args.lightgcn_layers,
            neumf_mlp_dims=self.args.neumf_mlp_dims,
            neumf_dropout=self.args.neumf_dropout,
            framework=self.args.student_framework,
            graph_builder=build_lightgcn_graph,
        ).to(self.device)
        self.distiller = build_distiller_from_args(
            args=self.args,
            teacher_state=teacher_state,
            student_dim=self.args.student_embedding_dim,
        )
        teacher_state_for_distiller = teacher_state.to(self.device)
        if self.distiller is not None:
            self.distiller = self.distiller.to(self.device)
            self.distiller.on_train_start(teacher_state_for_distiller, self.dataset)
            prepare_distiller_trainable_modules(self.distiller, int(self.args.student_embedding_dim), self.device)
            setattr(self.distiller, "_recdistill_initialized", True)

        trainable_params = list(self.model.parameters())
        if self.distiller is not None:
            trainable_params += list(self.distiller.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)
        self.trainer = DistillationTrainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=train_loader,
            distiller=self.distiller,
            device=self.device,
            teacher_state=teacher_state,
            dataset=self.dataset,
        )

        if not self.args.skip_eval:
            scorer_note = "exact scorer" if teacher_state.scorer is not None else "embedding dot product"
            teacher_eval = evaluate_embeddings(
                user_embeddings=teacher_state.user_embeddings,
                item_embeddings=teacher_state.item_embeddings,
                train_seen=self.dataset.train_dict,
                ground_truth=self.val_dict,
                top_k=self.args.eval_k,
                batch_size=self.args.eval_batch_size,
                device=self.device,
                scorer=teacher_state.scorer,
            )[0]
            print(
                f"Teacher baseline @ {self.args.eval_k} (val, {scorer_note}): "
                f"P={teacher_eval['precision']:.4f} "
                f"R={teacher_eval['recall']:.4f} "
                f"NDCG={teacher_eval['ndcg']:.4f} "
                f"HR={teacher_eval['hr']:.4f}"
            )

    def run(self) -> dict[str, Any]:
        self.prepare()
        return self.train()

    def train(self) -> dict[str, Any]:
        args = self.args
        assert self.teacher_state is not None
        assert self.dataset is not None
        assert self.model is not None
        assert self.optimizer is not None
        assert self.trainer is not None

        start_payload = {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            **self.run_config(),
            "teacher_embedding_dim": int(self.teacher_state.embedding_dim) if self.teacher_state.has_embeddings else None,
        }
        if self.wandb_logger is not None:
            self.wandb_logger.log_start(start_payload)

        history: list[dict[str, float | int]] = []
        best_score = float("-inf")
        best_epoch = 0
        best_checkpoint = self.output_path
        saved_best_checkpoint = False
        early_best_value: float | None = None
        early_best_epoch = 0
        early_bad_steps = 0
        early_stopped = False
        early_stop_reason: str | None = None
        early_monitor_name = "total_loss" if args.early_stop_mode == "loss" else f"val_{args.early_stop_metric}"
        early_best_checkpoint = self.output_path.with_name(f"{self.output_path.stem}.earlystop_best{DISTILLED_STUDENT_EXT}")

        run_status = "completed"
        run_error: str | None = None
        caught_exception: Exception | None = None
        final_test_eval: dict[str, Any] | None = None
        try:
            for epoch in range(1, args.epochs + 1):
                metrics = self.trainer.train_epoch()
                row = {"epoch": epoch, **metrics}
                current_eval: dict[str, dict[str, float] | int] | None = None
                if not args.skip_eval and args.eval_every > 0 and (epoch % args.eval_every == 0):
                    current_eval = evaluate_student(
                        model=self.model,
                        train_seen=self.dataset.train_dict,
                        val_gt=self.val_dict,
                        test_gt=self.test_dict,
                        top_k=args.eval_k,
                        batch_size=args.eval_batch_size,
                        device=self.device,
                        eval_val_only=args.eval_val_only,
                    )
                    self._update_eval_row(row, current_eval)
                    leaked_users_test = int(current_eval.get("leaked_users_test", 0))
                    if args.assert_no_train_leak and (current_eval["leaked_users_val"] > 0 or leaked_users_test > 0):
                        raise RuntimeError(
                            "Train-item leakage detected in recommendations. "
                            f"val_leaks={current_eval['leaked_users_val']} test_leaks={leaked_users_test}"
                        )

                    selected_split_metrics = current_eval[args.selection_split]
                    selected_score = float(selected_split_metrics[args.selection_metric])
                    row["selection_score"] = selected_score
                    if selected_score > best_score:
                        best_score = selected_score
                        best_epoch = epoch
                        self._save_checkpoint(best_checkpoint, epoch, history + [row], best_epoch, best_score)
                        saved_best_checkpoint = True

                history.append(row)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_epoch(row)
                self._print_epoch(epoch, metrics, current_eval)

                if args.save_every > 0 and (epoch % args.save_every == 0):
                    periodic_path = self.output_path.with_name(f"{self.output_path.stem}.ep{epoch}{DISTILLED_STUDENT_EXT}")
                    self._save_checkpoint(periodic_path, epoch, history, best_epoch, best_score if best_epoch > 0 else None)
                    print(f"Saved periodic checkpoint: {periodic_path}")

                early_state = self._maybe_early_stop(
                    epoch=epoch,
                    row=row,
                    current_eval=current_eval,
                    early_best_value=early_best_value,
                    early_best_epoch=early_best_epoch,
                    early_bad_steps=early_bad_steps,
                    early_monitor_name=early_monitor_name,
                    early_best_checkpoint=early_best_checkpoint,
                )
                early_best_value = early_state["best_value"]
                early_best_epoch = early_state["best_epoch"]
                early_bad_steps = early_state["bad_steps"]
                if early_state["stopped"]:
                    early_stopped = True
                    early_stop_reason = early_state["reason"]
                    print(f"Early stopping triggered at epoch {epoch}: {early_stop_reason}")
                    break
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            caught_exception = exc

        if (
            caught_exception is None
            and args.early_stop
            and args.early_stop_restore_best
            and early_best_epoch > 0
            and early_best_checkpoint.exists()
        ):
            payload = load_student_checkpoint(early_best_checkpoint, map_location=self.device)
            self.model.load_state_dict(payload["student_state_dict"])
            print(
                f"Restored early-stop best checkpoint from epoch {payload['epoch']} "
                f"({payload['monitor_name']}={payload['monitor_value']:.6f})"
            )

        if caught_exception is None and not args.skip_eval and args.eval_val_only and len(self.test_dict) > 0:
            final_test_eval = evaluate_student(
                model=self.model,
                train_seen=self.dataset.train_dict,
                val_gt=self.val_dict,
                test_gt=self.test_dict,
                top_k=args.eval_k,
                batch_size=args.eval_batch_size,
                device=self.device,
                eval_val_only=False,
            )
            print(
                f"Final Test@{args.eval_k}: "
                f"P={final_test_eval['test']['precision']:.4f} "
                f"R={final_test_eval['test']['recall']:.4f} "
                f"NDCG={final_test_eval['test']['ndcg']:.4f} "
                f"HR={final_test_eval['test']['hr']:.4f} "
                f"| leaks={final_test_eval['leaked_users_test']}"
            )

        history_path = self.output_path.with_suffix(".history.json")
        should_save_final = caught_exception is None and (
            not saved_best_checkpoint or (args.early_stop and args.early_stop_restore_best and early_best_epoch > 0)
        )
        if should_save_final:
            final_epoch = int(history[-1]["epoch"]) if history else 0
            self._save_checkpoint(
                self.output_path,
                final_epoch,
                history,
                best_epoch,
                best_score if best_epoch > 0 else None,
                extra={
                    "early_stopped": early_stopped,
                    "early_stop_reason": early_stop_reason,
                    "early_best_epoch": early_best_epoch if early_best_epoch > 0 else None,
                    "early_best_value": early_best_value,
                    "early_monitor_name": early_monitor_name if args.early_stop else None,
                    "final_test_eval": final_test_eval,
                },
            )
        if caught_exception is None:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        end_payload = {
            "status": run_status,
            "ended_at_utc": utc_now_iso(),
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score) if best_epoch > 0 else None,
            "best_checkpoint": str(best_checkpoint) if best_epoch > 0 else None,
            "final_checkpoint": str(self.output_path) if caught_exception is None else None,
            "history_file": str(history_path) if history_path.exists() else None,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "early_best_epoch": early_best_epoch if early_best_epoch > 0 else None,
            "early_best_value": early_best_value,
            "early_monitor_name": early_monitor_name if args.early_stop else None,
            "final_test_eval": final_test_eval,
            "error": run_error,
        }
        if self.wandb_logger is not None:
            self.wandb_logger.log_end(end_payload)
        if caught_exception is not None:
            raise caught_exception

        print("\nTraining complete.")
        print(f"Student checkpoint: {self.output_path}")
        if best_epoch > 0:
            print(
                f"Best checkpoint ({args.selection_split}.{args.selection_metric}): "
                f"epoch={best_epoch} score={best_score:.6f} path={best_checkpoint}"
            )
        if final_test_eval is not None:
            print(
                f"Final test metrics: "
                f"NDCG={final_test_eval['test']['ndcg']:.4f} "
                f"HR={final_test_eval['test']['hr']:.4f}"
            )
        print(f"History JSON: {history_path}\n")
        return end_payload

    def _checkpoint_payload(
        self,
        epoch: int,
        history: list[dict[str, Any]],
        best_epoch: int,
        best_score: float | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "epoch": epoch,
            "student_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history,
            "config": vars(self.args),
            "teacher_path": str(self.teacher_path) if self.teacher_path is not None else None,
            "teacher_dim": self.teacher_state.embedding_dim if self.teacher_state.has_embeddings else None,
            "num_users": self.dataset.num_users,
            "num_items": self.dataset.num_items,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "best_selection_split": self.args.selection_split,
            "best_selection_metric": self.args.selection_metric,
        }
        if extra:
            payload.update(extra)
        return payload

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        history: list[dict[str, Any]],
        best_epoch: int,
        best_score: float | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        save_student_checkpoint(path, self._checkpoint_payload(epoch, history, best_epoch, best_score, extra))

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
            f"Epoch {epoch:03d}/{self.args.epochs:03d} | "
            f"base={metrics['base_loss']:.6f} "
            f"distill={metrics['distill_loss']:.6f} "
            f"total={metrics['total_loss']:.6f}"
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
        if not self.args.eval_val_only:
            print(
                f"  Test@{self.args.eval_k}: "
                f"P={current_eval['test']['precision']:.4f} "
                f"R={current_eval['test']['recall']:.4f} "
                f"NDCG={current_eval['test']['ndcg']:.4f} "
                f"HR={current_eval['test']['hr']:.4f} "
                f"| leaks={current_eval['leaked_users_test']}"
            )

    def _maybe_early_stop(
        self,
        *,
        epoch: int,
        row: dict[str, Any],
        current_eval: dict[str, Any] | None,
        early_best_value: float | None,
        early_best_epoch: int,
        early_bad_steps: int,
        early_monitor_name: str,
        early_best_checkpoint: Path,
    ) -> dict[str, Any]:
        if not self.args.early_stop:
            return {
                "best_value": early_best_value,
                "best_epoch": early_best_epoch,
                "bad_steps": early_bad_steps,
                "stopped": False,
                "reason": None,
            }

        current_monitor_value: float | None = None
        if self.args.early_stop_mode == "loss":
            current_monitor_value = float(row["total_loss"])
        elif current_eval is not None:
            current_monitor_value = float(current_eval["val"][self.args.early_stop_metric])

        if current_monitor_value is None:
            return {
                "best_value": early_best_value,
                "best_epoch": early_best_epoch,
                "bad_steps": early_bad_steps,
                "stopped": False,
                "reason": None,
            }

        improved = False
        if early_best_value is None:
            improved = True
        elif self.args.early_stop_mode == "loss":
            improved = current_monitor_value < (early_best_value - self.args.early_stop_min_delta)
        else:
            improved = current_monitor_value > (early_best_value + self.args.early_stop_min_delta)

        if improved:
            early_best_value = current_monitor_value
            early_best_epoch = epoch
            early_bad_steps = 0
            save_student_checkpoint(
                early_best_checkpoint,
                {
                    "epoch": epoch,
                    "monitor_name": early_monitor_name,
                    "monitor_value": current_monitor_value,
                    "student_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": vars(self.args),
                },
            )
        else:
            early_bad_steps += 1

        stopped = False
        reason = None
        if epoch >= self.args.early_stop_warmup and early_bad_steps >= self.args.early_stop_patience:
            stopped = True
            reason = (
                f"no improvement on {early_monitor_name} for {early_bad_steps} step(s); "
                f"best={early_best_value:.6f} at epoch={early_best_epoch}"
            )
        return {
            "best_value": early_best_value,
            "best_epoch": early_best_epoch,
            "bad_steps": early_bad_steps,
            "stopped": stopped,
            "reason": reason,
        }


def runner_args_from_config(config: RecDistillConfig) -> SimpleNamespace:
    train = config.train_student
    teacher = train.teacher
    student = train.student
    distillation = train.distillation
    optimization = train.optimization
    runtime = train.runtime
    evaluation = train.evaluation
    early = train.early_stopping

    topology = _dict_section(distillation, "topology")
    rrd = _dict_section(distillation, "rrd")
    unkd = _dict_section(distillation, "unkd")
    teacher_noise = _dict_section(teacher, "noise")
    wandb = runtime.wandb if isinstance(runtime.wandb, dict) else {}

    return SimpleNamespace(
        dataset=train.dataset,
        teacher_model=teacher.model,
        teacher_embedding_dim=teacher.embedding_dim,
        teacher_path=teacher.path,
        teacher_framework=getattr(teacher, "framework", "auto"),
        teacher_format=getattr(teacher, "format", "auto"),
        teacher_adapter=getattr(teacher, "adapter", None),
        teacher_user_embeddings_path=getattr(teacher, "user_embeddings_path", None),
        teacher_item_embeddings_path=getattr(teacher, "item_embeddings_path", None),
        teacher_score_matrix_path=getattr(teacher, "score_matrix_path", None),
        teacher_topk_items_path=getattr(teacher, "topk_items_path", None),
        teacher_topk_scores_path=getattr(teacher, "topk_scores_path", None),
        teacher_noise_scale=teacher_noise.get("scale", getattr(teacher, "noise_scale", 0.0) or 0.0),
        teacher_noise_target=teacher_noise.get("target", getattr(teacher, "noise_target", "both") or "both"),
        teacher_noise_seed=teacher_noise.get("seed", getattr(teacher, "noise_seed", None)),
        student_backbone=student.backbone,
        student_framework=getattr(student, "framework", "recbole"),
        student_embedding_dim=student.embedding_dim,
        lightgcn_layers=getattr(student, "num_layers", getattr(student, "lightgcn_layers", 2)),
        neumf_mlp_dims=getattr(student, "mlp_hidden_size", getattr(student, "mlp_dims", "64,32,16,8")),
        neumf_dropout=getattr(student, "dropout", 0.0),
        epochs=optimization.epochs,
        batch_size=optimization.batch_size,
        learning_rate=optimization.learning_rate,
        l2_reg=optimization.l2_reg,
        lambda_de=getattr(distillation, "lambda_de", 0.0),
        num_experts=getattr(distillation, "num_experts", getattr(student, "num_experts", 10)),
        temperature=getattr(distillation, "temperature", getattr(student, "temperature", 1.0)),
        lambda_rrd=getattr(distillation, "lambda_rrd", 0.0),
        rrd_interesting_size=rrd.get("interesting_size", getattr(student, "rrd_interesting_size", 10)),
        rrd_uninteresting_size=rrd.get("uninteresting_size", getattr(student, "rrd_uninteresting_size", 50)),
        rrd_temperature=rrd.get("temperature", getattr(student, "rrd_temperature", 1.0)),
        rrd_teacher_topk=rrd.get("teacher_topk", getattr(student, "rrd_teacher_topk", 500)),
        lambda_unkd=getattr(distillation, "lambda_unkd", 0.0),
        unkd_sample_num=unkd.get("sample_num", getattr(student, "unkd_sample_num", 30)),
        unkd_group_count=unkd.get("group_count", getattr(student, "unkd_group_count", 2)),
        unkd_popularity_lambda=unkd.get("popularity_lambda", getattr(student, "unkd_popularity_lambda", 1.0)),
        unkd_rank_top_k=unkd.get("rank_top_k", getattr(student, "unkd_rank_top_k", 1000)),
        unkd_rank_temperature=unkd.get("rank_temperature", getattr(student, "unkd_rank_temperature", 20.0)),
        lambda_td=topology.get("lambda_td", getattr(distillation, "lambda_td", 0.0)),
        td_type=str(topology.get("type", getattr(distillation, "strategy", "HTD"))).upper(),
        td_entity_sample_size=topology.get("entity_sample_size", getattr(student, "td_entity_sample_size", 0)),
        htd_alpha=topology.get("alpha", getattr(student, "htd_alpha", 0.5)),
        htd_num_groups=topology.get("num_groups", getattr(student, "htd_num_groups", 40)),
        htd_topology_mode=topology.get("topology_mode", getattr(student, "htd_topology_mode", "group_pe")),
        htd_initial_tau=topology.get("initial_tau", getattr(student, "htd_initial_tau", 1.0)),
        htd_min_tau=topology.get("min_tau", getattr(student, "htd_min_tau", 1e-10)),
        htd_decay_epochs=topology.get("decay_epochs", getattr(student, "htd_decay_epochs", 100)),
        seed=runtime.seed,
        device=runtime.device,
        num_workers=runtime.num_workers,
        output_path=runtime.output_path,
        output_strategy=getattr(runtime, "output_strategy", "fixed"),
        save_every=runtime.save_every,
        skip_eval=not evaluation.enabled,
        eval_k=evaluation.k,
        eval_every=evaluation.every,
        eval_batch_size=evaluation.batch_size,
        eval_val_only=evaluation.val_only,
        selection_split=evaluation.selection_split,
        selection_metric=evaluation.selection_metric,
        assert_no_train_leak=evaluation.assert_no_train_leak,
        early_stop=bool(early.enabled) if early is not None else False,
        early_stop_mode=early.mode if early is not None else "loss",
        early_stop_metric=early.metric if early is not None else "ndcg",
        early_stop_patience=early.patience if early is not None else 5,
        early_stop_min_delta=early.min_delta if early is not None else 0.0,
        early_stop_warmup=getattr(early, "warmup", 0) if early is not None else 0,
        early_stop_restore_best=getattr(early, "restore_best", False) if early is not None else False,
        wandb_log=bool(wandb.get("enabled", False)),
        wandb_project=wandb.get("project"),
        wandb_entity=wandb.get("entity"),
        wandb_run_name=wandb.get("run_name"),
        wandb_tags=",".join(str(tag) for tag in wandb.get("tags", [])) if isinstance(wandb.get("tags"), list) else wandb.get("tags"),
        wandb_group=wandb.get("group"),
        wandb_notes=wandb.get("notes"),
    )


def _dict_section(config: Any, name: str) -> dict[str, Any]:
    value = getattr(config, name, None)
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _teacher_source_from_args(args: Any) -> TeacherSource:
    user_embeddings_path = getattr(args, "teacher_user_embeddings_path", None)
    item_embeddings_path = getattr(args, "teacher_item_embeddings_path", None)
    score_matrix_path = getattr(args, "teacher_score_matrix_path", None)
    topk_items_path = getattr(args, "teacher_topk_items_path", None)
    topk_scores_path = getattr(args, "teacher_topk_scores_path", None)
    teacher_path = getattr(args, "teacher_path", None)
    if teacher_path is None and not (user_embeddings_path and item_embeddings_path) and not score_matrix_path and not topk_items_path:
        teacher_path = resolve_teacher_checkpoint_from_args(args)
    return TeacherSource(
        path=Path(teacher_path) if teacher_path is not None else None,
        framework=getattr(args, "teacher_framework", "auto") or "auto",
        format=getattr(args, "teacher_format", "auto") or "auto",
        model_name=getattr(args, "teacher_model", None),
        adapter=getattr(args, "teacher_adapter", None),
        user_embeddings_path=Path(user_embeddings_path) if user_embeddings_path is not None else None,
        item_embeddings_path=Path(item_embeddings_path) if item_embeddings_path is not None else None,
        score_matrix_path=Path(score_matrix_path) if score_matrix_path is not None else None,
        topk_items_path=Path(topk_items_path) if topk_items_path is not None else None,
        topk_scores_path=Path(topk_scores_path) if topk_scores_path is not None else None,
        metadata={},
    )


def _distilled_student_path(path: Path) -> Path:
    return path if path.suffix == DISTILLED_STUDENT_EXT else path.with_suffix(DISTILLED_STUDENT_EXT)
