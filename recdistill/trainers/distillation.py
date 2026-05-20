from __future__ import annotations

from dataclasses import dataclass

import torch

from config import RecDistillConfig
from recdistill.data.batch import InteractionBatch
from recdistill.data.interactions import InteractionDataset
from recdistill.distillers.base import Distiller
from recdistill.teachers.state import TeacherState
from recdistill.trainers.base import Trainer
from recdistill.config_integration import load_recdistill_experiment


@dataclass
class TrainerMetrics:
    base_loss: float
    distill_loss: float
    total_loss: float


class DistillationTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        distiller: Distiller | None = None,
        device: torch.device | None = None,
        teacher_state: TeacherState | None = None,
        dataset: InteractionDataset | None = None,
        config: RecDistillConfig | None = None,
    ):
        self.config = config
        self.train_config = config.train_student if config is not None else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.distiller = distiller.to(self.device) if distiller is not None else None
        self.teacher_state = teacher_state.to(self.device) if teacher_state is not None else None
        self.dataset = dataset

        if (
            self.distiller is not None
            and self.teacher_state is not None
            and self.dataset is not None
            and not getattr(self.distiller, "_recdistill_initialized", False)
        ):
            self.distiller.on_train_start(self.teacher_state, self.dataset)
            setattr(self.distiller, "_recdistill_initialized", True)

    @classmethod
    def load_config(
        cls,
        dataset_name: str,
        teacher_model: str,
        distiller_strategy: str,
        student_backbone: str | None = None,
        overrides: dict | None = None,
    ) -> RecDistillConfig:
        """Load the validated RecDistill config used by trainer callers."""
        return load_recdistill_experiment(
            dataset_name=dataset_name,
            teacher_model=teacher_model,
            distiller_strategy=distiller_strategy,
            student_backbone=student_backbone,
            overrides=overrides,
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        if self.distiller is not None:
            self.distiller.on_epoch_start()

        total_base_loss = 0.0
        total_distill_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for users, pos_items, neg_items in self.train_loader:
            batch = InteractionBatch(
                users=users.to(self.device),
                pos_items=pos_items.to(self.device),
                neg_items=neg_items.to(self.device),
            )

            batch_output = self.model(batch.users, batch.pos_items, batch.neg_items)
            base_loss = self.model.compute_base_loss(batch_output)

            distill_loss = torch.zeros((), device=self.device)
            if self.distiller is not None:
                aux_batch = self.distiller.build_aux_batch(batch, device=self.device)
                distill_loss = self.distiller.compute_loss(self.model, batch, aux_batch)

            loss = base_loss + distill_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_base_loss += float(base_loss.detach().cpu())
            total_distill_loss += float(distill_loss.detach().cpu())
            total_loss += float(loss.detach().cpu())
            num_batches += 1

        if num_batches == 0:
            return TrainerMetrics(0.0, 0.0, 0.0).__dict__

        return TrainerMetrics(
            base_loss=total_base_loss / num_batches,
            distill_loss=total_distill_loss / num_batches,
            total_loss=total_loss / num_batches,
        ).__dict__
