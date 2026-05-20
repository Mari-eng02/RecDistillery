from __future__ import annotations

import torch

from recdistill.data.batch import InteractionBatch
from recdistill.data.interactions import InteractionDataset
from recdistill.distillers.base import Distiller
from recdistill.teachers.state import TeacherState


class CompositeDistiller(Distiller):
    def __init__(self, distillers: list[Distiller]):
        super().__init__()
        self.distillers = torch.nn.ModuleList(distillers)

    @staticmethod
    def _distiller_key(distiller: Distiller, index: int) -> str:
        return f"{distiller.__class__.__name__}:{index}"

    def on_train_start(self, teacher_state: TeacherState, dataset: InteractionDataset) -> None:
        for distiller in self.distillers:
            distiller.on_train_start(teacher_state, dataset)

    def on_epoch_start(self) -> None:
        for distiller in self.distillers:
            distiller.on_epoch_start()

    def build_aux_batch(self, batch: InteractionBatch, device: torch.device) -> dict[str, object]:
        aux_batches: dict[str, object] = {}
        for index, distiller in enumerate(self.distillers):
            aux = distiller.build_aux_batch(batch, device)
            if aux is not None:
                aux_batches[self._distiller_key(distiller, index)] = aux
        return aux_batches

    def compute_loss(
        self,
        student: torch.nn.Module,
        batch: InteractionBatch,
        aux_batch: dict[str, object] | None = None,
    ) -> torch.Tensor:
        if not self.distillers:
            return torch.zeros((), device=batch.users.device)

        total_loss = torch.zeros((), device=batch.users.device)
        aux_batch = aux_batch or {}
        for index, distiller in enumerate(self.distillers):
            total_loss = total_loss + distiller.compute_loss(
                student,
                batch,
                aux_batch.get(self._distiller_key(distiller, index)),
            )
        return total_loss
