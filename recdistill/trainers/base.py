from __future__ import annotations

from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train_epoch(self) -> dict[str, float]:
        raise NotImplementedError
