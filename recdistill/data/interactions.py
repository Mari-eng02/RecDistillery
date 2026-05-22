from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InteractionDataset:
    num_users: int
    num_items: int
    train_dict: dict[int, set[int]]
    interactions: list[tuple[int, int]]

    @classmethod
    def from_train_dict(
        cls,
        train_dict: dict[int, set[int]] | dict[int, list[int]],
        num_users: int,
        num_items: int,
    ) -> "InteractionDataset":
        normalized = {int(u): set(map(int, items)) for u, items in train_dict.items()}
        interactions = [
            (user, item) for user, items in normalized.items() for item in sorted(items)
        ]
        return cls(
            num_users=num_users,
            num_items=num_items,
            train_dict=normalized,
            interactions=interactions,
        )

    def seen_items(self, user: int) -> set[int]:
        return self.train_dict.get(int(user), set())
