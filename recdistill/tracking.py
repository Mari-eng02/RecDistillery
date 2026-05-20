from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WandBRunLogger:
    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        notes: str | None = None,
        config: dict | None = None,
    ):
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb is required for W&B logging. Install it with `pip install wandb`."
            ) from exc

        self._wandb = wandb
        self.run = self._wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            tags=tags,
            group=group,
            notes=notes,
            config=config or {},
        )

    def log_start(self, payload: dict) -> None:
        if self.run is None:
            return
        self.run.summary["status"] = payload.get("status", "running")
        self.run.summary["started_at_utc"] = payload.get("started_at_utc", utc_now_iso())
        for key, value in payload.items():
            if key in {"status", "started_at_utc"}:
                continue
            self.run.config[key] = value

    def log_epoch(self, epoch_payload: dict) -> None:
        if self.run is None:
            return
        self._wandb.log(epoch_payload, step=int(epoch_payload.get("epoch", 0)))

    def log_end(self, payload: dict) -> None:
        if self.run is None:
            return
        self.run.summary["status"] = payload.get("status", "completed")
        self.run.summary["ended_at_utc"] = payload.get("ended_at_utc", utc_now_iso())
        for key, value in payload.items():
            if key in {"status", "ended_at_utc"}:
                continue
            self.run.summary[key] = value
        exit_code = 1 if payload.get("status") == "failed" else 0
        self._wandb.finish(exit_code=exit_code)


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def resolve_wandb_logger(args: Any, base_config: dict) -> WandBRunLogger | None:
    if not args.wandb_log:
        return None
    if not args.wandb_project:
        raise ValueError("--wandb-project is required when --wandb-log is enabled.")

    return WandBRunLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        tags=parse_csv_list(args.wandb_tags),
        group=args.wandb_group,
        notes=args.wandb_notes,
        config=base_config,
    )
