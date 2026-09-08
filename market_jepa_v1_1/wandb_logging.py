from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable


BYTES_PER_GIB = 1024 ** 3


class V11WandbLogger:
    """Failure-isolated W&B observability for formal V1.1 training.

    This class is deliberately the only V1.1 module that imports the W&B SDK.
    Local checkpoints and metric files remain authoritative; every SDK call is
    guarded so an observability failure cannot stop model training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output: str | Path,
        *,
        backend: Any | None = None,
        warning: Callable[[str], None] | None = None,
        info: Callable[[str], None] | None = None,
    ) -> None:
        self.config = dict(config)
        self.output = Path(output)
        self._backend = backend
        self._warning = warning or (lambda message: print(f"WARNING: {message}", flush=True))
        self._info = info or (lambda message: print(message, flush=True))
        self.run: Any | None = None
        self.run_id: str | None = None
        self.disabled = not bool(self.config.get("enabled", False)) or self.config.get("mode") == "disabled"
        self._interval = int(self.config.get("log_every_optimizer_steps", 50))
        self._step_buffer: list[dict[str, float | int | None]] = []

    @property
    def active(self) -> bool:
        return not self.disabled and self.run is not None

    def _warn(self, operation: str, error: BaseException) -> None:
        try:
            self._warning(
                f"W&B {operation} failed; W&B is disabled for the remaining run: "
                f"{type(error).__name__}: {error}"
            )
        except Exception:
            pass

    def _warn_persistence(self, error: BaseException) -> None:
        try:
            self._warning(
                "W&B run-id persistence failed; current logging continues, but a later resume "
                f"may need a new W&B run: {type(error).__name__}: {error}"
            )
        except Exception:
            pass

    def _disable_after(self, operation: str, error: BaseException) -> None:
        self.disabled = True
        self._step_buffer.clear()
        self._warn(operation, error)

    def _persist_run(self, run_name: str) -> None:
        value = {
            "run_id": self.run_id,
            "project": self.config["project"],
            "group": self.config["group"],
            "run_name": run_name,
        }
        url = getattr(self.run, "url", None)
        if url:
            value["url"] = str(url)
        path = self.output / "wandb_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def init(
        self,
        metadata: dict[str, Any],
        *,
        run_name: str,
        tags: list[str],
        resume_run_id: str | None = None,
        resumed: bool = False,
    ) -> str | None:
        if self.disabled:
            return None
        try:
            backend = self._backend or importlib.import_module("wandb")
            run_tags = list(tags)
            if resumed and not resume_run_id:
                run_tags.append("resumed-new-wandb-run")
            init_kwargs: dict[str, Any] = {
                "project": self.config["project"],
                "group": self.config["group"],
                "entity": self.config.get("entity"),
                "name": run_name,
                "mode": self.config["mode"],
                "dir": str(self.output.resolve()),
                "config": metadata,
                "tags": run_tags,
            }
            if resume_run_id:
                init_kwargs.update(id=resume_run_id, resume="allow")
            self.run = backend.init(**init_kwargs)
            self.run_id = str(getattr(self.run, "id", resume_run_id or "")) or None
            self.run.define_metric("global_step")
            self.run.define_metric("samples_seen")
            for namespace in ("train/*", "optim/*", "perf/*"):
                self.run.define_metric(namespace, step_metric="global_step")
            self.run.define_metric("epoch")
            self.run.define_metric("epoch/*", step_metric="epoch")
            try:
                self._persist_run(run_name)
            except Exception as error:
                # Persistence failure affects later W&B continuity, not this run or training.
                self._warn_persistence(error)
            url = getattr(self.run, "url", None) or "unavailable"
            self._info(
                f"W&B project={self.config['project']} run={run_name} "
                f"id={self.run_id or 'unavailable'} url={url}"
            )
            return self.run_id
        except Exception as error:
            self.run = None
            self.run_id = None
            self._disable_after("initialization", error)
            return None

    def log_step(
        self,
        *,
        global_step: int,
        samples_seen: int | None = None,
        loss: float,
        h16_loss: float,
        h64_loss: float,
        h256_loss: float,
        learning_rate: float,
        grad_norm: float | None,
        skipped_optimizer_steps: int,
        step_seconds: float,
        samples: int,
        allocated_vram_bytes: int | None = None,
        reserved_vram_bytes: int | None = None,
    ) -> None:
        if not self.active:
            return
        self._step_buffer.append({
            "global_step": int(global_step),
            "samples_seen": int(samples_seen) if samples_seen is not None else int(global_step) * int(samples),
            "loss": float(loss),
            "h16_loss": float(h16_loss),
            "h64_loss": float(h64_loss),
            "h256_loss": float(h256_loss),
            "learning_rate": float(learning_rate),
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "step_seconds": float(step_seconds),
            "samples": int(samples),
            "allocated_vram_bytes": allocated_vram_bytes,
            "reserved_vram_bytes": reserved_vram_bytes,
        })
        # Align logs to the formal optimizer counter, including after process resume.
        if int(global_step) % self._interval == 0:
            self._flush_steps()

    def _flush_steps(self) -> None:
        if not self.active or not self._step_buffer:
            return
        try:
            values = self._step_buffer
            count = len(values)
            total_seconds = sum(float(value["step_seconds"]) for value in values)
            total_samples = sum(int(value["samples"]) for value in values)
            grad_norms = [float(value["grad_norm"]) for value in values if value["grad_norm"] is not None]
            payload: dict[str, float | int] = {
                "global_step": int(values[-1]["global_step"]),
                "samples_seen": int(values[-1]["samples_seen"]),
                "train/loss": sum(float(value["loss"]) for value in values) / count,
                "train/h16_loss": sum(float(value["h16_loss"]) for value in values) / count,
                "train/h64_loss": sum(float(value["h64_loss"]) for value in values) / count,
                "train/h256_loss": sum(float(value["h256_loss"]) for value in values) / count,
                "optim/learning_rate": float(values[-1]["learning_rate"]),
                "optim/skipped_optimizer_steps": int(values[-1]["skipped_optimizer_steps"]),
                "perf/step_seconds": total_seconds / count,
                "perf/samples_per_sec": total_samples / total_seconds if total_seconds > 0 else 0.0,
            }
            if grad_norms:
                payload["optim/grad_norm"] = sum(grad_norms) / len(grad_norms)
            allocated = [int(value["allocated_vram_bytes"]) for value in values if value["allocated_vram_bytes"] is not None]
            reserved = [int(value["reserved_vram_bytes"]) for value in values if value["reserved_vram_bytes"] is not None]
            if allocated:
                payload["perf/allocated_vram_gb"] = max(allocated) / BYTES_PER_GIB
            if reserved:
                payload["perf/reserved_vram_gb"] = max(reserved) / BYTES_PER_GIB
            self.run.log(payload)
            self._step_buffer.clear()
        except Exception as error:
            self._disable_after("step logging", error)

    def log_metrics(self, payload: dict[str, Any]) -> None:
        """Mirror an already-aggregated local metric record without recomputing it."""
        if not self.active:
            return
        try:
            self.run.log(dict(payload))
        except Exception as error:
            self._disable_after("step logging", error)

    def log_epoch(self, record: dict[str, Any], *, samples_per_epoch: int) -> None:
        if not self.active:
            return
        try:
            elapsed = float(record["elapsed_seconds"])
            payload: dict[str, float | int] = {
                "epoch": int(record["epoch"]),
                "epoch/loss": float(record["train_loss"]),
                "epoch/h16_loss": float(record["prediction_loss_h16"]),
                "epoch/h64_loss": float(record["prediction_loss_h64"]),
                "epoch/h256_loss": float(record["prediction_loss_h256"]),
                "epoch/seconds": elapsed,
                "epoch/samples_per_sec": samples_per_epoch / elapsed if elapsed > 0 else 0.0,
                "epoch/global_step": int(record["global_step"]),
                "epoch/skipped_optimizer_steps": int(record["skipped_amp_steps"]),
                "epoch/peak_allocated_vram_gb": (
                    float(record["peak_vram_allocated"] or 0) / BYTES_PER_GIB
                ),
                "epoch/peak_reserved_vram_gb": (
                    float(record["peak_vram_reserved"] or 0) / BYTES_PER_GIB
                ),
            }
            self.run.log(payload)
        except Exception as error:
            self._disable_after("epoch logging", error)

    def finish(self, summary: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            self._flush_steps()
            if not self.active:
                return
            for name, value in summary.items():
                self.run.summary[name] = value
            self.run.finish()
            self.run = None
        except Exception as error:
            self._disable_after("finish", error)

    def fail(self, error: BaseException) -> None:
        if not self.active:
            return
        try:
            self.run.summary["status"] = "FAILED"
            self.run.summary["failure_message"] = f"{type(error).__name__}: {error}"[:500]
            self.run.finish(exit_code=1)
            self.run = None
        except Exception as wandb_error:
            self._disable_after("failure finalization", wandb_error)


def read_persisted_run_id(output: str | Path) -> str | None:
    path = Path(output) / "wandb_run.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        run_id = value.get("run_id")
        return run_id if isinstance(run_id, str) and run_id else None
    except (OSError, ValueError, TypeError):
        return None
