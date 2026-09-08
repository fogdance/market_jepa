from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_jepa_v1_1.config import DEFAULT_V11_CONFIG, validate_v11_config
from market_jepa_v1_1.formal_training import (
    FixedBudgetMetricLogger, _wandb_run_name, _wandb_tags, run_formal_training,
)
from market_jepa_v1_1.wandb_logging import V11WandbLogger, read_persisted_run_id


class FakeRun:
    def __init__(self, *, run_id: str = "run-123", fail_log: bool = False) -> None:
        self.id = run_id
        self.url = f"https://wandb.invalid/{run_id}"
        self.fail_log = fail_log
        self.metrics: list[tuple[str, str | None]] = []
        self.logs: list[dict] = []
        self.summary: dict = {}
        self.finish_calls: list[int | None] = []

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.metrics.append((name, step_metric))

    def log(self, value: dict) -> None:
        if self.fail_log:
            raise ConnectionError("network down")
        self.logs.append(dict(value))

    def finish(self, exit_code: int | None = None) -> None:
        self.finish_calls.append(exit_code)


class FakeWandb:
    def __init__(self, run: FakeRun | None = None, *, fail_init: bool = False) -> None:
        self.run = run or FakeRun()
        self.fail_init = fail_init
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise ConnectionError("cannot initialize")
        return self.run


def wandb_config(**updates) -> dict:
    value = deepcopy(DEFAULT_V11_CONFIG["logging"]["wandb"])
    value.update(updates)
    return value


def initialized_logger(tmp_path: Path, *, interval: int = 1, run: FakeRun | None = None):
    backend = FakeWandb(run)
    logger = V11WandbLogger(
        wandb_config(log_every_optimizer_steps=interval), tmp_path, backend=backend,
    )
    assert logger.init(
        {"design_version": "V1.1"}, run_name="test-run",
        tags=["formal", "v1.1", "S"],
    ) == backend.run.id
    return logger, backend


def log_step(logger: V11WandbLogger, step: int, loss: float) -> None:
    logger.log_step(
        global_step=step,
        loss=loss,
        h16_loss=loss + 1,
        h64_loss=loss + 2,
        h256_loss=loss + 3,
        learning_rate=3e-4,
        grad_norm=2.0 * loss,
        skipped_optimizer_steps=0,
        step_seconds=2.0,
        samples=8,
    )


def test_wandb_disabled_does_nothing(tmp_path):
    backend = FakeWandb()
    logger = V11WandbLogger(wandb_config(enabled=False), tmp_path, backend=backend)
    assert logger.init({}, run_name="unused", tags=[]) is None
    log_step(logger, 1, 1.0)
    logger.log_epoch({}, samples_per_epoch=1)
    logger.finish({"status": "unused"})
    assert backend.init_calls == []


def test_wandb_init_after_hard_gate():
    source = inspect.getsource(run_formal_training)
    hard_gate = source.index("if not hard_gate_pass")
    scaled_dataset = source.index("train_dataset = V11ContractDataset")
    model = source.index("model = MarketJEPAV11")
    wandb_init = source.index("run_id = wandb_logger.init")
    fit = source.index("history = trainer.fit")
    assert hard_gate < scaled_dataset < model < wandb_init < fit


def test_wandb_step_logging_uses_global_step(tmp_path):
    logger, backend = initialized_logger(tmp_path)
    log_step(logger, 37, 1.0)
    payload = backend.run.logs[-1]
    assert payload["global_step"] == 37
    assert ("train/*", "global_step") in backend.run.metrics
    assert ("optim/*", "global_step") in backend.run.metrics
    assert ("perf/*", "global_step") in backend.run.metrics


def test_wandb_interval_metrics_are_averaged(tmp_path):
    logger, backend = initialized_logger(tmp_path, interval=2)
    log_step(logger, 1, 1.0)
    assert backend.run.logs == []
    log_step(logger, 2, 3.0)
    payload = backend.run.logs[-1]
    assert payload["global_step"] == 2
    assert payload["train/loss"] == pytest.approx(2.0)
    assert payload["train/h16_loss"] == pytest.approx(3.0)
    assert payload["train/h64_loss"] == pytest.approx(4.0)
    assert payload["train/h256_loss"] == pytest.approx(5.0)
    assert payload["optim/grad_norm"] == pytest.approx(4.0)
    assert payload["perf/step_seconds"] == pytest.approx(2.0)
    assert payload["perf/samples_per_sec"] == pytest.approx(4.0)


def test_wandb_epoch_metrics_match_epoch_history(tmp_path):
    logger, backend = initialized_logger(tmp_path)
    record = {
        "epoch": 7,
        "train_loss": 0.4,
        "prediction_loss_h16": 0.1,
        "prediction_loss_h64": 0.2,
        "prediction_loss_h256": 0.3,
        "global_step": 900,
        "skipped_amp_steps": 2,
        "elapsed_seconds": 25.0,
        "peak_vram_allocated": 2 * 1024 ** 3,
        "peak_vram_reserved": 3 * 1024 ** 3,
    }
    logger.log_epoch(record, samples_per_epoch=1000)
    payload = backend.run.logs[-1]
    assert payload["epoch"] == record["epoch"]
    assert payload["epoch/loss"] == record["train_loss"]
    assert payload["epoch/h16_loss"] == record["prediction_loss_h16"]
    assert payload["epoch/h64_loss"] == record["prediction_loss_h64"]
    assert payload["epoch/h256_loss"] == record["prediction_loss_h256"]
    assert payload["epoch/seconds"] == record["elapsed_seconds"]
    assert payload["epoch/global_step"] == record["global_step"]
    assert payload["epoch/skipped_optimizer_steps"] == record["skipped_amp_steps"]
    assert payload["epoch/samples_per_sec"] == 40.0
    assert payload["epoch/peak_allocated_vram_gb"] == 2.0
    assert payload["epoch/peak_reserved_vram_gb"] == 3.0
    assert ("epoch/*", "epoch") in backend.run.metrics


def test_wandb_failure_does_not_abort_training(tmp_path):
    warnings: list[str] = []
    run = FakeRun(fail_log=True)
    backend = FakeWandb(run)
    logger = V11WandbLogger(
        wandb_config(log_every_optimizer_steps=1), tmp_path,
        backend=backend, warning=warnings.append,
    )
    logger.init({}, run_name="failure-test", tags=[])
    log_step(logger, 1, 1.0)  # must not raise
    log_step(logger, 2, 2.0)  # disabled no-op
    assert logger.disabled is True
    assert len(warnings) == 1
    assert "network down" in warnings[0]

    init_warnings: list[str] = []
    init_logger = V11WandbLogger(
        wandb_config(), tmp_path / "init", backend=FakeWandb(fail_init=True),
        warning=init_warnings.append,
    )
    assert init_logger.init({}, run_name="failure-test", tags=[]) is None
    assert init_logger.disabled is True
    assert "cannot initialize" in init_warnings[0]


def test_wandb_run_id_persisted(tmp_path):
    logger, _ = initialized_logger(tmp_path)
    value = json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8"))
    assert value == {
        "run_id": "run-123",
        "project": "market-jepa",
        "group": "v1.1-formal-fixed-budget-v1",
        "run_name": "test-run",
        "url": "https://wandb.invalid/run-123",
    }
    assert read_persisted_run_id(tmp_path) == "run-123"


def test_wandb_resume_reuses_run_id(tmp_path):
    backend = FakeWandb(FakeRun(run_id="original-id"))
    logger = V11WandbLogger(wandb_config(), tmp_path, backend=backend)
    logger.init(
        {}, run_name="resumed", tags=["formal"],
        resume_run_id="original-id", resumed=True,
    )
    assert backend.init_calls[0]["id"] == "original-id"
    assert backend.init_calls[0]["resume"] == "allow"
    assert "resumed-new-wandb-run" not in backend.init_calls[0]["tags"]

    new_backend = FakeWandb(FakeRun(run_id="new-id"))
    new_logger = V11WandbLogger(wandb_config(), tmp_path / "new", backend=new_backend)
    new_logger.init({}, run_name="resumed", tags=["formal"], resumed=True)
    assert "resumed-new-wandb-run" in new_backend.init_calls[0]["tags"]


def test_wandb_not_used_by_model_dataset_sampler():
    root = Path(__file__).resolve().parents[1] / "market_jepa_v1_1"
    for name in ("model.py", "dataset.py", "sampler.py", "imc.py"):
        assert "wandb" not in (root / name).read_text(encoding="utf-8").lower()


def test_wandb_finish_on_success(tmp_path):
    logger, backend = initialized_logger(tmp_path, interval=50)
    log_step(logger, 1, 1.0)
    logger.finish({"status": "V1_1_FORMAL_TRAINING_PASS", "final_epoch": 49})
    assert backend.run.logs[-1]["global_step"] == 1  # partial interval is flushed
    assert backend.run.summary["status"] == "V1_1_FORMAL_TRAINING_PASS"
    assert backend.run.summary["final_epoch"] == 49
    assert backend.run.finish_calls == [None]


def test_wandb_config_modes_and_unsupported_uploads():
    for mode in ("online", "offline", "disabled"):
        config = deepcopy(DEFAULT_V11_CONFIG)
        config["logging"]["wandb"]["mode"] = mode
        validate_v11_config(config)
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["logging"]["wandb"]["upload_checkpoint"] = True
    with pytest.raises(ValueError, match="upload_checkpoint=true"):
        validate_v11_config(config)


def test_wandb_default_formal_name_and_tags():
    config = deepcopy(DEFAULT_V11_CONFIG)
    assert _wandb_run_name(config, 8_262_661) == "v1.1-S-8.26M-250Kstep-seed42"
    assert _wandb_tags(config, 8_262_661) == ["formal", "v1.1", "S", "8M", "BF16"]
    config["logging"]["wandb"]["run_name"] = "manual-name"
    assert _wandb_run_name(config, 8_262_661) == "manual-name"


class _MirrorLogger:
    def __init__(self) -> None:
        self.payloads = []

    def log_metrics(self, payload):
        self.payloads.append(dict(payload))


def test_fixed_budget_metric_logger_local_first_and_exact_mirror(tmp_path):
    mirror = _MirrorLogger()
    logger = FixedBudgetMetricLogger(tmp_path, mirror, interval=2)
    common = dict(
        skipped_optimizer_steps=0, allocated_vram_bytes=None, reserved_vram_bytes=None,
    )
    logger.log_step(
        global_step=1, samples_seen=8, loss=1.0, h16_loss=2.0, h64_loss=3.0,
        h256_loss=4.0, learning_rate=1e-4, grad_norm=2.0, step_seconds=2.0,
        samples=8, **common,
    )
    assert not (tmp_path / "step_metrics.jsonl").exists()
    assert mirror.payloads == []
    logger.log_step(
        global_step=2, samples_seen=16, loss=3.0, h16_loss=4.0, h64_loss=5.0,
        h256_loss=6.0, learning_rate=2e-4, grad_norm=4.0, step_seconds=2.0,
        samples=8, **common,
    )
    records = [json.loads(line) for line in (tmp_path / "step_metrics.jsonl").read_text().splitlines()]
    assert records == [{
        "global_step": 2, "samples_seen": 16, "loss": 2.0, "h16_loss": 3.0,
        "h64_loss": 4.0, "h256_loss": 5.0, "learning_rate": 2e-4,
        "grad_norm": 3.0, "skipped_optimizer_steps": 0, "step_seconds": 2.0,
        "samples_per_sec": 4.0, "allocated_vram": None, "reserved_vram": None,
    }]
    assert mirror.payloads == [{
        "global_step": 2, "samples_seen": 16, "train/loss": 2.0,
        "train/h16_loss": 3.0, "train/h64_loss": 4.0, "train/h256_loss": 5.0,
        "optim/learning_rate": 2e-4, "optim/skipped_optimizer_steps": 0,
        "perf/step_seconds": 2.0, "perf/samples_per_sec": 4.0,
        "optim/grad_norm": 3.0,
    }]


def test_fixed_budget_metric_logger_resume_buffer_and_reconcile(tmp_path):
    mirror = _MirrorLogger()
    logger = FixedBudgetMetricLogger(tmp_path, mirror, interval=2)
    logger.log_step(
        global_step=1, samples_seen=8, loss=1.0, h16_loss=1.0, h64_loss=1.0,
        h256_loss=1.0, learning_rate=1e-4, grad_norm=1.0,
        skipped_optimizer_steps=0, step_seconds=1.0, samples=8,
        allocated_vram_bytes=None, reserved_vram_bytes=None,
    )
    state = logger.state_dict()
    resumed = FixedBudgetMetricLogger(tmp_path, mirror, interval=2)
    resumed.load_state_dict(state)
    resumed.log_step(
        global_step=2, samples_seen=16, loss=3.0, h16_loss=3.0, h64_loss=3.0,
        h256_loss=3.0, learning_rate=2e-4, grad_norm=3.0,
        skipped_optimizer_steps=0, step_seconds=1.0, samples=8,
        allocated_vram_bytes=None, reserved_vram_bytes=None,
    )
    path = tmp_path / "step_metrics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"global_step": 999, "samples_seen": 999}) + "\n")
    resumed.reconcile(2)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [record["global_step"] for record in records] == [2]
    assert records[0]["loss"] == pytest.approx(2.0)
