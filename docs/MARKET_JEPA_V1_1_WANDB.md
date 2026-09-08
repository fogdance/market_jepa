# Market-JEPA V1.1 W&B logging

W&B is an optional observability layer for formal V1.1 training. Local
`last.pt`, `step_metrics.jsonl`, `interval_metrics.csv`, `interval_history.json`,
`final_summary.json`, and `protocol.json` remain authoritative. W&B metrics never select a checkpoint,
stop training, or alter the optimizer, scheduler, EMA, or learning rate.

## Setup

Install project dependencies and authenticate once:

```bash
wandb login
```

The base formal config enables online logging:

```yaml
logging:
  wandb:
    enabled: true
    mode: online
    project: market-jepa
    group: v1.1-formal-fixed-budget-v1
    entity: null
    run_name: null
    log_every_optimizer_steps: 50
    watch_model: false
    upload_checkpoint: false
```

Set `enabled: false` or `mode: disabled` to turn logging off. Set
`mode: offline` to create a local W&B run that can later be uploaded with
`wandb sync <run-dir>`. `entity` is intentionally unset so the SDK uses the
authenticated user's default account. A null run name generates
`v1.1-<SIZE>-<PARAMS>M-<STEPS>Kstep-seed<SEED>`.

## Metrics

`train/*`, `optim/*`, and `perf/*` use successful optimizer `global_step` as
their x-axis. Losses are interval means, not the last microbatch value.
`samples_seen` is logged alongside `global_step`. The fixed-budget formal runner
writes 50-step local aggregates to `step_metrics.jsonl` and 5,000-step recovery
intervals to `interval_metrics.csv` / `interval_history.json`; W&B mirrors the same
step aggregates but is never the source of truth. Legacy `epoch/*` helpers remain
only for backward-compatible tests and legacy runs. W&B SDK system metrics may
also appear. The model is not passed to `wandb.watch`, and checkpoints are not
uploaded.

## Resume and failures

The formal runner stores `wandb_run.json` and also writes the run ID into
`last.pt`. `--resume last.pt` reuses that ID. If the ID is unavailable, model
resume still proceeds and a new W&B run is tagged `resumed-new-wandb-run`.
Model resume validation remains entirely checkpoint/config/data/scaler/RNG
based.

Every W&B SDK operation is failure-isolated. Initialization, logging, network,
or finish errors produce a warning, disable W&B for the remaining invocation,
and do not fail formal training.
