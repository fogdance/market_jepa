# V1.1 formal evaluation implementation

Implementation lives in `market_jepa2`. The sibling `market_jepa` directory is a
read-only artifact source; its running training job is not modified. Historical
V0 checkpoints provide research context only, never matched numerical controls.

## Frozen protocol and explicit user override

The main design is `MARKET_JEPA_V1_1_EVALUATION_DESIGN_REVIEWED_V2.md`.
The user explicitly selected **V0 exact outcome parity** after the discrepancy
between V2 formulas and actual V0 code was demonstrated. Consequently Return,
MFE and MAE use price ratios minus one; RV is `sqrt(mean(diff(log(close))**2))`.
The V0 scalar reduction order and float32 results are preserved. This override
is recorded in every manifest. V0 source code remains unchanged.

Defaults are frozen: seed 42; episode chronological ProbeFit/Dev/Test 70/15/15;
4096 total anchors per Train commodity; RB-Dev 32768, RB-Test 65536; structural
recipient cap 16384; 2000 contract-day bootstrap replicates and 95% intervals.
Budgets are upper bounds, without replacement. Episode selection is balanced
before selecting anchors. Probe allocation floors the 70%/15% counts while
reserving one episode for each partition. Commodities with fewer than three
eligible episodes are excluded from probe fitting and the probe aggregate, but
remain in A1. RB splits each eligible lineage at `max(1, floor(n/3))`; lineages
with fewer than two eligible episodes are excluded. Re-main episodes whose raw
anchors overlap across partitions are rejected instead of splitting minutes.

The actual Train commodity list and epoch budget come from the resolved training
snapshot/checkpoint. They are not hardcoded to five commodities or 50 epochs.
The running M artifact reports 67 commodities and 100 epochs; its formal endpoint
must therefore be epoch 99. Model metrics require a completed, hash-matching
training summary, contiguous budget history and a valid gradient audit.

## Data and metrics

`evaluate_market_jepa_v1_1.py` exposes `manifest`, `train-domain`, `compare`,
`freeze-rb`, `rb-dev`, and `rb-test`. All production input CSVs in the requested
scope are verified against the build manifest; Train commands do not open RB
bar files. Datasets are loaded one commodity at a time using existing V1.1
IMC, history lineage, partial bar and target construction. The checkpoint's
model/encoder/dataset/IMC implementation hashes must match these semantics.

Manifest SHA256 covers ordered anchors, partitions, horizon validity and protocol.
Model artifacts separately record checkpoint/config/scaler/training implementation
and evaluation code hashes. Every size/control must use the same manifest.
Only final fixed-budget checkpoints are accepted. Checkpoints are read without
changing training or optimizer state; models and EMA are frozen for evaluation.

A1 encodes the current market-only observation with the checkpoint's EMA core
for persistence. Each prediction and persistence is compared to its own model's
clean same-origin future EMA target. Gain is a ratio of mean cosine errors, not
the average of per-anchor ratios. Cross-size absolute latent-loss ranking is
not implemented.

A2 fits Ridge separately for all twelve outcomes. Belief, baseline and target
normalizers fit ProbeFit only and remain frozen during Fit+Dev refitting.
Alpha is chosen on ProbeDev from the frozen seven-value grid; ties select the
first/smallest value. Spectral factorization shares the Ridge computation across
alphas/outcomes without changing the objective. ProbeTest outcomes do not affect
fitting. The stored probes can be applied directly to RB with no fitting.

Baselines are unconditional ProbeFit target mean, causal current context,
MinuteRaw24 plus context, and primary MultiScaleSummary plus context. Raw24 uses
k-step price/OI change and minute realized RMS volatility for k=4/16/64/256,
high-low log range and previous-20-volume median ratios. MultiScaleSummary uses
the V2 source/window matrix and last/mean/std/min/max/valid_fraction per IMC
channel; validity fractions divide by nominal window capacity. Empty features
fill zero. Historical boundary counts and source availability are explicit.

Bootstrap resamples `(contract_uid, trading_date)` clusters within each commodity,
preserves observation weights within commodity, and gives each commodity equal
weight. Ratios are recalculated on every resample. Pairwise comparisons reuse
identical resampling draws. Zero baseline denominators and degenerate latents
raise an integrity error rather than producing a fabricated gain.

A3 matches a deterministic full-minute donor within commodity/series/partition,
preferring another contract and requiring disjoint minute windows. OI indices
5/6 and Volume indices 7/8 include corresponding validity replacement. The same
donor supplies joint replacement; clean recipient targets remain unchanged.
Missing donors are reported. Evidence concerns Price-aligned OI/Volume dependence.

## Controls and interpretation

`LateFusionControl` independently compresses Minute, Daily, CurrentWeekly and
HistoricalWeekly into four vectors. Only then does a 1024→1536→256 MLP fuse them.
The three higher-scale compressors each use four independent latent queries;
minute and EMA/predictor topology match S. Exact trainable parameters are
**8,318,208**, versus Full S **8,262,661** (+0.672%). EMA is separate.

`SourceControl` applies unavailable PAD/MASK semantics from epoch 0 for MinuteOnly
or NoHistoricalWeekly. Permanently unavailable tokenizer/read parameters are
frozen and excluded from intended trainable parameters; retained latent priors
and all active paths remain connected. Parameter differences are reported.
`ControlTrainer` reuses the existing optimizer/loss/EMA/sampler/scheduler loop and
adds explicit variant metadata. Full loaders reject control checkpoints. Resume
requires the same variant as well as existing strict configuration checks.

The explicit control training command requires a completed Full S reference,
reuses its frozen scaler, data population, effective batch and epoch budget,
and initializes model weights from scratch. This implementation task runs no
formal control training.

Frozen source diagnostics evaluate the original Full probe on ablated beliefs,
report per-horizon Gain/Skill deltas, and are always labelled OOD diagnostics.
Formal source claims require independently retrained controls and RB evidence.
No kNN gate, predictive-state identification claim or automatic checkpoint
selection is introduced.

The document does not quantify **H16 severe regression**. Architecture PASS can
be established when H16 has no negative Gain/Skill difference and all other
conditions pass. If H16 declines, the gate reports
`BLOCKED_H16_SEVERE_REGRESSION_THRESHOLD_UNDEFINED`; it never invents a tolerance.
This remaining definition must be frozen before reviewing such a result.

## RB campaign and local outputs

RB inventory reads timestamps/bars to construct eligible anchor membership, but
does not run a model or calculate outcomes/metrics. RB-Dev inference requires
the frozen checkpoint/probe/evaluation cohort. RB-Test additionally requires a
human approval JSON with the exact `freeze_sha256`, `human_review_pass: true`
and `tests_pass: true`. No approval is automatically generated.

The exclusive campaign ledger `rb_test_consumption.json` lives beside the freeze
file, independently of the result output path. It is marked consumed before
model evaluation begins. A failed attempt remains consumed and cannot silently
restart. All checkpoint/probe/data integrity checks precede consumption.
The CLI's `rb-test` command is implemented for later review, never run here.

Outputs include the requested A1/A2/A3 CSV/JSON files, fitted probe coefficients,
paired per-anchor errors/records, source diagnostics, provenance and summary.
Train-only runs emit RB files explicitly marked NOT_RUN. `compare` optionally
accepts matching completed RB-Test outputs and reports paired control/scaling
differences. Without RB, dependent formal claims remain NOT_EVALUATED.

W&B is optional (`--wandb-mode online|offline|disabled`), project `market-jepa`,
group `v1.1-evaluation`. It copies completed local summaries using the existing
failure-isolated SDK wrapper. Logging errors do not change metrics or gates.

## Commands after training and human code review

Use the existing interpreter from the sibling directory while running in
`/home/v/Documents/work/market_jepa2`:

```bash
/home/v/Documents/work/market_jepa/.venv/bin/python evaluate_market_jepa_v1_1.py manifest \
  --config /home/v/Documents/work/market_jepa/artifacts/training/market_jepa_v1_1_m/config_snapshot.yaml \
  --output artifacts/evaluation/v1_1_cohort/evaluation_manifest.json

/home/v/Documents/work/market_jepa/.venv/bin/python evaluate_market_jepa_v1_1.py train-domain \
  --checkpoint /home/v/Documents/work/market_jepa/artifacts/training/market_jepa_v1_1_m/checkpoints/last.pt \
  --manifest artifacts/evaluation/v1_1_cohort/evaluation_manifest.json \
  --output artifacts/evaluation/v1_1_m --device cuda --batch-size 8 --wandb-mode online

/home/v/Documents/work/market_jepa/.venv/bin/python train_market_jepa_v1_1_control.py \
  --reference /path/to/completed/full_s/checkpoints/last.pt \
  --variant LateFusion --output artifacts/training/v1_1_s_latefusion
```

For the other retrain controls substitute `MinuteOnly` or `NoHistoricalWeekly`
and use separate outputs. Only a completed S reference is accepted.

```bash
/home/v/Documents/work/market_jepa/.venv/bin/python evaluate_market_jepa_v1_1.py compare \
  --runs artifacts/evaluation/v1_1_s artifacts/evaluation/v1_1_m \
  --output artifacts/evaluation/v1_1_scaling
```

After every intended model/control and Train probe is frozen, prepare RB inventory
with `manifest --rb-inventory`, then use `freeze-rb --checkpoints ... --evaluations ...
--manifest ... --output ...` to bind the whole cohort. `rb-dev --freeze ... --output ...`
is a separate sanity action. Human review and authorization must precede
`rb-test --freeze ... --approval ... --output ...`. This task stops before those
formal executions.

Existing baseline tests are accepted as user-reported PASS. New evaluation/control
unit and fixture integration tests are run separately, without re-running the
expensive pre-existing capacity suite or using the training GPU.
