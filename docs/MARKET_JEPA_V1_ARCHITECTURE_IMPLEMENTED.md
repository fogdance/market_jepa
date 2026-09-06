# Market World Model V1.0 — implemented architecture

V1 implements Cross-Scale Conditional Representation at width 256. It is an architecture development deliverable; no formal multi-commodity training, held-out RB evaluation, or V0/V1 benchmark was run. Authoritative measurements and final gate results are in `artifacts/evaluation/v1_architecture_development/`.

**Final acceptance: V1_ARCHITECTURE_IMPLEMENTATION_PASS.** The authorized host run completed **115 passed / 0 failed / 0 skipped**, including all 29 V1 tests and the V0 multiworker resume regression. CUDA B=2/B=8 and a 100-step JM Train-only smoke passed with measured VRAM, finite loss/gradients, EMA updates and checkpoint restoration.

## Entry points and compatibility

- Model: `market_jepa_v1.model.MarketJEPAV1(config["model"])`.
- Configuration: `configs/v1/market_jepa_v1.yaml`, loaded by `load_v1_config`; `design_version="1.0"`.
- Training: `market_jepa_v1.training.V1Trainer`; default fixed-budget training without validation. Supplying an explicit validation dataset enables the existing H64 evaluation/selection mechanism; this path was exercised only with synthetic data.
- Development: `develop_market_jepa_v1.py`. It runs full-width synthetic B=2 and B=8 steps, a bounded JM Train smoke, and the complete pytest suite. It refuses to overwrite an existing final report; use a new output directory for a rerun.
- Checkpoint: `load_v1_checkpoint` validates version/config/schema, and `model_from_checkpoint` uses strict state loading. `V1Trainer.resume` restores optimizer, scheduler, scaler, RNG, sampler epoch and step with source/config/normalization/implementation checks.

V0's Python implementation and configuration files are unchanged. Its manifest enumerates `market_jepa/**/*.py` and top-level `configs/*.yaml`; therefore V1 is an independent package and its YAML lives in a config subdirectory. This preserves the pre-development V0 manifest:

```text
4e1b548e27dca817365a62b1d634d1e3ca701f7f0461037610e6e1b7448548a5
```

V1 has its own manifest including V1 code/config/entrypoint and the reused V0 implementation. No historical V0 checkpoint was migrated or partially loaded into V1.

## Data contract and tensor shapes

`B` is batch size, `M/D/W` are padded sequence lengths, `F` denotes market dimensions and `C` context dimensions. All input features are upstream-prepared floating tensors. Padding masks are boolean with **True = padding**.

| Tensor | Shape | Existing upstream profile |
|---|---|---|
| minute_market | B × M × Fm | Fm=14, M=512 |
| minute_context | B × M × Cm | Cm=5 |
| daily_market / daily_context | B × D × Fd/Cd | Fd=14, Cd=1 |
| weekly_market / weekly_context | B × W × Fw/Cw | Fw=14, Cw=1 |
| minute/daily/weekly_padding_mask | B × M/D/W | Optional; omitted means all valid |
| targets[h] / persistence[h] | B × h × Fm | h ∈ {16,64,256} |
| target/persistence_padding_masks[h] | B × h | Optional |
| minute_local_tokens | B × (M+1) × 256 | Includes CLS |
| daily_local_tokens | B × D × 256 | Full GRU sequence |
| weekly_local_tokens | B × W × 256 | Full GRU sequence |
| T, feedback output | B × (M+1+D+W) × 256 | All scales concatenated as tokens |
| S, each round's state | B × 8 × 256 | Slot 0 is BELIEF |
| final_belief / z_market | B × 256 | Same tensor |
| predictions[h] / targets[h] outputs | B × 256 | Unchanged JEPA loss interface |

All six feature dimensions are config-controlled. Tests also use seven market coordinates at every scale, and distinct dimensions 7/9/11 with context dimensions 3/2/0. This verifies the interface is IMC-compatible. The model does not calculate IMC or fit a commodity-specific normalizer. The real-data development smoke uses existing V0 features and must not be described as an IMC training experiment.

`adapt_market_batch` slices V0's combined Daily/Weekly arrays according to the declared dimensions, derives masks from lengths and checks source_max ≤ anchor when provenance is present. It passes only model tensors; identity/logging fields, outcomes and all future-context metadata remain outside forward. Direct unexpected model keys are rejected. The dataset supplies strictly historical completed bars plus causal partial Daily/Weekly snapshots, and purges anchors whose maximum future horizon exceeds the training boundary.

## Local representations

**Minute:** `MinuteSequenceCore` contains Linear(Fm→256), a learned minute scale embedding, CLS, reused sinusoidal positions, four pre-norm Transformer layers (8 heads, FFN 1024, GELU, dropout 0.1), and output LayerNorm. The online-only Linear(Cm→256, no bias) is a sibling of the core. Its projection is added to market token projections before positional encoding and the Transformer. The complete normalized sequence is retained.

**Daily/Weekly:** two independent unidirectional, two-layer GRUs with hidden size 128 consume their concatenated market/context inputs. The complete causal hidden sequence is projected 128→256, then receives the corresponding learned scale embedding. Neither branch reduces to a final hidden vector before cross-scale interaction.

**Padding:** valid GRU steps are stably compacted, packed, then scattered back to their original token positions. Left, right and interior padding are supported. Padded input values are zeroed before projection/GRU, padded keys are excluded from attention, and padded output tokens are zeroed after local encoders and feedback. Minute positions count valid tokens, making padding placement irrelevant to valid representations. Daily/Weekly may be wholly absent, including zero-length sequences; minute histories must contain at least one valid observation per sample. NaN/Inf in valid input tokens raises an error; NaN padding is excluded. The local minute Transformer has bidirectional historical attention, consistent with V0: all observation tokens are already known at the anchor.

## Two complete cross-scale rounds and final belief

Each of the two independently parameterized `CrossScaleInteractionBlock` instances executes:

```text
S = S + MHA(LN(S), LN(T), LN(T), key_padding_mask=T_mask)
S = S + FFN(LN(S))
T = T + MHA(LN(T), LN(S), LN(S))
T = T + FFN(LN(T))
```

Feedback attention/FFN are shared across minute, Daily and Weekly tokens within a round. Cross-scale attention has K×N and N×K query-key interactions, where K=8 and N=M+1+D+W; there is no N×N all-scale attention. Minute local attention retains its existing quadratic local cost.

After round 1, minute tokens contain information from Daily and Weekly through shared State. Round 2 reads these conditioned minute tokens. Final belief is `Linear(256→256)(LayerNorm(S_round2[:,0]))`; there is no 544-dimensional branch-vector fusion.

The user explicitly selected **two complete bidirectional rounds**. Consequently round-2 feedback executes, but its resulting T has no downstream use in the current loss. Its 790,272 parameters have `requires_grad=True` yet receive no JEPA gradient, and are included in the trainable count. This is an explicit limitation of the frozen topology, verified by perturbing terminal-feedback weights and observing unchanged belief but changed terminal tokens. No extra read, hidden parameter sharing, auxiliary loss or objective was added to train that terminal branch.

`cross_scale_rounds=0` is allowed only with `debug=True` and a debug config. It constructs no blocks and uses the minute CLS through the belief head; it is a bypass diagnostic, not a claimed scientific ablation. Formal configs enforce two rounds and all specified capacity settings.

## EMA and unchanged JEPA objective

EMA is an initially identical deep copy of the minute market core only. It includes market projection, minute scale embedding, CLS, sinusoidal buffer, local Transformer and normalization. It excludes minute context projection, Daily/Weekly encoders, state tokens, cross-scale blocks, belief head and predictors.

The public target path accepts only market observations and a padding mask. It produces the normalized minute CLS for each future interval `(anchor, anchor+h]`. Targets stay in eval mode with `requires_grad=False`, are evaluated under no_grad, have no gradients and are excluded from optimizer parameters. EMA uses tau=0.996, matching parameter names and copying named buffers. It updates only after a successful optimizer step, using V0's unchanged AMP skip handling.

H16/H64/H256 reuse the exact V0 `Predictor` implementation, each 256→512→256. `jepa_loss` is imported unchanged: equal-weight mean cosine prediction loss, lambda_var=lambda_cov=0. No CDF/CME/RSSM, supervised future labels, actor/critic, reward/PnL, regime labels or commodity identity modules exist in V1.

## Parameter accounting

Counts exclude EMA from trainable totals and use the existing 14/5/14/1/14/1 feature profile.

| Component | V0 | V1 |
|---|---:|---:|
| Minute market local encoder | 3,163,648 | 3,163,904 |
| Minute context branch/projection | 3,744 | 1,280 |
| Daily encoder including projection/scale | 154,752 | 188,032 |
| Weekly encoder including projection/scale | 154,752 | 188,032 |
| Cross-scale blocks (two) | 0 | 3,161,088 |
| Learned state tokens | 0 | 2,048 |
| Fusion / belief head | 410,880 | 66,304 |
| Three predictors | 788,736 | 788,736 |
| **Trainable total** | **4,676,512** | **7,559,424** |
| EMA, separately | 3,163,648 | 3,163,904 |

Difference: **2,882,912**, ratio **1.6164662894**. The >3× stop threshold is checked automatically. V1 trainable parameters with a connected loss path total 6,769,152 after subtracting terminal feedback; the primary report still uses all requires_grad parameters, as requested.

## Debug API and validation

`model(batch, return_intermediates=True)` additionally returns `intermediates` containing:

```text
minute_local_tokens, daily_local_tokens, weekly_local_tokens
state_tokens_after_round_1, tokens_after_feedback_round_1
state_tokens_after_round_2, tokens_after_feedback_round_2
final_belief, token_lengths, token_padding_mask
```

These tensors are attached to the current graph so callers may retain gradients for audits. Default forward creates no debug dictionary, and modules do not retain intermediate tensors across calls. Attention weights are not returned or interpreted as causal importance. Evidence comes from controlled interventions, feedback removal, source-history integrity checks and gradients.

Targeted tests verify that changing only Daily, Weekly or minute context changes belief; only context changes the local minute representation before cross-scale interaction. Both higher scales change non-CLS minute tokens at the actual input to round 2. Removing first-round feedback removes that effect. The loss B.sum() reaches every valid intermediate period token, non-CLS minute tokens, minute context projection and all learned state slots. Tests also verify future-context invariance of targets, target/persistence horizon contracts, zero-round bypass, padding invariance, GRU causality, strict checkpoint failures, exact resumed model/optimizer trajectory, synthetic evaluation metrics and unchanged V0 manifest/counts.

## Changes from the initial proposal and self-review

1. **Location:** independent package and nested YAML replace the suggested package/top-level config location, because inspection proved additive files there would break V0 exact resume. No V0 manifest rule was weakened.
2. **Terminal feedback:** both complete rounds are retained per the user's explicit choice; unused terminal-loss connectivity is disclosed and tested.
3. **EMA ownership:** context projection is outside the minute market core, making it structurally impossible for ordinary target forward to consume future context while sharing all necessary market parameters.
4. **Debug zero rounds:** the unspecified output is defined as projected minute CLS and is excluded from formal configuration.
5. **Real smoke input:** existing Train features verify training compatibility. Production multiscale IMC preprocessing remains upstream; no feature formula, scaler policy or IMC baseline was changed.
6. **Self-review fixes:** added exact H16/H64/H256 target/persistence shape checks and cross-checked redundant checkpoint config metadata before loading. No unrelated refactor was introduced.
7. **Test-runner adjustment after observed failure:** PyTorch worker queues default to indefinite waits when descriptor-sharing fails in a background feeder thread. A separate V1 development test runner bounds otherwise-unlimited DataLoader waits at 15 seconds. It retains all 115 cases and assertions and records IPC failures explicitly; it changes no V0 source, worker count, sampling, loss or arithmetic. Standard `python -m pytest -q` remains available on the unrestricted host.

Research review: cross-scale interaction precedes final compression; Daily and Weekly influence minute representation through first feedback; market/context interact before local compression; target inputs remain market-only; identity shortcuts are absent; parameter ratio remains bounded; loss/predictor/optimizer scientific controls are preserved. These are architecture/connectivity conclusions, not evidence of OOS predictive improvement or proof that V0 late fusion caused prior generalization failures.

## Runtime evidence and remaining boundary

The configured Python is `/home/v/Documents/work/market_jepa/.venv/bin/python`, PyTorch 2.13.0+cu130. The initial restricted sandbox could not initialize CUDA or local multiprocessing IPC. After explicit authorization, the host environment exposed the RTX 4060 Ti and allowed both CUDA execution and the unchanged V0 multiworker regression. No host driver, service, process or device configuration was changed.

Observed full-width CUDA smoke results (forward includes JEPA loss; first-use overhead is included):

| Batch | M / D / W | Forward seconds | Backward seconds | Pre-clip gradient norm |
|---|---|---:|---:|---:|
| 2 | 512 / 32 / 8 | 0.034216 | 0.020142 | 2.692820 |
| 8 | 512 / 32 / 8 | 0.036086 | 0.028996 | 2.472203 |

Both completed forward, unchanged JEPA loss, backward, optimizer step and checked EMA update, with frozen target gradients. Peak allocated/reserved VRAM was **221,760,512 / 306,184,192 bytes** for B=2 and **447,068,672 / 585,105,408 bytes** for B=8. The maximum allocated figure is 426.36 MiB; maximum reserved is 558.00 MiB.

The final real JM Train smoke materialized 11,086 rows through **2018-02-12**, selected 200 training anchors and completed **100 successful optimizer steps**, B=2, on CUDA in **10.675 seconds** with zero skipped steps. Maximum pre-clip gradient norm was **2.620644**; finite gradients were clipped at the unchanged norm 1.0. Peak allocated/reserved VRAM was **307,959,808 / 633,339,904 bytes** (293.69 / 604.00 MiB). Loss/parameters remained finite. Saved model, EMA, predictor outputs and resumed optimizer step 100 were verified; the final checkpoint manifest equals the current V1 source manifest. No validation dataset was evaluated, no best checkpoint was selected, and no RB data was loaded. These measurements establish execution integrity only; the observed loss and latent diagnostics are not evidence of model quality.

The complete test-suite counts and final gates are recorded in `artifacts/evaluation/v1_architecture_development/{summary.json,test_results.txt,test_results.xml}`. Production multiscale IMC preprocessing integration remains a prerequisite before any formal benchmark; this task stops at architecture development.

The final development suite completed in **8.75 seconds** with **115 passed, 0 failed, 0 skipped**. A direct standard `pytest -q` confirmation also passed all 115 tests in **8.92 seconds**. Both include `test_multiworker_runtime_preserves_resume_trajectory`, which also passed independently once local descriptor-sharing IPC was authorized. The initial sandbox failure and minimal reproduction remain archived as environment evidence; no failed test was reclassified.

CUDA AMP evidence: the first synthetic probe at PyTorch's default loss scale 65,536 produced nonfinite gradients before any optimizer/EMA update. FP32 was finite; FP16 scales 32,768 and 8,192 were finite. The development runner now performs deterministic, no-update calibration and selected 32,768 for B=2, B=8 and real-data smoke. A CUDA test verifies calibration preserves weights, empty optimizer state, gradients cleared and CPU/CUDA RNG states. All 100 real-data optimizer steps succeeded at the calibrated scale.
