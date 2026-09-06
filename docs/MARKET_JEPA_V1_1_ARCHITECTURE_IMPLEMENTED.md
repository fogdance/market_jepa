# Market-JEPA V1.1 — Hierarchical IMC implemented architecture

## Status and scope

V1.1 is implemented as a separate `market_jepa_v1_1` package. V0/design 0.6.1 source, configuration, checkpoint contract, evaluation and training entrypoint are unchanged. V1.0 receives only the two required corrections: its terminal no-loss feedback module is no longer instantiated, and validation JEPA H64 cannot select a checkpoint.

This implementation stage does not start the formal 50-epoch run or RB evaluation.

## Actual hierarchy

```text
Historical Weekly [B,156,256] --reads--> Commodity C [B,4,256]
                                                |
                         +----------------------+
                         v
Daily [B,256,256] --> Contract K [B,4,256] <-- Current Weekly [B,64,256]
                         |
Commodity C -------------+----> H=[C;K] [B,8,256]
                                      |
Minute IMC+context [B,512,256] --> local Transformer x4
                                      |
                                      v
                         conditioned Minute [B,512,256]
                                      |
                   +------------------+------------------+
                   v                  v                  v
              Minute read        Contract read      Commodity read
                   +------------------+------------------+
                                      v
                          Belief tokens [B,8,256]
                                      v
                              final belief [B,256]
```

The four memories are tokenized and read separately. Daily and Current Weekly have independent attention softmaxes before source gating. Belief likewise reads Minute, Contract and Commodity with three independent attention operations. There is no pooled `[M;D;W]` attention.

## Data hierarchy and causal as-of construction

`contract_episodes.csv` provides `contract_uid`, `main_start_date`, `main_end_date`, `anchor_end_date` and role. Each real contract is an independent episode. Minute windows, Daily lifecycle, Current Weekly lifecycle and every JEPA target remain inside that `contract_uid`.

At each minute anchor:

- Minute is the latest 512 same-contract rows, left padded.
- Daily is completed lifecycle days plus exactly one current partial bar aggregated from same-contract minute rows with `datetime <= anchor` and the anchor `trading_date`.
- Current Weekly is completed lifecycle weeks plus exactly one partial lifecycle week aggregated only through anchor.
- Historical Weekly is the preceding three-year interval `[main_start-3Y, main_start)`, clipped to 156 tokens. Each visible real-contract segment resets IMC and marks its first token with `contract_boundary=1`.
- Anchors after loss of main status are bounded by `anchor_end_date`; the supplied package uses the episode calendar's 15-trading-day extension policy.

Cached 1d/1w bars are never used as the current forming bar. Weekly lifecycle bars are reconstructed from contract-local closed Daily lifecycle bars so a main-start week cannot import pre-main days. `bar_is_partial` and causal progress fields are explicit context features.

The implementation requires all H16/H64/H256 targets to exist. An anchor without H256 inside its real-contract minute array is excluded; no per-horizon fallback can cross a roll.

## Tensor and mask contract

| Input | Shape | Mask |
|---|---:|---|
| `minute_market`, `minute_imc_validity` | `[B,512,9]` | `minute_mask [B,512]` |
| `minute_context` | `[B,512,5]` | same |
| `daily_market`, `daily_imc_validity` | `[B,256,9]` | `daily_mask [B,256]` |
| `daily_context` | `[B,256,5]` | same |
| `current_weekly_market`, validity | `[B,64,9]` | `current_weekly_mask [B,64]` |
| `current_weekly_context` | `[B,64,5]` | same |
| `history_weekly_market`, validity | `[B,156,9]` | `history_weekly_mask [B,156]` |
| `history_weekly_context` | `[B,156,5]` | same |
| `history_weekly_contract_boundary` | `[B,156]` | same |

`True` means PAD. Padding is on the left and has zero numeric features plus false IMC validity. Fully masked memories use a safe dummy key internally and then multiply the source contribution by zero; unavailable source gates are renormalized, and an all-unavailable gate is exactly zero. Tests require finite outputs for these cases.

Symbol, commodity and contract metadata remain outside model inputs and are used only by sampling, integrity checks and logging.

## IMC coordinates

All four market inputs and future targets use the same nine ordered coordinates:

1. `log(Open/P0)`
2. `log(High/P0)`
3. `log(Low/P0)`
4. `log(Close/P0)`
5. `log(Close_t/Close_{t-1})`
6. `log(OI/OI0)`
7. `log(OI_t/OI_{t-1})`
8. `Volume/M0_volume`
9. `Volume/median(previous 20 Volume)`

Minute `P0` is the first valid window Close and `OI0` is its OI. Its fixed Volume baseline is the median of exactly 20 bars immediately before the window. Daily and Current Weekly Price/OI origins are fixed for the whole lifecycle at the first observable main-start minute Open/OI; their Volume baselines use 20 closed same-contract Daily/Weekly bars before main start. Historical Weekly resets independently at each visible contract segment.

Unavailable or nonpositive baselines produce numeric zero plus coordinate validity false; no epsilon and no future fill are used. A single post-IMC scaler is fitted jointly from FG/SA/JM/SH/SP online-memory snapshots. It stores means, standard deviations, feature counts, per-source population counts, exact feature order, fitted commodities and a SHA-256 checksum. The fitter rejects RB and rejects a population missing any of the five Train commodities.

## Encoders and information flow

`CommodityMemoryEncoder` uses four learned queries to cross-attend Historical Weekly, followed by a 256→1024→256 FFN.

`ContractLifecycleEncoder` uses four learned queries. It first reads Commodity State, then independently reads Daily and Current Weekly. Learned two-source logits are softmaxed only over available sources before their updates are fused.

`MinuteLocalEncoder` adds separate market, IMC-validity and causal-context projections before four 256D, eight-head, 1024-FFN local Transformer layers. It preserves all 512 tokens. `MinuteHigherScaleConditioner` then cross-attends the concatenated four Commodity plus four Contract state tokens before final compression.

`BeliefEncoder` uses eight learned belief tokens and separate Minute, Contract and Commodity reads. Learned source gates fuse the three updates. `LayerNorm + Linear` on belief token zero produces `[B,256]`.

`return_intermediates=True` exposes Commodity State, Contract State, local and conditioned Minute tokens, belief tokens, final belief and source gates. Attention weights are not treated as causal importance; scientific tests use controlled interventions.

## JEPA target, optimization and checkpoint semantics

The three unchanged comparable predictors are 256→512→256 for H16/H64/H256. The EMA target is a frozen deep copy of the minute market core only. It accepts future minute market IMC, validity and padding mask; it has no time context, Daily, Weekly, commodity identity or higher-scale state.

Future IMC uses the online window's exact `P0`, `OI0` and fixed Volume baseline. It does not reset at the target interval. Future Q20 may use prior future bars inside the target interval, but these target values never enter the online encoder. EMA defaults to `tau=0.996` and updates only after a successful optimizer step.

All intended online parameters must have non-`None`, finite gradients after real JEPA backward. AMP performs a replay-safe initial scale calibration without consuming an optimizer, scheduler or EMA step.

The sole formal checkpoint policy is `fixed_budget_final`; `last.pt` at the final budget endpoint is official. Validation JEPA loss may be logged as a diagnostic but cannot create or select a best checkpoint. Checkpoints persist model/EMA, optimizer, scheduler, AMP scaler, RNG, hierarchical sampler, shared IMC scaler, data/implementation hashes, gradient audit and truncation counters.

## Sampler, parameters and production integration

`HierarchicalCommodityContractSampler` draws uniformly in the order Commodity → Contract episode → Anchor and supports deterministic seed plus epoch state. RB is held out from scaler fitting, optimization, architecture selection and checkpoint selection.

Trainable parameter counts are:

| Version | Trainable | EMA (separate) |
|---|---:|---:|
| V0 | 4,676,512 | 3,163,648 |
| V1.0 corrected | 6,769,152 | 3,163,904 |
| V1.1 | 8,262,661 | 3,164,928 |

V1.1 is 1.766843× V0, below the 3× stop threshold.

Production integration uses `/data/jepa/v1_1_raw`. All six commodities have real-contract 1m/1d/1w files and the episode calendar. Existing source audit findings are handled explicitly: missing contracts are excluded, missing warmup becomes invalid coordinates, anchors use observed same-contract rows, and extra post-anchor rows never enter online memory. Detailed evidence is in `artifacts/evaluation/v1_1_architecture_development/data_contract_report.json`.

## Deviations from the frozen design

There are no theory or architecture deviations. The development 100-step smoke deliberately limits loaded current-minute episodes to one recent valid episode per Train commodity to bound runtime; it is not an official scaler fit or formal benchmark run. The production loader, dataset and sampler support all supplied episodes. No RB data was read by the smoke.
