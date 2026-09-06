# Market-JEPA V1.1 — Hierarchical IMC implemented architecture

## Status and scope

V1.1 is implemented as a separate `market_jepa_v1_1` package. V0/design 0.6.1 source, configuration, checkpoint contract, evaluation and training entrypoint are unchanged. V1.0 receives only the two required corrections: its terminal no-loss feedback module is no longer instantiated, and validation JEPA H64 cannot select a checkpoint.

This implementation stage does not start the formal 50-epoch run or RB evaluation.

The separate formal entrypoint is `train_market_jepa_v1_1.py`; the development
entrypoint remains bounded smoke only. Formal training runs all Train-only data
hard gates, loads every eligible FG/SA/JM/SH/SP episode, fits one deterministic
shared scaler, rebuilds a new scaled dataset, and uses the hierarchical sampler
for exactly 50 epochs. It writes only `last.pt` under
`artifacts/training/v1_1_formal/checkpoints`, supports strict `--resume`, and
never constructs a validation loader or opens an RB bar file.

The commodity population is read from `data.train_commodities`, and the held-out
commodity from `data.held_out_commodity`; dataset eligibility, audits, shared
scaler validation and hierarchical sampling all use those configured values.
The current frozen run remains FG/SA/JM/SH/SP with RB held out. Newly configured
commodities inherit `history_week.years=3` unless they receive an explicit
`history_week.commodity_years` override; `SH: 2` is an intentional current-run
override rather than a fallback.

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

`contract_episodes.csv` provides explicit `contract_uid`, `delivery_year`, `delivery_month`, `series_key`, `main_start_date`, `main_end_date`, `anchor_end_date` and role. Each real contract is an independent episode. Minute windows, Daily lifecycle, Current Weekly lifecycle and every JEPA target remain inside that `contract_uid`.

At each minute anchor:

- Minute is the latest 512 same-contract rows, left padded.
- Daily is completed lifecycle days plus exactly one current partial bar aggregated from same-contract minute rows with `datetime <= anchor` and the anchor `trading_date`.
- Current Weekly is completed lifecycle weeks plus exactly one partial lifecycle week aggregated only through anchor.
- Historical Weekly is the configured interval `[main_start-years, main_start)`, clipped to 156 tokens and selected strictly from the current contract's explicit `series_key`. It is not constructed from chronological main-contract episodes. Closed pre-main Weekly rows of the current `contract_uid` form the most recent historical segment. Each visible real contract resets IMC and marks its first token with `contract_boundary=1`.
- Anchors after loss of main status are bounded by `anchor_end_date`; the supplied package uses the episode calendar's 15-trading-day extension policy.

Cached 1d/1w bars are never used as the current forming bar. Weekly lifecycle bars are reconstructed from contract-local closed Daily lifecycle bars so a main-start week cannot import pre-main days. `bar_is_partial` and causal progress fields are explicit context features. Context numeric values are bounded to `[0,1]`: days since main/256, days since lost-main/21, observed intraday bar count/512, weeks ago/156, contract age/64 and observed trading days/5. Intraday progress therefore does not use wall-clock gaps across night sessions.

The implementation requires all H16/H64/H256 targets to exist. An anchor without H256 inside its real-contract minute array is excluded; no per-horizon fallback can cross a roll.

Before anchors are built, the dataset precomputes `eligible_contracts_by_commodity`. Formal configuration is:

```yaml
history_week:
  years: 3
  commodity_years:
    SH: 2
  capacity: 156
  require_full_history: true
  series_mode: same_delivery_month
```

For each lineage independently, the first reliable Weekly row defines the coverage origin. An episode is eligible exactly when `main_start >= first_lineage_weekly_date + required_years`, where `required_years` is its explicit commodity override or the global default. Capacity is not used as a proxy for coverage: an early contract cannot enter training with mostly PAD, while genuine missing weeks inside an eligible configured calendar window remain PAD plus mask. Sampling receives only this prefiltered contract population and never retries an ineligible draw. `series_key` is data-selection metadata only; it remains absent from model construction and forward representation.

Within one lineage, Weekly rows are keyed by ISO calendar week. When adjacent delivery years overlap, the previous (smaller) `delivery_year` contract supplies that week; the later contract's row is discarded. No OHLCV/OI aggregation occurs across contracts. The supplied lineage audit reports 17 such one-week FG transitions: six in FG-01, six in FG-05, and five in FG-09. FG-09 also contains one reported gap week, represented naturally by missing time rather than fabricated data.

Historical market tokens read only the selected Weekly rows. They do not consult Daily bars; their closed-week progress context is the completed value `1`. Current Weekly retains its separate causal Daily/minute lifecycle construction.

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

Minute `P0` is the first valid window Close and `OI0` is its OI. Its fixed Volume baseline is the median of exactly 20 bars immediately before the window. Daily and Current Weekly Price/OI origins are fixed for the whole lifecycle at the first observable main-start minute Open/OI; their Volume baselines use 20 closed same-contract Daily/Weekly bars before main start. Each Historical Weekly real-contract segment uses its first visible Weekly Close/OI as fixed origin and only preceding rows from that same contract for step and Volume warm-up. This prevents a current contract's pre-main history from reading its future main-start origin and prevents all cross-contract deltas.

Unavailable or nonpositive baselines produce numeric zero plus coordinate validity false; no epsilon and no future fill are used. A single post-IMC scaler is fitted jointly from FG/SA/JM/SH/SP online-memory snapshots. It stores means, standard deviations, feature counts, per-source population counts, exact feature order, fitted commodities and a SHA-256 checksum. The fitter rejects RB and rejects a population missing any of the five Train commodities. Historical-memory cache entries are always raw IMC; the frozen scaler is applied only when a sample is returned. A read-only `scaler` property plus `set_scaler()` prevents the former stale-cache lifecycle bug.

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

`HierarchicalCommodityContractSampler` draws uniformly in the order Commodity → eligible Contract episode → Anchor and supports deterministic seed plus epoch state. It validates that every episode in its hierarchy belongs to the dataset's precomputed eligible set. RB is held out from scaler fitting, optimization, architecture selection and checkpoint selection.

Trainable parameter counts are:

| Version | Trainable | EMA (separate) |
|---|---:|---:|
| V0 | 4,676,512 | 3,163,648 |
| V1.0 corrected | 6,769,152 | 3,163,904 |
| V1.1 | 8,262,661 | 3,164,928 |

V1.1 is 1.766843× V0, below the 3× stop threshold.

Production integration uses `/data/jepa/v1_1_raw`. All six commodities have real-contract 1m/1d/1w files and the episode calendar. Existing source audit findings are handled explicitly: missing contracts are excluded, missing warmup becomes invalid coordinates, anchors use observed same-contract rows, and extra post-anchor rows never enter online memory. The development hard gate scans every Train minute row for Price/OI/Volume/OHLC validity and contract coverage, verifies every available minute→Daily cache row, and verifies every Daily→Weekly cache row. Nonpositive OI is not silently accepted: its IMC validity is false. Detailed evidence is in `artifacts/evaluation/v1_1_architecture_development/data_contract_report.json`.

The configured eligibility audit over the supplied production package is:

| Commodity | Required years | First reliable Weekly | Total episodes | Filtered | Eligible | Earliest eligible episode | Earliest eligible main-start |
|---|---:|---:|---:|---:|---:|---|---:|
| FG | 3 | 2017-12-22 | 33 | 16 | 17 | FG202109 | 2021-04-08 |
| SA | 3 | 2019-12-06 | 21 | 10 | 11 | SA202309 | 2023-03-27 |
| JM | 3 | 2017-12-29 | 38 | 21 | 17 | JM202109 | 2021-04-21 |
| SH | 2 | 2023-09-15 | 12 | 10 | 2 | SH202605 | 2026-02-25 |
| SP | 3 | 2018-11-30 | 34 | 17 | 17 | SP202209 | 2022-04-08 |

Every formal Train commodity has a nonempty lineage-qualified population. The implementation does not silently alter either the global default or the SH override and does not exclude a missing commodity from the uniform benchmark. Per-contract evidence includes `series_key`, delivery metadata, lineage bounds and required years. The development artifact directory contains both history eligibility and contract-lineage CSV/JSON reports.

## Deviations from the frozen design

There are no theory or architecture deviations. Historical contract selection follows explicit `series_key`, with the user-frozen overlap rule retaining the previous delivery year. The configured SH eligibility window is two calendar years while every unlisted commodity inherits the three-year default; all retain capacity 156 and explicit PAD/mask semantics. The development smoke limits loaded current-minute episodes to recent eligible episodes per Train commodity to bound runtime; it is not a formal benchmark run. No RB data is used for fitting. Historical visibility is causal by timestamp and independent of role labels.
