# Market-JEPA V1.1 Hierarchical IMC Implementation Plan

## Existing implementations

- V0/design 0.6.1 lives in `market_jepa/`, the top-level configs, and the V0 entrypoints. Its model, dataset, configuration, checkpoints, evaluation, training entrypoint, and numerical behavior remain frozen.
- V1.0 lives in `market_jepa_v1/`. `CrossScaleStateEncoder` is the pooled `[Minute; Daily; Weekly]` prototype; `MarketJEPAV1` owns the market-only EMA target and V0-compatible predictors; `V1Trainer` currently exposes the two reviewed defects: a terminal feedback module with no loss path and validation-H64 checkpoint selection.
- Reused shared components are limited to the V0 `Predictor`, JEPA loss, atomic checkpoint IO, deterministic RNG helpers, and diagnostics. V0 data features and the V1.0 pooled encoder/trainer are not reused by V1.1.

## V1.1 components and superseded behavior

The new `market_jepa_v1_1/` package contains explicit configuration, real-contract data loading, IMC transforms, one shared scaler, hierarchical sampling, separated-memory encoders, the V1.1 JEPA model, fixed-budget trainer/checkpoints, and development reporting. V1.0's pooled attention and terminal feedback are superseded. V1.0 receives only the minimal no-dead-feedback and fixed-budget-checkpoint corrections; V0 is untouched.

```text
Previous 3Y real-contract Weekly IMC [B,156,*]
                    |
                    v
          CommodityState C [B,4,256]
                    |
          +---------+----------+
          |                    |
 Daily lifecycle          Current Weekly
 [B,256,*]                  [B,64,*]
          |                    |
          +---------v----------+
             ContractState K [B,4,256]
                    |
     Minute IMC + causal context [B,512,*]
                    |
        4-layer local Transformer
                    |
          Minute local [B,512,256]
                    |
              cross-attend [C;K]
                    |
       Conditioned minute [B,512,256]
                    |
       separate reads of Minute, K, C
                    |
          Belief tokens [B,8,256]
                    |
               B_t [B,256]
```

## Real-contract and IMC contract

- Production input is `/data/jepa/v1_1_raw`: `contract_episodes.csv` plus per-commodity real-contract 1m/1d/1w CSVs. `main_start_date`, `main_end_date`, and `anchor_end_date` are authoritative. FG/SA/JM/SH/SP are train; RB is held out.
- Historical eligibility is a dataset-initialization contract, configured by `history_week.years`, `history_week.commodity_years`, `history_week.capacity`, `history_week.require_full_history`, and `history_week.series_mode= same_delivery_month`. `years=3` is the default and the formal baseline uses `SH: 2`. Eligibility is computed from the first reliable Weekly timestamp of the candidate episode's explicit `series_key`, never from the commodity-wide first date. This is calendar coverage, not a token-count test. The dataset precomputes `eligible_contracts_by_commodity`; early episodes are removed before anchor creation and sampling, with no retry or shorter-than-configured fallback.
- Minute, Daily, Current Weekly, and every H16/H64/H256 target remain within one `contract_uid`. All horizons must be valid, so an invalid H256 makes the anchor invalid. Historical Weekly may span real contracts only inside the current contract's explicit `series_key` (same commodity and delivery month). It includes closed current-contract pre-main Weekly rows, while Current Weekly begins at `main_start`. Every visible real-contract segment resets IMC and starts with `contract_boundary=1`.
- Capacities are fixed at Minute 512, Daily 256, Current Weekly 64, Historical Weekly 156. For eligible episodes, real missing weeks inside `[main_start-years, main_start)` are left padded and masked; padding does not make an otherwise history-ineligible episode trainable. Padding masks use `True=PAD`. IMC validity uses `True=available` per coordinate. Daily truncation retains the latest 256 and is counted.
- Market feature order is `open_origin_log, high_origin_log, low_origin_log, close_origin_log, close_step_log, oi_origin_log, oi_step_log, volume_fixed_ratio, volume_q20`. Invalid coordinates are numeric zero plus an explicit validity bit.
- Minute origin is the first valid window close/OI and the preceding same-contract 20-bar Volume median. Daily and Current Weekly Price/OI origins are the first observable minute Open/OI at main-start, never the final main-start Daily/Weekly value. Historical Weekly resets at each real-contract segment's first visible Weekly row and uses only same-contract warm-up for step/Volume coordinates; this also keeps current-contract pre-main history causal. No missing baseline is filled from the future.
- For every anchor, Current Daily is closed lifecycle days plus one Daily bar dynamically aggregated from same-contract minutes `<= anchor`; Current Weekly is closed clipped lifecycle weeks plus one similarly causal partial week. Both contexts include `bar_is_partial`. All numeric context coordinates are bounded to `[0,1]`: lifecycle days/256, lost-main days/21, observed intraday bar count/512, weeks-ago/156, contract-age/64 and observed trading days/5. Cached final 1d/1w rows are never used for an in-progress period and must pass end-of-period parity audits.
- Future target Price/OI/fixed Volume reuse the online minute origin. Future Q20 may use earlier target bars. Target input is future minute market IMC and validity only.
- Exactly one scaler is fitted from valid FG/SA/JM/SH/SP training coordinates and frozen for every source and RB. It persists feature order, population/counts, statistics, source hashes, and checksum.

## Model, masks, sampler, and checkpoints

- Four Commodity tokens safely read Historical Weekly. Four Contract tokens first read Commodity state, then Daily and Current Weekly separately with availability-renormalized learned source gates. Fully masked sources contribute zero.
- The full minute token sequence receives additive market, validity, context, and position representations before four local Transformer layers, then cross-attends the eight higher-scale `[C;K]` tokens before final compression.
- Eight Belief tokens read conditioned Minute, Contract, and Commodity sources separately and fuse them with learned source gates. The first token is normalized and projected to 256D.
- Model forward fields are explicit; commodity/contract metadata never enter the representation. `return_intermediates=True` exposes scientific audit tensors without retaining them in normal training.
- Training sampling is deterministic `Commodity -> history-eligible episode -> valid anchor`, uniformly at each level. The sampler consumes only the dataset's precomputed eligibility population and never samples then retries/filters. Checkpoint selection is always `fixed_budget_final`; validation is diagnostic only and cannot create a best checkpoint.
- Historical stitching groups by ISO calendar week inside one `series_key`. If adjacent delivery years overlap, the frozen rule keeps the previous (smaller) `delivery_year` contract and drops the later contract's overlapping week. It never merges OHLCV/OI across real contracts.
- Checkpoints strictly persist model/EMA, optimizer/scheduler/scaler, RNG/sampler state, shared IMC scaler, data and implementation manifests, feature schema, truncation counts, epoch, and global step.

## Regression risks, tests, and blockers

- Primary V0 risks are accidental edits to frozen implementation paths, altered shared loss/predictor behavior, or changes to V0 checkpoint/trainer semantics. The frozen implementation manifest and complete V0 suite are hard gates.
- Tests cover contract boundaries, causal memories, exact IMC equations, same-origin targets, shared-scaler/RB isolation, all-masked safety, hierarchical interventions before belief compression, market-only EMA, all-parameter finite gradients, no dead branch, fixed final checkpointing, exact resume, balanced sampling, and V0 regression.
- Development smoke uses realistic B=2/B=8 capacities on CUDA when available. The 100-step real-data smoke and shared-scaler fit run only if every formal Train commodity has at least one eligible episode. No full 50-epoch training or RB evaluation is started.
- Observed source audit errors (missing old contracts/prefixes/future tails and nonpositive OI) are handled by episode/anchor filtering or validity masks and are reported. They are not silently reinterpreted. A new unrecoverable source-semantic defect would be reported as `DATA_BLOCKER` while architecture work continues.
- Eligibility audit artifacts are `history_week_eligibility.csv` and `history_week_eligibility.json`, containing per-lineage coverage facts and per-contract decisions. `contract_lineage_audit.csv/json` preserve every supplied lineage transition, overlap/gap counts, and the explicit overlap disposition. With the formal `SH: 2` override, every current Train commodity has a nonempty eligible population; changing the requirement still never triggers an automatic fallback.
