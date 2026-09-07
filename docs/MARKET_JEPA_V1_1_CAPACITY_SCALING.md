# Market-JEPA V1.1 Controlled Capacity Scaling

## Scope and claim boundary

This work defines four capacities of the same reviewed V1.1 hierarchy. It is
not a new architecture and makes no claim that a larger profile improves held-out
generalization. Only a future controlled Train-only run followed by the frozen RB
evaluation can answer that question.

The only scaled fields are `d_model`, `num_heads`, `ffn_dim`, the online/EMA
minute transformer depth, and predictor hidden width. The predictor remains
`d_model -> 2*d_model -> d_model` for every profile.

## Frozen semantics

All profiles use the single `market_jepa_v1_1.model.MarketJEPAV11` class and the
same information flow:

```text
Historical Weekly -> Commodity State
Commodity State + (Daily, Current Weekly) -> Contract State
Minute Local + (Commodity State, Contract State) -> Conditioned Minute
Conditioned Minute + Contract State + Commodity State -> Belief
```

Commodity/Contract/Belief slot counts stay at 4/4/8. Scaling their count would
change semantic capacity rather than only representation width. Minute, Daily,
Current Weekly and Historical Weekly capacities stay at 512/256/64/156 so every
profile receives identical information.

IMC formulas and features, validity masks, same-origin targets, contract boundary
reset, partial Daily/Weekly construction, same-del-month-lineage stitching,
FG/SA/JM/SP 3-year eligibility, SH 2-year eligibility, shared scaler, balanced
Commodity -> Contract -> Anchor sampling, and H16/H64/H256 JEPA losses are frozen.

## Profiles and exact parameters

| Profile | d | heads | FFN | minute layers | predictor hidden | trainable | EMA | resident |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 256 | 8 | 1024 | 4 | 512 | 8,262,661 | 3,164,928 | 11,427,589 |
| M | 384 | 6 | 1536 | 6 | 768 | 22,086,917 | 10,655,616 | 32,742,533 |
| L | 512 | 8 | 2048 | 8 | 1024 | 45,518,853 | 25,230,848 | 70,749,701 |
| XL | 768 | 12 | 3072 | 8 | 1536 | 102,291,461 | 56,720,640 | 159,012,101 |

EMA is frozen, market-only, and excluded from the trainable target. Resident
parameters are trainable plus all non-trainable parameters; registered positional
buffers are not parameters.

S retains the reviewed parameter count and 241-entry state-dict schema. The
pre-scaling S development checkpoint loads strictly; new checkpoints add explicit
size/dimension/count metadata, while legacy S loading remains supported.

## Gradient checkpointing and 16GB engineering policy

L and XL checkpoint only the online minute transformer using
`torch.utils.checkpoint.checkpoint(use_reentrant=False, preserve_rng_state=True)`.
The EMA target runs under `no_grad` and is never checkpointed. S and M leave
checkpointing disabled. With dropout disabled, checkpointed and ordinary paths
produce identical forward values; backward connectivity is checked separately.

On the RTX 4060 Ti 16GB, all four profiles completed FP16-autocast full
forward, H16/H64/H256 JEPA loss, backward, optimizer step and EMA update at the
candidate-list upper bound of micro-batch 64. XL used 10,591,282,176 allocated
bytes and 11,108,614,144 reserved bytes. Therefore the selected engineering
configuration is `64 x 2 = 128` for every profile. This is a bounded synthetic
capacity smoke, not an epoch throughput or convergence result.

No reliable FLOP counter is installed, so FLOPs are deliberately not reported.
Measured forward/backward/step wall times are preserved in the capacity artifacts.

## Configuration and checkpoint rules

The profile files extend the reviewed S configuration and override only declared
capacity/runtime fields. Config expansion is materialized in formal
`config_snapshot.yaml`; implementation hashes include all profile files.

New checkpoints record `model_size`, `d_model`, `num_heads`, `ffn_dim`,
`minute_layers`, `predictor_hidden_dim`, and exact trainable parameter count.
Resume validates the complete config, implementation/data/scaler manifests,
model size and parameter count. Cross-size resume is rejected before loading.

## Future scientific protocol

Run in order: S (~8M), M (~22M), L (~46M), XL (~102M). Use the same training
population, data passes/epochs, effective batch 128, optimizer objective,
checkpoint endpoint and frozen evaluation. RB must not choose model size, learning
rate or architecture. The first question is whether increased representation
capacity improves held-out-commodity generalization—not per-size hyperparameter
optimization.
