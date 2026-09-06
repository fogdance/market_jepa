# Market-JEPA V1.1 — Hierarchical IMC Architecture Design

**Document status:** Reviewed and frozen architecture specification  
**Version:** V1.1  
**Purpose:** Preserve the V1.1 architecture decisions reached after reviewing the V1 prototype against the actual futures data structure.  
**Relationship to earlier versions:**  
- **V0 / design 0.6.1:** late-fusion scientific baseline  
- **V1.0 prototype:** generic cross-scale state-token interaction prototype  
- **V1.1:** hierarchical commodity → contract → minute architecture integrated with IMC

---

# 1. Why V1.1 exists

The V1.0 prototype correctly addressed one major V0 limitation:

\[
\boxed{
\text{Cross-scale interaction must happen before final compression}
}
\]

However, code review and clarification of the real futures data organization showed that the market hierarchy is more structured than the generic V1.0 design assumed.

The real data hierarchy is:

\[
\boxed{
\text{Commodity}
\supset
\text{Contract}
\supset
\text{Minute Market Process}
}
\]

For example, when the current sample belongs to `FG601`:

- minute data belong to the real contract `FG601`,
- daily data describe the lifecycle of `FG601`,
- current weekly data describe the slower lifecycle of `FG601`,
- historical weekly data represent the commodity FG's prior market history, across earlier real contracts.

Therefore V1.1 changes the architectural principle from generic symmetric multi-scale interaction to:

\[
\boxed{
\text{Commodity Memory}
\rightarrow
\text{Contract State}
\rightarrow
\text{Minute State}
\rightarrow
\text{Belief}
}
\]

This is the core V1.1 change.

---

# 2. Scientific role of V0, V1.0, and V1.1

## V0

V0 remains the official late-fusion baseline:

\[
M\rightarrow Z_m,\quad
C\rightarrow Z_c,\quad
D\rightarrow Z_d,\quad
W\rightarrow Z_w
\]

followed by:

\[
[Z_m,Z_c,Z_d,Z_w]\rightarrow Z_{market}
\]

Its main limitation is:

\[
\boxed{
\text{compress first, condition later}
}
\]

## V1.0 prototype

V1.0 introduced state tokens and bidirectional cross-scale feedback:

\[
S\leftarrow T,\qquad
T\leftarrow S,\qquad
S\leftarrow T
\]

where:

\[
T=[M;D;W]
\]

This demonstrated that Daily/Weekly information could alter minute representation before final belief compression.

However, the prototype treated Minute, Daily, and Weekly as largely symmetric token pools.

## V1.1

V1.1 adopts the actual market hierarchy:

\[
\boxed{
W^H \rightarrow C
}
\]

\[
\boxed{
(D,W^C)\mid C \rightarrow K
}
\]

\[
\boxed{
M\mid(C,K)\rightarrow M'
}
\]

\[
\boxed{
(M',K,C)\rightarrow B_t
}
\]

where:

- \(W^H\): previous 3-year commodity weekly memory,
- \(W^C\): current-contract weekly lifecycle,
- \(D\): current-contract daily lifecycle,
- \(M\): current-contract minute window,
- \(C\): commodity state,
- \(K\): contract state,
- \(B_t\): final market belief.

---

# 3. Real-contract semantics

V1.1 does **not** treat an 8-year commodity history as one continuously tradable contract.

For FG, for example, the training population may include:

\[
FG501,\ FG505,\ FG509,\ FG601,\ FG605,\ FG609,\ldots
\]

Each real main contract is an independent episode.

Therefore:

\[
\boxed{
\text{Minute and current-contract Daily/Weekly must never cross a real contract boundary}
}
\]

This is particularly important for:

\[
Price,\ OI,\ Volume
\]

because Open Interest and Volume cannot be made economically continuous across contract rolls by ordinary price back-adjustment.

---

# 4. Four V1.1 memory blocks

## 4.1 Current Contract Minute Memory

Fixed capacity:

\[
\boxed{N_M=512}
\]

For an anchor inside FG601:

\[
M_t=
\{\text{up to 512 most recent 1-minute bars from FG601}\}
\]

Rules:

1. Same real contract only.
2. Never borrow bars from FG509 or FG605.
3. If fewer than 512 valid bars exist, left-pad and mask.
4. Historical bars before the contract became main may be used if they belong to the same real contract and were already observable at the anchor time.
5. No future minute bar may enter the online input.

Semantic role:

\[
\boxed{
\text{Current Contract Micro State}
}
\]

## 4.2 Current Contract Daily Lifecycle Memory

Fixed capacity:

\[
\boxed{N_D=256}
\]

The meaningful daily lifecycle begins when the real contract becomes the main contract.

For a given anchor:

\[
D_t=
\{\text{current contract daily bars from main-start through anchor}\}
\]

If the contract has already lost main status, valid anchors may continue for at most:

\[
\boxed{\text{3 weeks after losing main status}}
\]

Rules:

1. Current real contract only.
2. Fixed tensor capacity of 256.
3. Missing history is represented by PAD + mask.
4. Do not fill missing capacity with prior-contract daily bars.
5. If a valid lifecycle ever exceeds 256 trading days, retain the most recent 256 valid lifecycle bars for the anchor; such truncations must be counted and reported.

Semantic role:

\[
\boxed{
\text{Current Contract Lifecycle State}
}
\]

## 4.3 Current Contract Weekly Lifecycle Memory

Fixed capacity:

\[
\boxed{N_{WC}=64}
\]

This memory contains weekly bars belonging only to the current real contract.

Rules:

1. Same current contract only.
2. No cross-contract continuation.
3. Fixed capacity 64.
4. PAD + mask when insufficient.
5. No future weekly information.

Semantic role:

\[
\boxed{
\text{Current Contract Slow Lifecycle}
}
\]

## 4.4 Previous 3-Year Commodity Weekly Memory

Fixed capacity:

\[
\boxed{N_{WH}=156}
\]

This memory contains up to approximately 3 years of already-observed commodity weekly history.

For a current contract with main-start time \(T_0\):

\[
W_t^H
=
\text{commodity weekly history in }
[T_0-3Y,\ T_0)
\]

subject to actual data availability.

For newly listed commodities such as SH, the valid history may be shorter.

Rules:

1. Never invent missing history.
2. Missing historical slots are PAD + mask.
3. Historical weekly memory is strictly causal.
4. A historical sample from 2019 may only see information available up to that historical time.
5. The fixed capacity remains 156 regardless of whether the commodity has 1 year, 3 years, or 8 years of total historical data.

Semantic role:

\[
\boxed{
\text{Commodity Historical Memory}
}
\]

---

# 5. Why the weekly memory is split

The current contract weekly path and the historical commodity weekly memory have different meanings.

\[
W^C
\]

answers:

> What has the current real contract been doing at a slow weekly scale?

while:

\[
W^H
\]

answers:

> What did this commodity look like during the previous three years before the current contract lifecycle began?

Therefore V1.1 must not collapse them into a single anonymous weekly token sequence without source identity.

They are separate memory sources.

---

# 6. Contract boundaries inside historical weekly memory

Historical 3-year weekly memory can span multiple real contracts.

These must **not** be interpreted as a continuous OI or Volume sequence.

Every historical real-contract segment must have explicit boundary semantics.

Define:

\[
boundary_i=
\begin{cases}
1,&\text{first weekly token of a new real contract segment}\\
0,&\text{otherwise}
\end{cases}
\]

At every contract boundary:

\[
\boxed{
IMC\ Price/OI/fixed\ Volume\ coordinates\ reset
}
\]

Do not compute:

\[
\log\frac{OI_{\text{new contract}}}
{OI_{\text{old contract}}}
\]

as a market OI transition.

The same rule applies to any fixed-origin Price or Volume coordinate.

---

# 7. IMC integration

V1.1 uses IMC as the market-coordinate layer.

Its purpose is to remove commodity-specific multiplicative scales while preserving relative market behavior.

The core invariance remains:

\[
\boxed{
T(aP,bOI,cV)=T(P,OI,V)
}
\]

for positive multiplicative scale constants \(a,b,c\).

IMC is upstream of the neural representation.

V1.1 does not replace IMC and does not ask the neural network to relearn commodity scale normalization.

---

# 8. Minute IMC

Let \(t_0\) be the first valid minute bar in the current 512-bar observation window.

## Price

For OHLC, use one shared fixed origin \(P_{t_0}\):

\[
X_{\tau}^{O,M}
=
\log\frac{O_\tau}{P_{t_0}}
\]

\[
X_{\tau}^{H,M}
=
\log\frac{H_\tau}{P_{t_0}}
\]

\[
X_{\tau}^{L,M}
=
\log\frac{L_\tau}{P_{t_0}}
\]

\[
X_{\tau}^{C,M}
=
\log\frac{C_\tau}{P_{t_0}}
\]

The single-step close movement is also provided:

\[
\boxed{
\Delta P_\tau^M
=
\log\frac{C_\tau}{C_{\tau-1}}
}
\]

## Open Interest

Use the first valid minute OI as origin:

\[
\boxed{
X_\tau^{OI,M}
=
\log\frac{OI_\tau}{OI_{t_0}}
}
\]

Also provide the single-step OI change:

\[
\boxed{
\Delta OI_\tau^M
=
\log\frac{OI_\tau}{OI_{\tau-1}}
}
\]

Thus:

- \(X^{OI}\) preserves cumulative OI path,
- \(\Delta OI\) exposes sudden OI changes directly.

## Volume

Define the fixed-origin minute volume baseline from the 20 bars before \(t_0\):

\[
M_0^{V,M}
=
Median(V_{t_0-20:t_0-1})
\]

Then:

\[
\boxed{
X_\tau^{V,M}
=
\frac{V_\tau}{M_0^{V,M}}
}
\]

This preserves the entire relative Volume path.

Local Volume surprise remains:

\[
\boxed{
Q_\tau^{V,M}
=
\frac{V_\tau}
{Median(V_{\tau-20:\tau-1})}
}
\]

Therefore:

- fixed-origin \(X^V\): sustained activity,
- rolling \(Q_{20}\): local N-fold surprise.

---

# 9. Daily IMC

The Daily branch represents the current contract lifecycle.

Let:

\[
t_D^0
=
\text{first daily bar of the main-contract lifecycle}
\]

The fixed origin remains the same throughout the current real contract lifecycle.

## Price

\[
X_d^{P,D}
=
\log\frac{P_d}{P_{t_D^0}}
\]

## Open Interest

\[
X_d^{OI,D}
=
\log\frac{OI_d}{OI_{t_D^0}}
\]

## Volume

Use pre-main warm-up data from the same real contract:

\[
M_0^{V,D}
=
Median(
V_{\text{20 daily bars before }t_D^0}
)
\]

Then:

\[
X_d^{V,D}
=
V_d/M_0^{V,D}
\]

and:

\[
Q_d^{V,D}
=
\frac{V_d}
{Median(V_{d-20:d-1})}
\]

The pre-main bars used to compute the baseline are warm-up only and do not become lifecycle tokens unless they are otherwise part of the formal input definition.

---

# 10. Current-contract Weekly IMC

The current-contract weekly branch follows the same fixed-origin lifecycle logic as Daily.

Let:

\[
t_W^0
=
\text{first weekly bar of the current contract lifecycle}
\]

Then:

\[
X_w^{P,WC}
=
\log\frac{P_w}{P_{t_W^0}}
\]

\[
X_w^{OI,WC}
=
\log\frac{OI_w}{OI_{t_W^0}}
\]

and corresponding fixed-origin and Q20 Volume coordinates are computed causally.

No coordinate may cross into another real contract.

---

# 11. Historical Weekly IMC

Historical weekly memory contains multiple real contract segments.

For each historical real contract \(c\), define its own origins:

\[
P_0^{(c)},\quad
OI_0^{(c)},\quad
M_0^{V,(c)}
\]

and compute IMC only inside that contract segment.

Therefore:

\[
\boxed{
IMC\ resets\ at\ every\ historical\ contract\ boundary
}
\]

The boundary flag tells the network that a reset corresponds to a new real contract, not to a market shock.

---

# 12. Missing IMC baselines

If a required baseline is unavailable, V1.1 must not create future leakage or invent data.

Examples:

- fewer than 20 prior bars for a Volume fixed-origin baseline,
- OI unavailable or invalid,
- insufficient historical contract warm-up.

In such cases:

1. the corresponding transformed value may be filled with a neutral numeric value such as zero,
2. a dedicated validity mask must mark the coordinate as unavailable,
3. future data must never be used to back-fill the baseline.

Formally:

\[
\boxed{
\text{Missing baseline}
\Rightarrow
\text{masked information, not future-assisted estimation}
}
\]

---

# 13. Shared numerical scaler after IMC

IMC removes absolute commodity scale, but feature magnitudes may still differ numerically.

A single shared affine scaler may therefore be fitted after IMC:

\[
x'=\frac{x-\mu}{\sigma}
\]

The scaler must be fitted using only training commodities:

\[
\boxed{
FG+SA+JM+SH+SP
}
\]

There is one shared scaler.

Forbidden:

\[
\boxed{
\text{per-commodity scaler}
}
\]

RB must use the same frozen training scaler without refitting.

---

# 14. Causal context features

V1.1 allows context variables that are genuinely known at the anchor time.

## Minute context

Examples:

- time-of-day sin/cos,
- weekday,
- elapsed/delta minutes.

## Daily lifecycle context

Examples:

\[
days\_since\_main
\]

\[
is\_main
\]

and, when already no longer main:

\[
days\_since\_lost\_main
\]

Forbidden:

\[
days\_until\_lost\_main
\]

because that is future information.

## Weekly context

Examples:

\[
weeks\_ago
\]

\[
contract\_boundary
\]

\[
contract\_age
\]

and memory-source identity:

\[
historical
\quad\text{vs}\quad
current\_contract
\]

These are causal context, not commodity identity.

---

# 15. Commodity identity in V1.1

V1.1's first universal benchmark does **not** provide:

- commodity ID,
- commodity embedding,
- symbol embedding,
- commodity-specific encoder,
- commodity-specific predictive head.

This is an experimental isolation condition.

It is **not** a permanent claim that commodity identity is illegitimate information.

The first benchmark asks:

\[
\boxed{
\text{Can a shared representation learn reusable market structure without explicit commodity identity?}
}
\]

A later production model may introduce explicit controlled commodity conditioning as a separate experiment.

---

# 16. Why V1.1 rejects one pooled attention over all scale tokens

The V1.0 prototype used:

\[
T=[M;D;W]
\]

followed by one attention operation over all tokens.

V1.1 does not use this as its main architecture.

Reason:

- Minute may have 512 tokens,
- Daily may have up to 256,
- current Weekly up to 64,
- historical Weekly up to 156.

If all tokens compete in one softmax, scale importance becomes entangled with token count.

V1.1 instead treats each memory source as a separate information source.

---

# 17. Stage 1 — Commodity State

Historical 3-year weekly memory:

\[
W^H
\]

is read by:

\[
\boxed{K_C=4}
\]

learned Commodity State Tokens:

\[
C^{(0)}
\in
\mathbb{R}^{4\times256}
\]

The commodity state update is conceptually:

\[
C
=
C^{(0)}
+
MHA(
LN(C^{(0)}),
LN(W^H),
LN(W^H)
)
\]

followed by:

\[
C
\leftarrow
C+FFN(LN(C))
\]

Result:

\[
\boxed{
C_t
=
CommodityState
}
\]

The four latent slots preserve more information than compressing the entire 3-year history into one vector immediately.

---

# 18. Stage 2 — Current Contract State

Use:

\[
\boxed{K_K=4}
\]

learned Contract State Tokens:

\[
K^{(0)}
\in
\mathbb{R}^{4\times256}
\]

First condition the contract representation on commodity state:

\[
K_1
=
K^{(0)}
+
Attn(K^{(0)},C,C)
\]

Then read the two contract-lifecycle memories separately:

\[
A_D
=
Attn(K_1,D,D)
\]

\[
A_{WC}
=
Attn(K_1,W^C,W^C)
\]

These two memories do not share one token-level softmax.

Use learned or state-conditioned source gates:

\[
(\alpha_D,\alpha_{WC})
=
softmax(g_D,g_{WC})
\]

and update:

\[
\boxed{
K_2
=
K_1+
\alpha_DA_D+
\alpha_{WC}A_{WC}
}
\]

followed by FFN.

Result:

\[
\boxed{
K_t
=
CurrentContractState
}
\]

Thus:

\[
\boxed{
K_t=f(D,W^C\mid C_t)
}
\]

---

# 19. Stage 3 — Higher-scale conditioning of Minute tokens

Minute Market + causal Minute Context first form local minute tokens:

\[
M^{local}
\]

using the Minute local Transformer.

V1.1 keeps:

- \(d_{model}=256\),
- 8 heads,
- FFN 1024,
- 4 local Transformer layers,

unless changed later in a separate capacity experiment.

The important point is that the entire minute token sequence survives local encoding.

Construct higher-scale state:

\[
H=[C;K]
\]

Then every valid minute token may read the higher-scale state:

\[
M'
=
M^{local}
+
Attn(
M^{local},
H,
H
)
\]

followed by FFN.

Therefore:

\[
\boxed{
M'
=
f(
MinutePath
\mid
ContractState,\ CommodityState
)
}
\]

This is the central V1.1 mechanism.

---

# 20. Stage 4 — Final Belief State

Use:

\[
\boxed{K_B=8}
\]

learned Belief Tokens.

They read three information sources separately:

\[
A_M=Attn(B,M',M')
\]

\[
A_K=Attn(B,K,K)
\]

\[
A_C=Attn(B,C,C)
\]

Then:

\[
(\beta_M,\beta_K,\beta_C)
=
softmax(g_M,g_K,g_C)
\]

and:

\[
B
\leftarrow
B+
\beta_MA_M+
\beta_KA_K+
\beta_CA_C
\]

followed by FFN.

The first belief token is projected:

\[
\boxed{
B_t
=
Linear(LN(B_1))
\in
\mathbb{R}^{256}
}
\]

This is the final online market belief used by JEPA predictors.

---

# 21. V1.1 information-flow diagram

```text
Previous 3Y Commodity Weekly IMC
               |
               v
      Commodity State C (4)
               |
               v
        +-------------+
        |             |
 Current Daily    Current Weekly
        |             |
        +------v------+
          conditioned
             by C
               |
               v
       Contract State K (4)
               |
        C -----+
               |
               v
 Current Minute IMC + Context
               |
      Local Transformer
               |
               v
       Minute Tokens M
               |
        conditioned by
            [C, K]
               |
               v
      Conditioned M'
               |
       +-------+-------+
       |               |
       K               C
       |               |
       +-------v-------+
          Belief Tokens (8)
               |
               v
           B_t 256D
```

The hierarchy is:

\[
\boxed{
CommodityMemory
\rightarrow
ContractState
\rightarrow
MinuteConditionalRepresentation
\rightarrow
Belief
}
\]

---

# 22. JEPA objective remains unchanged

V1.1 changes representation architecture, not the core JEPA objective.

For horizons:

\[
H\in\{16,64,256\}
\]

the online belief predicts future target representation:

\[
\boxed{
B_t
\xrightarrow{P_H}
Z_{t+H}^{EMA}
}
\]

The predictor architecture should remain comparable to V0 where practical.

Do not add in V1.1 baseline:

- CDF loss,
- CME loss,
- RSSM,
- Actor,
- Critic,
- reward,
- PnL,
- regime labels,
- manually defined Price/OI quadrants.

---

# 23. Same-origin future IMC target

The JEPA future target must use the **same minute observation origin** as the online minute input.

Let the current minute observation window begin at \(t_0\).

The reference state is:

\[
R_t=
(
P_{t_0},
OI_{t_0},
M_0^{V,M}
)
\]

For a future target bar \(\tau>t\):

\[
\boxed{
P_\tau^*
=
\log\frac{P_\tau}{P_{t_0}}
}
\]

\[
\boxed{
OI_\tau^*
=
\log\frac{OI_\tau}{OI_{t_0}}
}
\]

\[
\boxed{
V_\tau^*
=
\frac{V_\tau}{M_0^{V,M}}
}
\]

The future target must **not** reset its own fixed origin.

Reason:

If the target were re-zeroed at the first future bar, the representation would discard the displacement from the current market state to the future state.

V1.1 therefore freezes:

\[
\boxed{
\text{online minute IMC and future target IMC share the same fixed origin}
}
\]

---

# 24. Future Volume Q20

For a future bar \(\tau\):

\[
Q_\tau^V
=
\frac{V_\tau}
{Median(V_{\tau-20:\tau-1})}
\]

The denominator may include preceding bars inside the target interval because it defines the target representation itself.

These values must never be exposed to the online encoder.

Therefore this does not constitute online leakage.

---

# 25. JEPA targets must never cross a real contract boundary

For any anchor and horizon \(H\):

if the future target crosses from the current real contract to another contract, the anchor is invalid for that horizon.

Formally:

\[
\boxed{
contract(anchor)=contract(target_{H})
}
\]

must hold.

This prevents artificial:

- OI jumps,
- Volume jumps,
- price splice effects,

from entering the JEPA transition target.

---

# 26. EMA target remains future market-only

The target encoder remains market-only.

Forbidden target inputs:

- future clock context,
- future weekday,
- future session context,
- future Daily,
- future Current Weekly,
- future Historical Weekly,
- commodity ID.

The target remains conceptually:

\[
\boxed{
Z_{future}^{EMA}
=
EMAEncoder(
FutureMinuteMarketIMC
)
}
\]

The online model may use already-known Commodity/Contract history to predict the future market-only representation.

---

# 27. Checkpoint selection

V1.1 explicitly rejects validation JEPA loss as an epoch-selection criterion.

Because the EMA target representation changes across epochs:

\[
Z_{target}^{EMA}(epoch_i)
\neq
Z_{target}^{EMA}(epoch_j)
\]

absolute JEPA losses across epochs are not directly comparable as a fixed validation objective.

Therefore formal checkpoint selection is:

\[
\boxed{
\text{fixed training budget}
\rightarrow
\text{final endpoint checkpoint}
}
\]

For a 50-epoch experiment:

\[
\boxed{
epoch49\ / last.pt
}
\]

Validation may be logged diagnostically, but must not choose the checkpoint.

---

# 28. V1.0 dead terminal feedback is removed

The reviewed V1.0 prototype contained an unused terminal token-feedback stage.

The correct generic interaction sequence is:

\[
S_1\leftarrow T_0
\]

\[
T_1\leftarrow S_1
\]

\[
S_2\leftarrow T_1
\]

\[
B=Head(S_2)
\]

and stop.

V1.1's hierarchical design supersedes that prototype structure entirely.

No trainable module may execute if its outputs cannot affect the formal training loss.

A gradient-connectivity test must verify that all intended trainable V1.1 parameters receive finite gradients.

---

# 29. Multi-commodity training sampler

Because commodities have very different history lengths, raw-anchor concatenation would overweight long-history commodities.

V1.1 freezes the first universal-model sampler as:

\[
\boxed{
Commodity
\rightarrow
Contract
\rightarrow
Anchor
}
\]

## Step 1

Choose training commodity uniformly:

\[
Commodity
\sim
Uniform(FG,SA,JM,SH,SP)
\]

## Step 2

Choose a valid main-contract episode uniformly within that commodity:

\[
Contract
\sim
Uniform(contracts_{commodity})
\]

## Step 3

Choose a valid anchor uniformly within that contract episode:

\[
Anchor
\sim
Uniform(valid\ anchors_{contract})
\]

This prevents:

- FG/JM from dominating because they have longer histories,
- SH from becoming nearly irrelevant because it was listed recently.

---

# 30. Held-out RB semantics

RB is held out from optimization.

RB must not participate in:

- model-weight training,
- shared-scaler fitting,
- architecture tuning,
- IMC formula selection,
- hyperparameter selection.

However, when evaluating an RB contract, the model may use the RB market history that would have been known at that anchor:

- up to 512 same-contract minute bars,
- current RB contract Daily lifecycle,
- current RB contract Weekly lifecycle,
- up to previous 3 years of RB commodity Weekly history.

This is valid because:

\[
\boxed{
\text{unseen commodity}
\neq
\text{blind commodity}
}
\]

The model parameters have never been optimized using RB, but the model may observe RB's already-existing historical market data exactly as it would in live trading.

---

# 31. V1.1 fixed architecture constants

The first V1.1 baseline freezes:

| Component | Value |
|---|---:|
| Current-contract Minute capacity | 512 |
| Current-contract Daily capacity | 256 |
| Current-contract Weekly capacity | 64 |
| Previous commodity Weekly capacity | 156 |
| Historical window | 3 years |
| Commodity State Tokens | 4 |
| Contract State Tokens | 4 |
| Belief Tokens | 8 |
| Core latent dimension | 256 |
| Attention heads | 8 |
| Core FFN dimension | 1024 |
| Minute local Transformer layers | 4 |
| Commodity embedding | none |
| Formal checkpoint selection | fixed-budget final |

These values define the first controlled V1.1 benchmark.

Model-size scaling is a separate future experiment.

---

# 32. Required V1.1 implementation tests

The implementation must include more than shape tests.

## Data-boundary tests

1. Minute never crosses real contract boundary.
2. Current Daily never crosses real contract boundary.
3. Current Weekly never crosses real contract boundary.
4. Historical Weekly boundary flags are correct.
5. Future JEPA targets never cross contract boundary.
6. Historical 3Y weekly input is strictly causal.
7. PAD positions are masked correctly.

## IMC tests

8. Minute Price/OI fixed-origin coordinates are correct.
9. Minute Volume fixed baseline is causal.
10. Minute Q20 is causal.
11. Daily lifecycle origin remains fixed across anchors in the same contract.
12. Current Weekly lifecycle origin remains fixed across anchors.
13. Historical Weekly IMC resets at contract boundaries.
14. Missing baselines never use future data.
15. RB receives no refitted scaler.

## Architecture information-flow tests

16. Historical Weekly affects Commodity State.
17. Commodity State affects Contract State.
18. Daily affects Contract State.
19. Current Weekly affects Contract State.
20. Contract/Commodity State affects Minute representation before final belief compression.
21. Minute input affects final Belief.
22. Removing the higher-scale conditioning path changes the belief.
23. No commodity ID is present in the first benchmark.

## Target tests

24. Future context does not affect EMA target.
25. Future Daily/Weekly do not affect EMA target.
26. Same-origin target transformation is exact.
27. Target encoder remains market-only and frozen under gradient.

## Optimization integrity

28. Every intended trainable V1.1 parameter receives finite gradient.
29. No dead terminal branch exists.
30. Validation JEPA loss cannot select a checkpoint.
31. V0 baseline still passes regression tests.

---

# 33. What V1.1 is trying to solve

## 33.1 Commodity-scale nuisance

Solved at the coordinate layer by IMC:

\[
T(aP,bOI,cV)=T(P,OI,V)
\]

## 33.2 Contract-roll semantic contamination

Handled by real-contract episodes and contract-boundary resets.

## 33.3 Late-fusion information loss

Higher-scale state conditions minute tokens before final belief compression.

## 33.4 Generic scale symmetry that ignores market structure

V1.1 models:

\[
Commodity
\rightarrow
Contract
\rightarrow
Minute
\]

instead of treating all scales as interchangeable token pools.

## 33.5 Token-count-induced scale weighting

The four memories are read as separate information sources instead of one concatenated softmax pool.

## 33.6 Unequal commodity history lengths

Fixed capacities + masks preserve real available history without letting input tensor size grow indefinitely.

## 33.7 Training domination by long-history commodities

Handled by Commodity → Contract → Anchor balanced sampling.

---

# 34. What V1.1 still does NOT guarantee

V1.1 does not automatically establish:

\[
\boxed{
OOS\ generalization
}
\]

It does not automatically establish:

\[
\boxed{
InternalRepresentation=PredictiveState
}
\]

It does not prove:

\[
\boxed{
Price/OI/Volume\ relations\ are\ universally\ predictive
}
\]

It does not prove that larger capacity is better.

Those remain empirical questions.

---

# 35. V0 vs V1.1 scientific benchmark

The intended controlled comparison is:

\[
\boxed{
FG+SA+JM+SH+SP
\rightarrow
RB
}
\]

Both models should use, as far as experimentally possible:

- the same real-contract data population,
- the same IMC market coordinates,
- the same shared scaler,
- the same balanced sampler,
- the same JEPA future target semantics,
- the same H16/H64/H256 objective,
- the same fixed-budget checkpoint rule,
- the same evaluation interventions.

The main manipulated variable should remain:

\[
\boxed{
\text{representation architecture}
}
\]

V0:

\[
\boxed{
LateFusion
}
\]

V1.1:

\[
\boxed{
HierarchicalConditionalRepresentation
}
\]

---

# 36. Key evaluation questions

The formal comparison must ask:

1. Does V1.1 predict held-out RB future latent better than V0?
2. Does OI removal hurt V1.1 on RB?
3. Does Volume removal hurt V1.1 on RB?
4. Does removing OI+Volume hurt beyond Price-only?
5. Does Price × (OI,Volume) relation destruction hurt?
6. Are effects stable across \(H16,H64,H256\)?
7. Are the results stable across RB validation and test?
8. Does V1.1 improve long-horizon structural generalization relative to V0?

---

# 37. Final V1.1 architecture statement

V1.1 is defined by the following hierarchy:

\[
\boxed{
W^H
\rightarrow
CommodityState
}
\]

\[
\boxed{
(D,W^C)\mid CommodityState
\rightarrow
ContractState
}
\]

\[
\boxed{
MinuteIMC\mid(CommodityState,ContractState)
\rightarrow
ConditionedMinuteRepresentation
}
\]

\[
\boxed{
(ConditionedMinute,ContractState,CommodityState)
\rightarrow
B_t
}
\]

with:

\[
\boxed{
B_t\in\mathbb R^{256}
}
\]

and the unchanged JEPA future prediction objective.

---

# 38. Frozen V1.1 design summary

The central idea can be stated in one sentence:

> **V1.1 represents the futures market according to its actual hierarchy: commodity historical memory conditions the current real contract lifecycle, and both condition the current minute-level Price/OI/Volume process before the final market belief is compressed.**

Equivalently:

\[
\boxed{
\text{IMC}
+
\text{Commodity Memory}
+
\text{Contract Lifecycle}
+
\text{Minute Conditional Representation}
}
\]

This document freezes the V1.1 architectural design for implementation and controlled comparison against V0.

---

# 39. Review notes

The following points were reviewed before freezing this document.

## 39.1 Real-contract semantics

Minute, current Daily, current Weekly, and JEPA future targets are contract-local.

## 39.2 Historical Weekly semantics

Three-year commodity memory may span contracts, but IMC resets and boundary markers prevent false OI/Volume continuity.

## 39.3 IMC information preservation

The design retains:

- relative Price path,
- cumulative and sudden OI changes,
- fixed-origin Volume activity,
- local N-fold Volume surprise.

## 39.4 Causality

All observation inputs and baselines use only information available at the anchor.

## 39.5 Future-target coordinates

Future minute target uses the same fixed origin as the online minute window, preserving current-to-future displacement.

## 39.6 No accidental commodity-specific normalization

Only one shared post-IMC scaler is allowed for the universal benchmark.

## 39.7 No token-count-weighted pooled attention

Different memories are explicitly separated.

## 39.8 V1.0 prototype issues resolved by design

The terminal dead-feedback path and validation-H64 checkpoint selection are explicitly removed.

## 39.9 Capacity is not conflated with architecture

The first V1.1 baseline retains a 256-dimensional core and postpones model-size scaling.

## 39.10 Claim boundary

V1.1 is a representation architecture hypothesis. It must still earn its claims through held-out cross-commodity experiments.
