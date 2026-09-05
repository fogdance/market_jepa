# Market World Model Architecture Evolution
## V0 Baseline → V1 Cross-Scale Conditional Representation

**Document status:** Reviewed design rationale  
**Purpose:** Preserve the architectural history, scientific motivation, and exact reasons for moving from V0 to V1 without erasing the value of V0 as a baseline.  
**Scope:** Representation architecture only. This document does **not** freeze the final V1 layer counts, hidden sizes, loss weights, or training schedule.

---

# 1. Why this document exists

The architecture has changed because the understanding of the learning problem has changed.

V0 was designed when the primary question was:

\[
\boxed{
\text{Can multiple kinds and scales of market information be encoded into a predictive representation?}
}
\]

After subsequent experiments, ablations, predictive-state work, and the development of cross-commodity invariant coordinates, the question became more precise:

\[
\boxed{
\text{Must market structure be formed through interaction across scales before information is compressed?}
}
\]

This is not simply a model-size upgrade.

The central architectural transition is:

\[
\boxed{
\text{V0: encode} \rightarrow \text{compress} \rightarrow \text{interact}
}
\]

to:

\[
\boxed{
\text{V1: encode} \leftrightarrow \text{condition} \leftrightarrow \text{interact}
\rightarrow \text{compress}
}
\]

V0 remains scientifically important because it provides the clean baseline against which V1 must be tested.

---

# 2. Original market-world-model motivation

The project began from a mismatch between standard action-conditioned world models and financial markets.

A standard agent-centric world model often assumes:

\[
S_t + A_t \rightarrow S_{t+1}
\]

where the agent's action participates in the transition of the environment.

For an ordinary trader, however:

\[
\boxed{
\text{BUY / SELL / HOLD does not materially determine the next market state}
}
\]

The trader's action changes position, PnL, exposure, and risk, but not the market transition itself.

Therefore the market-world-model problem was reformulated as:

\[
\boxed{
\text{Observed Market History}
\rightarrow
\text{Hidden Market Representation}
\rightarrow
\text{Future Market Behavior}
}
\]

The goal is to model the market before attaching any trading policy.

---

# 3. V0 design background

## 3.1 The question V0 was intended to answer

At the time V0 was designed, the project did not yet have strong evidence about what a useful market representation should look like internally.

The immediate experimental question was:

> If the model sees historical market data only, can JEPA learn an internal representation that contains information relevant to future market behavior?

The priority was therefore:

\[
\boxed{
\text{Make sure the model can see the important sources of information}
}
\]

rather than:

\[
\boxed{
\text{Make every information source interact before compression}
}
\]

This distinction is important when reviewing V0 retrospectively.

---

# 4. V0 architecture

V0 uses separate branches for different information types and temporal scales.

## 4.1 Minute market branch

Representative V0 configuration:

- context length: approximately 512 minute bars;
- Transformer layers: 4;
- hidden dimension: 256;
- attention heads: 8;
- FFN dimension: 1024.

Conceptually:

\[
M_{1:T}
\rightarrow
Encoder_{minute}
\rightarrow
Z_m
\]

where:

\[
Z_m \in \mathbb{R}^{256}
\]

The minute branch is the main place where short-horizon Price / OI / Volume interactions can be learned.

## 4.2 Minute time/context branch

Clock/calendar/context information is processed separately:

\[
C_{1:T}
\rightarrow
GRU_{context}
\rightarrow
Z_c
\]

with approximately:

\[
Z_c \in \mathbb{R}^{32}
\]

## 4.3 Daily branch

\[
D_{1:n}
\rightarrow
GRU_{daily}
\rightarrow
Z_d
\]

with approximately:

\[
Z_d \in \mathbb{R}^{128}
\]

## 4.4 Weekly branch

\[
W_{1:k}
\rightarrow
GRU_{weekly}
\rightarrow
Z_w
\]

with approximately:

\[
Z_w \in \mathbb{R}^{128}
\]

## 4.5 Late fusion

The four compressed branch outputs are concatenated:

\[
[Z_m,Z_c,Z_d,Z_w]
\]

with total dimension:

\[
256+32+128+128=544
\]

and passed through:

\[
544 \rightarrow 512 \rightarrow 256
\]

producing:

\[
\boxed{Z_{market} \in \mathbb{R}^{256}}
\]

Thus V0 is:

\[
\boxed{
\text{Independent Encoding}
\rightarrow
\text{Branch Compression}
\rightarrow
\text{Late Fusion}
}
\]

---

# 5. Why V0 was reasonable at the time

V0 should not be described as an obvious architectural mistake.

It was a reasonable first scientific baseline because:

1. **Clear modularity** — minute, context, daily, weekly branches can be independently tested and ablated.
2. **Causal construction is easier to validate** — information flow is simple and explicit.
3. **Low computational cost** — repeated experiments are practical.
4. **It directly answers the original question** — whether predictive information can be learned at all.

Therefore V0 remains a valid:

\[
\boxed{\text{first scientific baseline}}
\]

---

# 6. What V0 experiments taught us

## 6.1 V0 did learn useful Price / OI / Volume information

Intervention and removal audits showed that V0 JEPA did use OI and Volume.

Relation-destruction experiments also provided evidence that V0 learned some:

\[
\boxed{Price \times (OI,Volume)}
\]

joint structure, with the strongest out-of-sample evidence concentrated at shorter horizons.

Therefore V0 did **not** fail because it was incapable of learning any market structure.

## 6.2 The structure did not generalize equally well across horizons and time

Learned effects weakened substantially out of sample, especially at longer horizons.

This suggested that the model could learn useful local structure while still failing to build a sufficiently stable representation across time.

## 6.3 Internal Representation is not Predictive State

Subsequent work established:

\[
\boxed{InternalRepresentation \neq PredictiveState}
\]

A representation may contain predictive information without its raw coordinates having stable future-distribution semantics.

This was supported by latent-geometry failures and frozen predictive-state mapping failures.

Therefore simply increasing representation size does not solve the predictive-state problem.

---

# 7. The deeper architectural issue discovered later

With a more mature understanding of the market-structure problem, the main V0 limitation became clearer:

\[
\boxed{\text{Conditioning happens after branch compression}}
\]

or:

\[
\boxed{Compress\ first,\ condition\ later}
\]

---

# 8. Why late fusion can lose important information

Assume a minute-scale event \(x\) appears in the recent market sequence.

Its importance may depend strongly on a higher-scale state \(D\):

\[
Importance(x\mid D) \gg Importance(x)
\]

Examples include:

- sudden OI increase after several days of persistent OI decline;
- the same OI increase during an already-established multi-day accumulation;
- a short Volume burst in a quiet background;
- the same Volume burst during an already highly active day.

In V0:

\[
MinutePath \rightarrow Z_m
\]

is computed before the minute branch knows \(D\).

A detail that appears locally unimportant may therefore be weakened or discarded during compression.

Later:

\[
Fusion(Z_m,Z_d)
\]

can only use information that survived inside \(Z_m\).

If a relevant minute detail has disappeared:

\[
\boxed{\text{late fusion cannot reconstruct it}}
\]

This is the **late-fusion information bottleneck**.

---

# 9. Market × Time has the same structural issue

In V0, minute market information and time/context are processed separately.

Therefore the minute market encoder does not directly know whether the same event occurs:

- near session open;
- mid-session;
- near close;
- in the night session;
- in a specific weekday context.

The interaction:

\[
Market \times Time
\]

occurs only after both branches have been compressed.

Yet the meaning of a market event may depend on temporal context, e.g.:

\[
VolumeShock \times SessionPosition
\]

Thus V0 has no token-level Market × Time interaction.

---

# 10. Cross-scale interpretation is the more important limitation

The project increasingly focuses on continuous market processes:

\[
\boxed{PricePath + OIPath + VolumePath}
\]

across minutes, tens of minutes, hours, a trading day, and multiple trading days.

A short-scale event may need to be interpreted conditionally on longer-scale state.

The desired structure is closer to:

\[
\boxed{
(\text{minute Price/OI/Volume process})
\times
(\text{daily state})
\times
(\text{weekly state})
}
\]

V0 allows these interactions only after each scale has already been reduced to a compressed vector.

---

# 11. IMC changed the architectural problem further

Cross-commodity learning introduced a different problem: absolute Price, OI, and Volume scales vary strongly by commodity.

Market Invariant Coordinates (IMC) are intended to remove multiplicative commodity-specific scales mathematically while preserving relative dynamics:

\[
\boxed{T(aP,bOI,cV)=T(P,OI,V)}
\]

Representative coordinates include:

\[
P_t^\star = \log(P_t/P_0)
\]

\[
OI_t^\star = \log(OI_t/OI_0)
\]

plus relative Volume coordinates.

This means the neural model should spend less capacity learning nuisance scale and more capacity learning:

\[
\boxed{\text{how Price, OI, and Volume move together}}
\]

IMC and V1 solve different problems:

- **IMC:** coordinate/scale problem;
- **V1:** representation information-flow problem.

---

# 12. V1 design background

The V1 research question is:

> How should Price / OI / Volume processes at different scales interact while the market representation is being formed?

V1 therefore changes the design principle from:

\[
\boxed{Independent\ Encoding + Late\ Fusion}
\]

to:

\[
\boxed{Conditional\ Multi\text{-}Scale\ Representation}
\]

The guiding phrase is:

\[
\boxed{Condition\ while\ representing}
\]

---

# 13. V1 design principle 1 — Market and context interact earlier

A minute token should not first become a market representation that is unaware of context.

Conceptually:

\[
Token_t = f(MarketIMC_t, Context_t)
\]

This allows the representation process itself to learn:

\[
Price \times OI \times Volume \times Context
\]

instead of delaying Market × Context interaction until final fusion.

The exact mechanism is not yet frozen. Candidates include:

- concatenation + projection;
- additive embeddings;
- gated conditioning;
- FiLM-style conditioning;
- token-level context attention.

---

# 14. V1 design principle 2 — Do not fully compress each scale before cross-scale interaction

Instead of:

\[
MinuteTokens \rightarrow Z_m
\]

\[
DailyTokens \rightarrow Z_d
\]

\[
WeeklyTokens \rightarrow Z_w
\]

followed only by final fusion, V1 should be closer to:

\[
MinuteTokens \rightarrow LocalMinuteRepresentation
\]

\[
DailyTokens \rightarrow LocalDailyRepresentation
\]

\[
WeeklyTokens \rightarrow LocalWeeklyRepresentation
\]

then:

\[
\boxed{CrossScaleInteraction(Minute,Daily,Weekly)}
\]

and only after that:

\[
\rightarrow B_t
\]

The core rule is:

\[
\boxed{\text{Cross-scale conditioning must occur before irreversible final compression}}
\]

---

# 15. V1 design principle 3 — Preserve hierarchy without requiring brute-force all-to-all attention

V1 does not require every minute token to attend to every daily and weekly token.

A more efficient design may use **cross-scale state tokens**.

For example:

\[
S^{(1)} = Attend(S^{(0)},MinuteTokens)
\]

\[
S^{(2)} = Attend(S^{(1)},DailyTokens)
\]

\[
S^{(3)} = Attend(S^{(2)},WeeklyTokens)
\]

or parallel cross-attention followed by state-token updates.

Another possibility is higher-scale conditioning of lower-scale tokens:

\[
MinuteTokens' = Condition(MinuteTokens,DailyState,WeeklyState)
\]

The exact topology remains open.

The frozen V1 requirement is architectural, not implementation-specific:

\[
\boxed{
\text{higher- and lower-scale information must influence representation formation before final compression}
}
\]

---

# 16. V1 design principle 4 — Avoid commodity identity shortcuts in the first cross-commodity experiment

For:

\[
FG+SA+JM+SH+SP \rightarrow RB
\]

the first V1 version should preferably avoid:

- commodity ID;
- commodity embedding;
- commodity-specific encoder;
- commodity-specific predictive head.

Combined with IMC, this increases pressure on the model to learn reusable market dynamics rather than commodity identity.

---

# 17. What V1 specifically tries to solve

## 17.1 Late-fusion information loss

**V0:** a branch may discard information before seeing other scales.  
**V1:** other scales can condition representation before final compression.

## 17.2 Market × Time separation

**V0:** market and context first form separate compressed states.  
**V1:** market behavior can be interpreted together with context earlier.

## 17.3 Cross-scale conditional interpretation

**V0:** minute/daily/weekly interaction occurs only between compressed branch states.  
**V1:** fine-scale details can be interpreted in the presence of higher-scale state.

## 17.4 Cross-commodity absolute-scale shortcuts

**V0:** absolute levels can provide easy commodity/era shortcuts.  
**V1 environment:** IMC is intended to remove multiplicative scale information before neural encoding.

---

# 18. What V1 does NOT automatically solve

This section is critical.

## 18.1 V1 does not guarantee OOS generalization

Cross-scale interaction increases representational capability. It does not guarantee stable generalization to unseen years or commodities.

## 18.2 V1 does not make the latent state a Predictive State automatically

The distinction remains:

\[
\boxed{InternalRepresentation \neq PredictiveState}
\]

V1 improves how the internal belief/representation is formed. Future-distribution semantics remain a separate problem.

## 18.3 V1 does not prove Price/OI/Volume relations are universal

Whether the learned relations survive:

\[
FG+SA+JM+SH+SP \rightarrow RB
\]

must be tested experimentally.

## 18.4 V1 does not mean larger is automatically better

The main change is information flow. Width/depth/parameter count should be studied separately to avoid confounding architecture and capacity.

---

# 19. V0 versus V1

| Dimension | V0 | V1 |
|---|---|---|
| Primary question | Can multiple market scales be encoded? | Must market structure be formed through cross-scale conditioning before compression? |
| Price/OI/Volume | Available to minute market branch | Preserved, preferably through IMC |
| Market × Time | Late fusion | Early/token-level or early conditional interaction |
| Minute × Daily | After branch compression | Before final compression |
| Minute × Weekly | After branch compression | Before final compression |
| Daily × Weekly | Late fusion | Cross-scale representation stage |
| Branch architecture | Independent encoders | Local encoders + cross-scale interaction |
| Final compression | Branches compressed before cross-scale interaction | After meaningful cross-scale interaction |
| Commodity-specific absolute scale | Can leak through raw inputs | Reduced mathematically with IMC |
| Commodity ID | Not central in single-commodity V0 | Prefer absent in first cross-commodity V1 |
| Main strength | Clean modular scientific baseline | Better conditional multi-scale representation capacity |
| Main limitation | Late-fusion information bottleneck | Greater complexity; still requires validation |
| Predictive-State semantics | Not guaranteed | Still not guaranteed |

---

# 20. The architectural transition in equations

V0:

\[
\boxed{
B_t^{V0}
=
F(
Compress_m(M),
Compress_c(C),
Compress_d(D),
Compress_w(W)
)
}
\]

V1 conceptual form:

\[
\boxed{
B_t^{V1}
=
Compress(
Interact(
Encode_m(M,C),
Encode_d(D),
Encode_w(W)
)
)
}
\]

The essential difference is ordering:

\[
\boxed{V0:\ Compression \prec CrossScaleInteraction}
\]

versus:

\[
\boxed{V1:\ CrossScaleInteraction \prec FinalCompression}
\]

---

# 21. Scientific role of V0 after V1 is introduced

V0 must be retained as:

\[
\boxed{V0 = Late\text{-}Fusion\ Baseline}
\]

V1 becomes:

\[
\boxed{V1 = Cross\text{-}Scale\ Conditional\ Representation}
\]

The ideal controlled benchmark is:

\[
\boxed{FG+SA+JM+SH+SP \rightarrow RB}
\]

with the same:

- source data;
- IMC transformation;
- train/validation/test split;
- JEPA objective;
- optimizer family;
- evaluation procedures;
- training budget as far as practical.

The main manipulated variable should be:

\[
\boxed{\text{representation architecture}}
\]

---

# 22. V0 vs V1 evaluation targets

Training loss alone is insufficient.

## 22.1 Future prediction

Does V1 support better future-latent prediction on held-out RB?

## 22.2 OI utilization

\[
D_{OI}=E_{NO\_OI}-E_{ORIGINAL}
\]

## 22.3 Volume utilization

\[
D_V=E_{NO\_VOLUME}-E_{ORIGINAL}
\]

## 22.4 Price × OI × Volume relation

\[
D_{P\times OV}=E_{BROKEN}-E_{ORIGINAL}
\]

## 22.5 Horizon stability

Keep results separated across:

\[
H16,\quad H64,\quad H256
\]

because previous evidence suggested stronger short-horizon than long-horizon structural transfer.

---

# 23. Interpretation rules

### If V1 > V0 only on training data
Do **not** conclude V1 is better. Greater capacity may simply fit training data better.

### If V1 > V0 on seen commodities but not held-out RB
Do **not** conclude cross-scale architecture improves structural generalization.

### If V1 > V0 on RB prediction but structural audits do not improve
The improvement may come from signals other than the intended Price/OI/Volume structure.

### Strongest evidence for V1
The strongest result would be V1 > V0 on held-out RB in:

- prediction;
- OI utilization;
- Volume utilization;
- Price × OI × Volume relation-destruction effect;
- validation-to-test stability.

Only then is there good empirical support for the claim that earlier cross-scale conditioning improves reusable market representation.

---

# 24. Architecture design constraints for eventual V1 implementation

Before code is frozen, the V1 proposal should satisfy:

1. IMC-compatible inputs;
2. no future leakage;
3. Market/context interaction before final compression;
4. Minute/daily/weekly interaction before final compression;
5. no requirement for brute-force all-to-all attention;
6. computational feasibility on the available GPU;
7. no commodity-ID shortcut in the first universal-model experiment;
8. JEPA objective kept separable from architecture;
9. architecture changes separable from model-size scaling;
10. V0 remains runnable under the same evaluation protocol.

---

# 25. V1 is not yet frozen

This document freezes the **reason for V1**, not the final V1 implementation.

Still open:

- exact tokenization;
- exact minute/daily/weekly token counts;
- cross-attention direction;
- state-token count;
- number of interaction rounds;
- hidden dimension;
- Transformer depth;
- FFN size;
- local encoder type;
- whether minute tokens receive direct higher-scale conditioning;
- parameter-matching strategy against V0.

These belong in a dedicated V1 architecture-design document.

---

# 26. Reviewed summary

## V0

\[
\boxed{
\textbf{Can multiple market scales be encoded into one predictive representation?}
}
\]

V0 answered this with independent branches and late fusion.

It was a reasonable first baseline and provided evidence that JEPA can use OI, Volume, and some Price × OI × Volume structure.

Its main newly recognized limitation is:

\[
\boxed{
\text{important conditional relationships may become visible only after the information needed to express them has already been compressed}
}
\]

## V1

\[
\boxed{
\textbf{Must market structure be formed through interaction across scales before final compression?}
}
\]

V1 changes the ordering of representation formation.

The core architectural hypothesis is:

\[
\boxed{\text{Condition while representing, not only after representing.}}
\]

---

# 27. Review notes

The following points were checked before freezing this document.

## 27.1 No retrospective overstatement of V0

V0 is not characterized as a failed or irrational architecture. It remains a valid baseline for the original question.

## 27.2 No unsupported root-cause claim

Existing experiments do **not** prove late fusion caused previous OOS failures. This document treats it as an architectural limitation/hypothesis, not an established root cause.

## 27.3 No conflation of IMC and V1

IMC solves scale/coordinate invariance. V1 addresses information flow and cross-scale conditioning.

## 27.4 No conflation of representation and Predictive State

V1 improves representation formation only. It does not establish future-distribution semantics.

## 27.5 No assumption that larger is better

Model capacity remains a separate experiment.

## 27.6 V0 remains the baseline

The V0 implementation should remain intact so that V1 can be evaluated under controlled conditions.

---

# 28. Final frozen architectural rationale

The V0 → V1 transition is:

\[
\boxed{\text{from multi-source late fusion}}
\]

to:

\[
\boxed{\text{cross-scale conditional representation before final compression}}
\]

because the project's understanding has evolved from:

> the model needs to see multiple market scales

into:

> the meaning of a market event may depend on other scales, so those scales must be able to interact before the event is irreversibly compressed.

That is the core reason for V1.
