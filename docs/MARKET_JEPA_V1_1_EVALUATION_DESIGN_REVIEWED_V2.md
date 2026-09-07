# Market-JEPA V1.1 正式评估设计与 Review 规范

**状态：DRAFT FOR REVIEW**  
**用途：在交给 Codex 实现之前冻结评估目标、对照实验、统计方法、PASS/FAIL 解释边界。**

## 0. 核心目标

Market-JEPA / Market World Model 的核心目标不是交易策略，而是：

> 仅使用截至时刻 t 已发生的市场历史，学习隐藏市场状态 B_t，使其包含对未来市场行为稳定、有泛化意义的预测信息。

形式化：

H_{<=t} -> B_t -> P(Future | H_{<=t})

当前 V1.1 中：

PricePath + OIPath + VolumePath + Daily/Weekly/CommodityHistory -> B_t

目标不是 BUY/SELL 分类、PnL、Actor/Critic、人工 regime 或多空标签。

最终希望接近：

H_i ~ H_j iff P(Future|H_i) ~ P(Future|H_j)

因此当前 V1.1 最多应被验证为：

**Predictive Market Belief / Predictive Representation**

不能仅凭 JEPA latent 宣称已经得到 mathematically identified Predictive State。

---

# 1. V0 的评估标准

## 1.1 JEPA Prediction vs Persistence

V0 直接比较：

JEPA:
hat Z_{t+H} = P_H(Z_t)

Persistence:
hat Z_{t+H} = Z_t

目标：
Z^{EMA}_{t+H}

loss:
L_JEPA = 1 - cos(P_H(Z_t), Z^{EMA}_{t+H})
L_Persist = 1 - cos(Z_t, Z^{EMA}_{t+H})

H in {16,64,256}。

V0 这一项 PASS。

它能证明当前表示含有某种未来预测信息，但不能证明 latent 几何就是 Predictive State，也不能证明可以跨商品泛化。

## 1.2 latent kNN

原假设：

distance(Z_i,Z_j) 小 => Future_i 与 Future_j 相似。

V0 结果 FAIL。

后来理论上发现这个标准本身有缺陷：对任意可逆变换 Z'=AZ，同时调整 predictor 后预测可以完全等价，但欧氏几何可任意改变。

因此：

**InternalRepresentation != PredictiveState**

结论：V1.1 中 kNN 只能 diagnostic，不再是 formal PASS/FAIL gate。

## 1.3 Frozen Probe

冻结 JEPA encoder：

Z_t -> FutureOutcome

目标包括 Return / MFE / MAE / RV。

方向本身正确，因为它直接问 latent 是否包含真实未来信息。

V0 的问题是 baseline 太弱，导致“probe PASS”不能充分说明 representation 强。

所以 V1.1 必须保留 probe，但升级成强 causal baseline。

---

# 2. V0.7 / V0.8 带来的理论边界

V0.7 / V0.8 尝试把 JEPA latent 显式变成 Predictive State，包括 CME/RFF 和 CDF head，但 OOS 不稳定。

因此：

JEPA contains predictive structure
does not imply
JEPA latent is already stable PredictiveState

V1.1 正式评估必须避免：

1. 把 latent geometry 当 predictive-state semantics；
2. 把某个 probe PASS 直接等同于 Predictive State 已解决。

V1.1 当前目标应表述为：

**Learn a transferable predictive market belief.**

---

# 3. V1.1 为什么存在：具体解决 V0 什么问题

V1.1 不是因为参数更多，而是针对 V0 的：

**Compress first, condition later**

V0 概念信息流：

Minute -> 独立编码/压缩
Daily  -> 独立编码/压缩
Weekly -> 独立编码/压缩
最后 Late Fusion -> Z_market

问题：

Importance(x | D,W) >> Importance(x)

某个 minute pattern 只有在 Daily / Weekly / Commodity context 下才重要。如果 Minute encoder 在看到高层 context 之前就把它压掉，late fusion 无法恢复。

V1.1 改为：

HistoricalWeekly -> CommodityState C
(Daily, CurrentWeekly) | C -> ContractState K
Minute | (C,K) -> M'
(M',K,C) -> B_t

核心：

**Condition while representing**

而不是：

**Represent independently, then fuse**

---

# 4. 为什么直接照搬 V0 标准评估 V1.1 不够

如果 V1.1：

- JEPA vs Persistence PASS；
- Frozen Probe PASS；

最多只能说明：

**V1.1 有预测能力**

但不能说明：

**V1.1 解决了 V0 的 late-fusion limitation**

因此必须分两层：

PART A — V0-Compatible Evaluation  
PART B — V1.1-Specific Evaluation

再加：

PART C — Capacity Scaling  
PART D — Claim Boundary

---

# 5. PART A — V0-Compatible Evaluation

这里的“V0-Compatible”只表示**研究问题与指标类型连续**，不表示可以把历史 V0 的绝对数值与 V1.1 直接做 apples-to-apples 比较。V0 与 V1.1 的数据、真实合约语义、IMC、商品范围都已经变化，因此旧 V0 checkpoint 只能作为历史背景，不能作为正式数值 control。

## A1. JEPA vs Persistence

### 目标

验证 V1.1 是否至少保留并提升 V0 的 latent future prediction 能力。

### 正确 persistence

V1.1 的 B_t 不在 EMA target space，因此 persistence 不能直接用 B_t。

必须：

Z_t^{EMA} = EMA market-only encoder(current market window)

JEPA:
hat Z_{t+H}^{JEPA} = P_H(B_t)

Persistence:
hat Z_{t+H}^{Persist} = Z_t^{EMA}

Target:
Z_{t+H}^{EMA}

### loss

L_JEPA(H,i) = 1 - cos(P_H(B_ti), Z^{EMA}_{ti+H})

L_Persist(H,i) = 1 - cos(Z^{EMA}_{ti}, Z^{EMA}_{ti+H})

### primary metric

Gain_H = 1 - mean(L_JEPA_H) / mean(L_Persist_H)

总体：

Gain_ALL = 1 - mean_over_horizons(L_JEPA) / mean_over_horizons(L_Persist)

### 为什么用 Gain

S/M/L/XL latent space 不同，所以不能比较 absolute latent loss。

跨规模只能比较：

- Gain_H
- Gain_ALL

### 样本

所有模型必须使用同一个 evaluation anchor manifest。

anchor 要求：

- full H256；
- 不跨真实 contract；
- deterministic seed；
- fixed across S/M/L/XL。

### 统计

禁止 IID minute bootstrap。

block：

(contract_uid, trading_date)

Train multi-commodity aggregate 使用 equal commodity weight。

默认：

bootstrap_replicates = 2000
confidence = 95%
seed = 42

### PASS

- Gain_ALL > 0
- lower 95% CI > 0
- H16/H64/H256 point estimate 均不为负

### PARTIAL

- Gain_ALL > 0
- 但 CI crosses zero
- 或某 horizon < 0

### FAIL

- Gain_ALL <= 0

### Review checklist

- [ ] Persistence 与 future target 在同一 EMA target space。
- [ ] future target 保持 same-origin。
- [ ] future Daily/Weekly/context 不进入 target encoder。
- [ ] 不跨 contract。
- [ ] S/M/L/XL 使用相同 anchor manifest。
- [ ] 不用 absolute latent loss 跨模型排名。
- [ ] bootstrap 为 contract-day block。

---

## A2. Frozen Probe vs Strong Causal Baseline

### 目标

冻结 B_t 后，验证其是否包含比简单 causal market statistics 更强的真实未来信息。

### World model 冻结

model.eval()
requires_grad = false
EMA frozen

Probe loss 禁止回传 world model。

### 12 个未来目标

H in {16,64,256}

每个 horizon：

- Return
- MFE
- MAE
- RV

Return_H = log(Close_{t+H}/Close_t)

MFE_H = max_{1<=k<=H} log(High_{t+k}/Close_t)

MAE_H = min_{1<=k<=H} log(Low_{t+k}/Close_t)

RV_H = sqrt(sum_{k=1..H} r_{t+k}^2)

所有目标禁止跨真实 contract。

### Probe split

不能 random split minute anchors。

按 real-contract episode chronologically split：

70% ProbeFit
15% ProbeDev
15% ProbeTest

每个 commodity 独立。

一个 episode 只能属于一个 partition。

### Belief preprocessing

每个模型单独：

- ProbeFit fit per-dimension mean/std；
- frozen；
- apply 到 ProbeDev / ProbeTest / RB。

禁止用 RB fit belief normalization。

### Target preprocessing

每个 outcome：

- ProbeFit fit mean/std；
- 后续只 apply。

Primary aggregate metric 使用 standardized targets。

同时报告 native-unit RMSE / MAE。

### Primary probe

**Ridge Linear Regression**

alpha grid：

1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100

每个 outcome：

1. ProbeFit fit；
2. ProbeDev 选 alpha；
3. ProbeFit + ProbeDev refit；
4. ProbeTest 一次评估；
5. 冻结后用于 RB。

RB 禁止选 alpha。

### Baseline 1 — Unconditional

预测 ProbeFit target mean。

只做 sanity。

### Baseline 2 — Context-only Ridge

只允许 causal context：

- time-of-day sin/cos；
- weekday；
- trading-day progress；
- days since main；
- days since lost-main；
- current-week progress。

不允许 Price/OI/Volume。

### Baseline 3 — MinuteRaw24 Ridge（V0 continuity baseline）

lookback k in {4,16,64,256}

每个 k：

1. Price log return
2. realized volatility
3. high-low log range
4. OI log change
5. mean Volume-Q20
6. max Volume-Q20

共 24 个 minute market statistics，再加 causal context。

这个 baseline 用于保持与 V0 probe 思路的连续性，但**不再作为 V1.1 formal primary baseline**，因为 V1.1 还看到了 Daily / CurrentWeekly / HistoricalWeekly。若 primary baseline 只看 Minute，会把“信息集更大”误当成“representation 更好”。

### Baseline 4 — MultiScaleSummary Ridge（formal primary baseline）

Formal Gate A2 的主 baseline 必须使用与 V1.1 **同样的四类 causal information source**，但不使用 learned sequence encoder。

输入来源：

- Minute market input
- Daily market input
- CurrentWeekly market input
- HistoricalWeekly market input
- 同样的 causal context

所有输入都必须是 anchor 时刻已经可见、已经按正式 IMC / mask 规则生成的原始模型输入；禁止未来数据。

建议固定 trailing windows：

Minute:
- 16 / 64 / 256 / 512 valid tokens

Daily:
- 5 / 20 / 60 / 256 valid tokens

CurrentWeekly:
- 4 / 13 / 26 / 64 valid tokens

HistoricalWeekly:
- 13 / 52 / 104 / 156 valid tokens

对每个 source、每个 trailing window、每个 market feature channel，做 mask-aware summary：

- last valid
- mean
- std
- min
- max
- valid_fraction

注意：

- 不跨真实 contract 重新计算 Price/OI delta；
- 直接总结已经正确做过 contract-boundary reset 的 IMC channels；
- HistoricalWeekly 的 contract boundary / mask 语义必须保持；
- 不允许 commodity ID / embedding；
- baseline feature normalization 只 fit ProbeFit。

这个 baseline 的目标是：

> 给 Ridge 一个与 V1.1 接近的信息集合，但只允许固定统计汇总，不允许 learned sequence / cross-scale representation。

因此它能更公平地回答：

\[
B_t
\]

是否比简单的 multiscale causal summary 更有预测价值。

### primary metric

Skill_j = 1 - MSE_Belief_j / MSE_MultiScaleSummary_j

总体：

Skill_ALL = 1 - mean_12(MSE_Belief) / mean_12(MSE_RawBaseline)

还要算：

- Skill_H16
- Skill_H64
- Skill_H256

### PASS

- Skill_ALL > 0
- lower 95% CI > 0
- H16/H64/H256 aggregate skill point estimate 均不为负

### PARTIAL

- Skill_ALL > 0
- 但 CI crosses zero
- 或某 horizon aggregate skill < 0

### FAIL

- Skill_ALL <= 0

### Review checklist

- [ ] world model 完全 frozen。
- [ ] split 按 contract episode。
- [ ] belief normalization Train-only。
- [ ] target normalization Train-only。
- [ ] alpha 只由 ProbeDev 选择。
- [ ] RB 完全不参与 probe fitting。
- [ ] MinuteRaw24 只作为 continuity baseline。
- [ ] Formal primary baseline 是 MultiScaleSummary Ridge。
- [ ] MultiScaleSummary 与 V1.1 使用相同 causal information sources，但没有 learned sequence encoder。
- [ ] baseline 只用 causal information。
- [ ] baseline 足够强，不重复 V0 弱 baseline 问题。
- [ ] 12 个 future outcomes 与既有 V0 定义做 parity 后冻结。

---

## A3. Price / OI / Volume Structural Audit

### 目标

验证模型是否依赖：

PricePath + OIPath + VolumePath

之间的时间对齐结构。

不声称识别真实 aggressor。

claim 只能是：

**Price-aligned OI/Volume structure contributes predictive information.**

### 第一阶段只 audit 当前 512-minute sequence

原因：

- 同一真实 contract；
- 无 historical stitching；
- intervention 最清楚。

### Donor matching

recipient anchor 选择 deterministic donor：

- same commodity；
- same series_key；
- full 512 valid minute；
- same evaluation partition；
- 与 recipient raw-time window 不重叠；
- 优先 different contract_uid；
- 否则至少 different trading_date。

seed=42。

donor mapping 写入 manifest。

### OI destruction

保留 recipient Price / Volume / context。

替换所有 OI-derived minute channels 为 donor OI channels。

future target 仍是 recipient clean future。

DeltaL_OI,H = L_OI-donor,H - L_clean,H

### Volume destruction

保留 recipient Price / OI。

替换 Volume fixed-origin / Q20 / corresponding validity。

DeltaL_V,H = L_V-donor,H - L_clean,H

### Joint alignment destruction

使用同一个 donor 的 OI + Volume：

Recipient Price + Donor OI + Donor Volume

保留 donor 自己 OI/Volume 关系，但破坏 recipient Price 与它们的对齐。

DeltaL_Joint,H = L_OI+V-donor,H - L_clean,H

注意：

DeltaL_Joint 不是 pure mathematical interaction。

它表示对 Price-aligned OI/Volume structure 的依赖。

### 统计

paired contract-day block bootstrap。

### STRUCTURE PASS

- DeltaL_Joint_ALL > 0
- lower 95% CI > 0

### WEAK EVIDENCE

- point estimate > 0
- CI crosses zero

### NO EVIDENCE

- point estimate <= 0

OI-only / Volume-only 必须逐 horizon 报告，但不作为整个模型独立 hard gate。

### Review checklist

- [ ] donor deterministic。
- [ ] donor 与 recipient window 不重叠。
- [ ] 不跨 commodity。
- [ ] 不破坏 series semantics。
- [ ] future target 始终是 clean recipient future。
- [ ] 不把结果解释成真实多空因果。
- [ ] Joint 不解释成 pure interaction。

---

## A4. kNN Diagnostic Only

可以输出：

- kNN future similarity；
- PCA；
- UMAP；
- t-SNE。

但：

**不进入 formal PASS/FAIL。**

---

# 6. PART B — V1.1-Specific Evaluation

这一部分回答：

**V1.1 是否真正解决了 V0 架构局限，而不仅仅是另一个能预测的网络？**

---

## B1. Matched Late-Fusion Control

这是 V1.1 最关键的新评估。

### 为什么不能直接拿旧 V0 checkpoint 比

旧 V0 与 V1.1 在：

- commodity 范围；
- real contract semantics；
- IMC；
- history；
- split；
- data；
- partial Daily/Weekly

上差别太大。

直接 V0 vs V1.1 无法归因。

### 正确 control

建立：

**V1.1-LateFusion-Control**

必须保持：

- same Train commodities；
- same real-contract data；
- same IMC；
- same Minute/Daily/Weekly/History windows；
- same anchors；
- same history eligibility；
- same sampler；
- same JEPA target；
- same H16/H64/H256；
- same optimizer；
- same effective batch；
- same epochs/data passes；
- same checkpoint rule。

只改变信息流。

Control：

Independent encode/compress each scale
-> Late Fusion
-> belief

V1.1：

Commodity context
-> Contract context
-> condition Minute before final compression
-> belief

### Parameter matching

建议 trainable params 在对应 V1.1 的 ±5% 内。

不得通过修改 observation windows / data / target 来凑参数。

允许调整：

- branch hidden width；
- post-fusion projection width。

参数分解必须报告。

### 第一轮只做 S

先做：

V1.1-S vs V1.1-S-LateFusion-Control

如果 S 层面 architecture 没有优势，没有必要先为 M/L/XL 全部训练 control。

### 评估

双方都跑：

- A1 JEPA Gain；
- A2 Probe Skill；
- A3 Structure；
- RB generalization。

重点：

H64 / H256 / RB。

### 为什么中长 horizon 是重点

V0 后期 Price×OI×Volume OOS evidence 主要集中在 H16。

V1.1 加入 Daily / Weekly / HistoricalWeekly，本来就应该更有机会改善中长 horizon。

### ARCHITECTURE PASS

相对 matched LateFusion：

1. A1 Gain_ALL improvement > 0；
2. A2 Skill_ALL improvement > 0；
3. 至少 H64 或 H256 improvement 的 95% CI lower bound > 0；
4. RB aggregate skill 不显著更差；
5. H16 不允许 severe regression。

### PARTIAL

- overall positive；
- 中长 horizon CI 不显著；
- RB 基本持平。

### FAIL

- overall 不优于 LateFusion；
- 或仅 H16 好，而 H64/H256/RB 明显更差。

### Review checklist

- [ ] control same-data。
- [ ] same-budget。
- [ ] params ±5%。
- [ ] 唯一主要变量是 information flow。
- [ ] History / Daily / Weekly 不得偷偷删掉。
- [ ] H64/H256 单独报告。
- [ ] RB 不用于 control hyperparameter tuning。
- [ ] 旧 V0 checkpoint 不冒充 matched control。

---

## B2. Multi-scale Memory Contribution

V1.1 有：

M, D, W^C, W^H

必须回答：

**是不是实际上只靠 Minute？**

### Phase 1 — Frozen source intervention

Full 模型不重训，做：

- Full
- No Daily
- No CurrentWeekly
- No HistoricalWeekly
- Minute Only

但必须标记：

**OOD diagnostic only**

因为训练时模型没见过某路突然消失。

### source removal 做法

优先使用训练时已有的 PAD/MASK semantics：

- market values -> padding；
- masks -> invalid；
- source unavailable 明确可识别。

不要随意全零而不设置 mask。

如果 architecture 不支持 all-masked source，必须先修并 review。

### diagnostic metrics

对每种 removal 报：

- Delta Gain_H；
- Delta Skill_H；
- Delta RB Skill。

去掉某路明显下降 => 模型使用该路。

但“去掉不下降”不能单独证明该路没价值，因为 intervention OOD。

### Phase 2 — Formal retrain ablations

第一轮只建议两个：

#### MinuteOnly

训练时从一开始就让：

- Daily unavailable；
- CurrentWeekly unavailable；
- HistoricalWeekly unavailable。

Minute 数据、target、sampler、训练预算保持不变。

#### NoHistoricalWeekly

训练时从一开始就让 HistoricalWeekly unavailable。

保留：

- Minute；
- Daily；
- CurrentWeekly。

#### Source-unavailable 的实现原则

formal retrain control 必须从 epoch 0 就使用同一种 unavailable semantics，不能在训练完成后突然清零。

优先复用正式 PAD/MASK 语义。

这类 ablation 回答的是：

> Full V1.1 system 是否从这个 source / source branch 获得增量能力。

它仍然不是严格“纯信息因果效应”，因为移除 source 会改变该 branch 的有效计算。报告中必须明确这一 claim boundary。

如果后续需要进一步隔离“source information”与“source-specific capacity”，再另做 parameter-matched null-source control，不在第一轮引入。

### 为什么不一开始全做

如果 Full / NoDaily / NoWC / NoWH / MinuteOnly 再乘 S/M/L/XL，会膨胀成大量训练。

第一轮：

- Full
- LateFusion Control
- MinuteOnly
- NoHistoricalWeekly

已经足以回答主要问题。

### MULTISCALE PASS

Full > MinuteOnly：

- A1 Gain_ALL 更好；
- A2 Skill_ALL 更好；
- 至少 H64 或 H256 显著改善；
- RB 不更差。

### HISTORY PASS

Full > NoHistoricalWeekly：

- H64/H256 或 RB 至少一个显著正增益；
- 不强求 H16 一定改善。

### Review checklist

- [ ] frozen source removal 明确是 diagnostic。
- [ ] all-masked handling 安全。
- [ ] retrain controls same-data / same-budget。
- [ ] 只移除 source，不偷偷改变其它语义。
- [ ] 参数差异报告。
- [ ] 重点看 H64/H256。

---

## B3. Held-out Commodity Generalization — RB

这是 V1.1 相比 V0 的关键新能力。

### RB isolation

World model：

- RB gradient = NEVER
- RB scaler fit = NEVER
- RB checkpoint selection = NEVER

Probe：

- RB fit = NEVER
- RB alpha selection = NEVER
- RB target normalization fit = NEVER

### RB split

Review 后建议不要使用 50/50，因为 RB-Dev 只做 pipeline sanity，不应消耗一半 held-out 数据。

按每个 series_key 的真实 contract episodes chronological split：

- earlier 1/3 -> RB-Dev
- later 2/3 -> RB-Test

约束：

- 每个可评估 lineage 至少保留 1 个 RB-Dev episode；
- 其余尽量进入 RB-Test；
- 若 lineage episode 数不足以同时形成 Dev/Test，则该 lineage 不参与 formal RB split，并在报告中列出；
- split membership 必须冻结进 evaluation manifest。

### RB-Dev

只有 checkpoint 和 evaluation protocol 都冻结后才打开。

用途：

- pipeline sanity；
- runtime；
- report format。

禁止据此：

- 改模型；
- 改 alpha；
- 改 baseline；
- 改 gate threshold。

### RB-Test

只有：

- evaluation code tests PASS；
- human review PASS；
- checkpoints frozen；

之后一次运行。

运行后记录：

rb_test_consumed = true

### RB-A — JEPA vs Persistence

输出：

- RB_Gain_H16
- RB_Gain_H64
- RB_Gain_H256
- RB_Gain_ALL

### RB-B — Train-fitted Frozen Probe

Train commodities 上完成：

- belief scaler；
- target scaler；
- Ridge alpha；
- final probe；
- raw baseline。

全部冻结后直接 apply RB。

RB_Skill_j = 1 - MSE_Belief_RB_j / MSE_RawBaseline_RB_j

输出：

- RB_Skill_H16
- RB_Skill_H64
- RB_Skill_H256
- RB_Skill_ALL

### RB PASS

要求两项都 PASS：

1. RB Gain_ALL lower 95% CI > 0；
2. RB Skill_ALL lower 95% CI > 0；
3. horizons point estimate 不为负。

### PARTIAL

- 一个 PASS，一个不显著；
- 或总体正但某 horizon 负。

### FAIL

- 两个 aggregate 都 non-positive；
- 或明显 negative transfer。

### claim boundary

RB PASS 只支持：

**Unseen Commodity Generalization**

若 Train commodities 含比 RB-Test 更晚的日期，则不等于 strict time-OOS。

---

# 7. PART C — Capacity Scaling

当前模型：

- S ~8.26M
- M ~22.09M
- L ~45.52M
- XL ~102.29M

### 目标

不是：

“大模型 train loss 是否更低？”

而是：

Params up -> predictive capability up?

尤其：

Params up -> RB generalization up?

### 禁止跨规模比较

- absolute JEPA latent loss。

### 可以比较

- A1 Gain_H / Gain_ALL；
- A2 Skill_H / Skill_ALL；
- RB Gain；
- RB Skill；
- Structure DeltaLoss；
- throughput；
- training cost；
- convergence。

### scaling table

| Model | Params | A1 Gain ALL | A2 Skill ALL | RB Gain | RB Skill | Joint Structure | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 8.26M | ... | ... | ... | ... | ... | ... |
| M | 22.09M | ... | ... | ... | ... | ... | ... |
| L | 45.52M | ... | ... | ... | ... | ... | ... |
| XL | 102.29M | ... | ... | ... | ... | ... | ... |

### Strong scaling

Train predictive metrics improve
+
RB predictive metrics improve

才支持：

**capacity scaling exists**

### Memorization scaling

Train improves
RB flat/worse

说明更大模型主要拟合 Train commodities。

### Review checklist

- [ ] same data。
- [ ] same eval anchors。
- [ ] same probe protocol。
- [ ] same RB-Test。
- [ ] absolute latent loss 不用于 ranking。
- [ ] RB-Test 不被反复用来挑模型规模。

---

# 8. Claim Boundary

即使 A1/A2/B1/B2/B3/Scaling 都很好，也只能说：

**V1.1 learned a transferable, multiscale, predictive market belief.**

不能直接说：

**V1.1 is a mathematically identified PredictiveState.**

真正的 Predictive State 仍需要显式 FutureDistribution semantics。

V0.7/V0.8 的失败说明这应是独立研究问题。

---

# 9. Evaluation Manifest

正式评估先生成：

evaluation_manifest.json

每个 anchor：

- commodity
- series_key
- contract_uid
- episode_id
- anchor_timestamp
- split
- H16_valid
- H64_valid
- H256_valid
- data_manifest_sha256

manifest 自己也要 SHA256。

### 建议 sample budgets

evaluation:
  seed: 42
  train_probe_anchors_per_commodity: 4096
  rb_dev_anchors: 32768
  rb_test_anchors: 65536
  structure_audit_anchors: 16384

Sampling：

- without replacement；
- contract-balanced first；
- anchor-balanced；
- deterministic；
- same across S/M/L/XL。

---

# 10. 统一统计规范

禁止 IID minute bootstrap。

primary block：

(contract_uid, trading_date)

Train-domain：

1. equal-weight commodity；
2. commodity 内 bootstrap contract-day block。

RB：

bootstrap RB contract-day blocks。

默认：

bootstrap_replicates = 2000
bootstrap_seed = 42
confidence_level = 0.95

---

# 11. 正式状态体系

## V0_CONTINUITY_PASS

要求：

- A1 PASS；
- A2 PASS。

A3 单独报告 structural evidence。

## V1_1_ARCHITECTURE_PASS

要求：

- matched LateFusion control 被击败；
- 中长 horizon 有正证据；
- RB 不显著退化。

## V1_1_MULTISCALE_PASS

要求：

Full > MinuteOnly。

## V1_1_HISTORY_PASS

要求：

Full > NoHistoricalWeekly，并在 H64/H256/RB 至少一个有显著增益。

## V1_1_RB_GENERALIZATION_PASS

要求：

- RB JEPA PASS；
- RB Frozen Probe PASS。

## V1_1_SCALING_SUPPORTED

只有当模型规模增加伴随 RB predictive metrics 改善时才使用。

最强 claim：

**V1.1 learned a transferable, multiscale, predictive market belief.**

仍然不能叫 Identified Predictive State。

---

# 12. 建议执行顺序

## Phase 1
Full V1.1 S/M/L/XL 训练完成。

不打开 RB-Test。

## Phase 2
Train-domain：

- A1
- A2
- A3
- frozen source diagnostic

先判断训练与 representation 是否正常。

## Phase 3
训练 S matched LateFusion Control。

只先做 S。

## Phase 4
训练 S-MinuteOnly 和 S-NoHistoricalWeekly。

验证 multiscale 和 long-history contribution。

## Phase 5
冻结：

- checkpoints
- evaluation code
- probe alpha grid
- baseline definitions
- gate thresholds
- anchor manifest

## Phase 6
RB-Dev pipeline sanity。

不得据此调模型或 gate。

## Phase 7
Human Review。

确认 RB-Test untouched。

## Phase 8
One-shot RB-Test：

- S/M/L/XL Full
- 必要 S controls

然后写：

rb_test_consumed=true

---

# 13. W&B 组织

Project：

market-jepa

Training group：

v1.1-formal

Evaluation group：

v1.1-evaluation

runs：

- v1.1-S-eval
- v1.1-M-eval
- v1.1-L-eval
- v1.1-XL-eval
- v1.1-S-latefusion-control
- v1.1-S-minuteonly
- v1.1-S-nohistory

本地 CSV/JSON 仍是 source of truth。

---

# 14. Formal outputs

每个模型：

artifacts/evaluation/v1_1_<model>/
- evaluation_protocol.json
- evaluation_manifest.json
- a1_jepa_vs_persistence.csv
- a1_bootstrap.json
- a2_probe_hyperparams.json
- a2_probe_results.csv
- a2_baselines.csv
- a2_bootstrap.json
- a3_structure_results.csv
- a3_donor_manifest.json
- a3_bootstrap.json
- rb_results.csv
- rb_bootstrap.json
- summary.json
- summary.md

Controls：

artifacts/evaluation/v1_1_controls/
- latefusion_s/
- minuteonly_s/
- nohistory_s/

Scaling：

artifacts/evaluation/v1_1_scaling/
- scaling_summary.csv
- scaling_summary.json
- scaling_report.md
- rb_test_consumption.json

---

# 15. Codex 实现前 Review Checklist

## A — 核心目标
- [ ] predictive market belief，不是 trading policy。
- [ ] 当前不宣称 Identified Predictive State。
- [ ] kNN 不作为正式 Gate。

## B — V0 continuity
- [ ] JEPA vs Persistence target-space 规则冻结。
- [ ] Frozen Probe 12 outcomes 冻结。
- [ ] Strong raw baseline 24 statistics 冻结。
- [ ] Probe split 按 real-contract episode。
- [ ] Ridge alpha grid 冻结。

## C — V1.1 architecture
- [ ] LateFusion matched control topology 明确。
- [ ] 参数匹配 ±5%。
- [ ] same-data / same-budget。
- [ ] H64/H256 是重点。
- [ ] 旧 V0 checkpoint 不当 matched control。

## D — Multiscale
- [ ] frozen removal 明确 diagnostic。
- [ ] MinuteOnly retrain control 定义冻结。
- [ ] NoHistoricalWeekly retrain control 定义冻结。
- [ ] all-masked source 安全。

## E — RB
- [ ] RB training isolation 可证明。
- [ ] RB scaler isolation 可证明。
- [ ] RB split 冻结。
- [ ] RB-Test consumption 冻结。
- [ ] RB 不参与 alpha / scaler / probe fitting。
- [ ] claim 仅 commodity OOS，不冒充 time OOS。

## F — Statistics
- [ ] block = (contract_uid, trading_date)。
- [ ] 2000 bootstrap。
- [ ] Train equal commodity weight。
- [ ] same anchors across models。

## G — Scaling
- [ ] absolute latent loss 不跨模型比较。
- [ ] 比较 Gain / Skill / RB。
- [ ] RB-Test 不被反复用于挑 model size。
- [ ] scaling claim 要求 RB 同时改善。

---

# 16. 当前还需人工拍板的 9 个参数

1. Probe split：70/15/15 是否冻结。
2. RB split：**建议改为 1/3 Dev + 2/3 Test**，是否冻结。
3. Anchor budgets：4096/commodity、32768 RB-Dev、65536 RB-Test、16384 structure 是否冻结；若可用 anchor 少于预算则不做 replacement，使用全部可用并报告。
4. LateFusion parameter matching：目标 ±5%，是否冻结。
5. Formal probe：Ridge only 是否冻结。
6. Strong baseline：**MinuteRaw24 作为 continuity；MultiScaleSummary Ridge 作为 formal primary baseline**，是否冻结。
7. Bootstrap：2000 × contract-day block 是否冻结。
8. Architecture PASS 是否必须要求至少 H64 或 H256 improvement CI > 0。
9. Multiscale 第一轮正式 retrain controls 是否只做 MinuteOnly + NoHistoricalWeekly。

---

# 17. Review 结论

当前逻辑链已经闭合：

Core Goal
-> V0 Evaluation
-> V0 Limitations
-> V1.1 Architecture Rationale
-> V0-Compatible Tests
-> V1.1-Specific Controls
-> RB Generalization
-> Capacity Scaling

与旧草案相比，最关键的修正：

1. 不再把 JEPA / Probe / RB / Structure 当成互不解释的平行 Gate。
2. 新增 Matched Late-Fusion Control，直接验证 V1.1 存在的理由。
3. 明确 frozen source ablation 只是 diagnostic，正式 source contribution 要靠 retrain control。
4. scaling 只有在 RB 同时改善时才有强意义。
5. 明确 Predictive Belief 与 Predictive State 的理论边界。

**当前状态：DRAFT FOR REVIEW。**

在第 16 节的 9 个参数拍板前，不建议直接交给 Codex 实现。


---

# 18. Review 后的关键修正与判断

本轮 Review 发现两个必须在 Codex 实现前修正/冻结的地方。

## 18.1 关键修正 1：Gate A2 不能只用 Minute 24-stat baseline 作为主 baseline

原因：

V1.1 输入包含：

- Minute
- Daily
- CurrentWeekly
- HistoricalWeekly

如果 baseline 只看到 Minute，那么：

Belief > MinuteRaw24

可能仅仅表示：

> V1.1 额外看到了高时间尺度信息。

这不足以证明：

> V1.1 的 learned representation 比简单 causal summary 更好。

因此正式设计改为：

- `MinuteRaw24 Ridge`：保留，作为 V0 continuity baseline；
- `MultiScaleSummary Ridge`：新增，作为 formal primary baseline。

这是本轮 Review 最重要的修正之一。

## 18.2 关键修正 2：RB-Dev 不应消耗 50% RB

RB-Dev 的用途只是：

- pipeline sanity；
- runtime；
- report format。

它不能用于科学调参。

因此没有必要把一半 RB episode 留给 Dev。

建议：

1/3 RB-Dev
2/3 RB-Test

并且按 series_key chronological split。

## 18.3 Matched LateFusion 是 V1.1 必须有的核心 control

这是整个 V1.1 evaluation 中不可删掉的一项。

如果没有它，即便 V1.1 所有 V0-compatible metric 都 PASS，也无法回答：

> V1.1 新 hierarchy 是否真的解决了 Compress-first / LateFusion limitation。

因此：

**Matched LateFusion S control = mandatory**

而不是 optional diagnostic。

## 18.4 Frozen source ablation 不能当正式 source contribution 结论

训练后突然删除 Daily/Weekly/History 是 OOD intervention。

它适合回答：

> 模型当前是否依赖这路 source。

不适合直接回答：

> 这路 source 在训练机制上是否真正提高了可泛化预测能力。

正式 source contribution 仍需从 scratch retrain control。

## 18.5 第一轮 retrain controls 数量保持最小

建议只做：

- S-LateFusion-Control
- S-MinuteOnly
- S-NoHistoricalWeekly

这样就能分别回答：

1. hierarchy vs late fusion；
2. multiscale vs minute-only；
3. long commodity history contribution。

不建议第一轮就做 NoDaily / NoWC / 每个 size 全套 control。

## 18.6 Architecture PASS 应强调 H64/H256

V1.1 的 Daily/Weekly/HistoricalWeekly 设计，本来就是为了让 Minute representation 在更大背景下被解释。

如果最终只看到 H16 改善，而 H64/H256 没有改善，则不能强 claim：

> V1.1 已解决 V0 的 cross-scale limitation。

因此建议冻结：

**至少 H64 或 H256 一个 horizon 的 V1.1-over-LateFusion improvement 95% CI lower bound > 0**

作为 `V1_1_ARCHITECTURE_PASS` 的必要条件之一。

## 18.7 Probe split 70/15/15 可用，但要增加 minimum-episode rule

建议冻结 70/15/15，但实现必须：

- 按 real-contract episode；
- 每个正式计入 ProbeTest aggregate 的 commodity 至少要有 Fit / Dev / Test 三部分；
- episode 数过少时不允许把同一 episode 拆成 minute-level partitions；
- 不足条件的 commodity 从 formal in-domain probe aggregate 排除，并单独报告。

不能为了凑样本重新回到 random-minute split。

## 18.8 Anchor budget 是上限，不是必须凑满

例如：

4096 anchors / commodity

含义应是：

min(4096, available_valid_anchors)

禁止为了凑满：

- replacement sampling；
- duplicate anchors。

这样能保持数据真实独立性。

---

# 19. 当前 Review Verdict

当前文档在逻辑上已经足够作为 Codex implementation spec 的基础，但建议先冻结第 16 节九个参数。

本轮 review 的总体判断：

### 已经成熟、建议冻结

- 核心目标与 claim boundary；
- A1 JEPA vs Persistence；
- Ridge probe；
- contract-level probe split 思路；
- block bootstrap；
- Matched LateFusion control；
- MinuteOnly / NoHistoricalWeekly minimal controls；
- RB zero-shot 原则；
- S/M/L/XL scaling 比较逻辑；
- kNN diagnostic-only。

### 本轮已修正

- Formal strong baseline 从 Minute-only 提升为 MultiScaleSummary；
- RB split 从 50/50 建议改成 1/3 Dev + 2/3 Test；
- Anchor budgets 明确为上限，不做 replacement；
- source-ablation claim boundary 进一步收紧。

### 仍需人工拍板

- 第 16 节九项具体数值/阈值。

在这些项目冻结后，状态可以从：

`DRAFT FOR REVIEW`

升级为：

`APPROVED FOR CODEX IMPLEMENTATION`
