# Market-JEPA V1.1 Stage-A 核心审计重构需求（Codex Implementation Spec）

**状态：REVIEWED / READY FOR CODEX**
**目的：把当前“全套 formal evaluation”收缩为 Stage A 的唯一核心问题：V1.1 fixed-window JEPA 是否真正学到了对未来有稳定意义的 predictive latent representation。**
**最高原则：先证明 `History -> Predictive Latent -> Future` 成立；V1 vs V0、A3、controls、RB、S/M/L/XL scaling 均后移。**

---

# 0. 核心问题重新冻结

当前最重要的问题不是：

```text
V1 是否比 V0 分数更高？
```

而是：

```text
V1.1 是否学到一个非平凡、对未来有稳定预测价值的 latent representation？
```

形式化：

\[
\boxed{
ObservedHistory_t
\rightarrow
Z_t
\rightarrow
FutureMarketBehavior
}
\]

Stage A 的 GO 只允许支持以下 claim：

> 在严格时间隔离的真实合约测试 episode 上，V1.1 fixed-window JEPA 学得的 latent representation 包含稳定的 future-relevant information，并且其冻结线性 probe 相比强 causal history summary baseline 有显著增益。

禁止升级成：

```text
Predictive State 已识别
HiddenMarketState 已被证明存在
完整 World Model 已成立
RSSM 已有必要
可以用于交易
```

---

# 1. 当前 evaluation 实现的问题

当前 `train-domain` 一次执行：

```text
A1 JEPA vs Persistence
A2 Frozen Probe
A3 OI/Volume structure audit
Source-removal diagnostics
Control/scaling/RB 所需 provenance 输出
```

后续又围绕：

```text
LateFusion
MinuteOnly
NoHistoricalWeekly
S/M/L/XL
RB inventory
RB freeze
RB-Dev
RB-Test
scaling comparison
```

构建了一套完整 formal campaign。

这些功能本身可以保留，但不是当前 Stage A Gate。

当前最严重的问题反而是：

## 1.1 现有 `probe_test` 不是 encoder-level strict temporal OOS

当前训练使用所有 Train commodity 的全部 eligible episodes。

evaluation 结束后才按 episode 做：

```text
probe_fit / probe_dev / probe_test
```

因此当前 `probe_test` 只对 Ridge probe 是 OOS：

```text
Ridge 没在 ProbeTest 拟合
```

但 JEPA encoder 在训练期间已经见过这些 episode / anchor。

所以不能声称：

```text
strict temporal OOS representation generalization
```

这必须修。

**结论：只改 evaluation 不够。**
为了回答 Stage A 核心命题，必须加一个最小的 training/data isolation 改动：
在正式训练开始前冻结 temporal split，并把 formal ProbeTest episode 从 JEPA training sampler 和 shared scaler 中排除。

---

# 2. 新的 Stage-A 实验结构

完整结构改成：

```text
Eligible real-contract population
        │
        ↓
Chronological temporal split
        │
        ├── ProbeFit   ┐
        ├── ProbeDev   ├── JEPA pretraining allowed
        │              │
        └── ProbeTest  ─── JEPA pretraining FORBIDDEN
                         shared-scaler fitting FORBIDDEN

JEPA training
        ↓
fixed 250K optimizer steps
        ↓
final S checkpoint
        ↓
Stage-A core audit
        ├── A0 latent-health / collapse sanity
        ├── A1 OOS JEPA future prediction
        └── A2 OOS frozen probe vs causal baseline
        ↓
STAGE_A_GO / NO_GO / INCONCLUSIVE / BLOCKED
```

当前阶段只要求 **Full S**。

不要要求：

```text
M/L/XL
LateFusion
MinuteOnly
NoHistoricalWeekly
RB
A3
```

这些都在 Stage A GO 之后再做。

---

# 3. Temporal OOS split：P0 必须实现

## 3.1 Split unit

为了避免同一真实合约的 re-main episode 跨 Train/Test，正式 split 单位使用：

```text
(commodity, contract_uid)
```

即一个 `contract_uid` 下的全部 ContractEpisode 必须属于同一 partition。

不要按单个 anchor split。
不要允许同一 contract_uid 同时出现在 ProbeFit/Dev/Test。

---

## 3.2 Chronological split

对每个 commodity：

1. 收集通过现有 history / full-bar / dataset eligibility hard gates 的 contract groups；
2. 按该 contract group 最早 `main_start` 排序；
3. 使用现有 70/15/15 思想，但按 contract group：
   - earlier groups -> `probe_fit`
   - middle -> `probe_dev`
   - latest suffix -> `probe_test`

至少保留：

```text
1 ProbeFit contract group
1 ProbeDev contract group
1 ProbeTest contract group
```

因此 formal Stage-A commodity 至少需要 3 个 eligible contract groups。

对于 `<3` contract groups 的 commodity：

```text
stage_a_formal = false
reason = insufficient_contract_groups
```

它可以继续参加 JEPA training，但不能进入 Stage-A formal OOS aggregate。

---

## 3.3 ProbeTest 必须是 latest suffix

这一点是 hard requirement。

不要随机抽 test episode。

原因：

```text
ProbeTest 是每个 commodity 的最后一段真实合同时间
```

因此正常 causal history 不可能让更早的 training anchor 使用未来 ProbeTest 数据。

这给 strict temporal OOS 一个清晰、容易审计的语义。

---

# 4. Split 必须在训练前冻结

不要等训练完成以后才重新计算 split。

formal training preflight 在所有 existing eligibility filtering 完成之后立即生成：

```text
stage_a_temporal_split.json
```

建议结构：

```json
{
  "version": "stage_a_temporal_oos_v1",
  "unit": "commodity_contract_uid",
  "policy": "chronological_70_15_15_latest_suffix_test",
  "data_manifest_sha256": "...",
  "commodities": {
    "FG": {
      "probe_fit_contracts": ["FG..."],
      "probe_dev_contracts": ["FG..."],
      "probe_test_contracts": ["FG..."],
      "formal_stage_a": true
    }
  },
  "excluded_formal_commodities": {
    "SH": "insufficient_contract_groups"
  },
  "sha256": "..."
}
```

这个文件：

- immutable；
- training checkpoint 必须记录其 SHA；
- resume 必须验证 SHA；
- S/M/L/XL / controls 将来必须复用同一 split；
- evaluation 只能使用 checkpoint 绑定的 split，不能重新生成一个新的。

---

# 5. Training isolation：P0

## 5.1 JEPA sampler

对于 `formal_stage_a=true` 的 commodity：

```text
ProbeFit   -> allowed
ProbeDev   -> allowed
ProbeTest  -> FORBIDDEN
```

对于 `<3 contract groups`、不参加 formal Stage-A 的 commodity：

```text
所有 eligible contracts 可以继续 training
```

Scheme A 本身不变：

```text
fixed_sample_budget_v1
250,000 optimizer steps
effective batch = 128
32M sample exposures
```

只改变 sampler population。

`sampling_population_sha256` 必须自然反映新的 restricted population。

---

## 5.2 Shared IMC scaler

这是 hard requirement：

```text
ProbeTest anchors MUST NOT be used to fit SharedIMCScaler.
```

当前 `fit_v11_shared_scaler()` 从整个 train hierarchy 中选 anchor。
需要增加 allowed training population filter。

建议：

```python
fit_v11_shared_scaler(
    dataset,
    anchors_per_commodity,
    allowed_contract_uids=...
)
```

或提供 filtered dataset view。

不要只过滤 sampler 而漏掉 scaler。

---

## 5.3 Training audit

`data_audit.json` / formal protocol 增加：

```text
stage_a_temporal_split_sha256
stage_a_formal_commodities
stage_a_probe_test_contract_count
stage_a_probe_test_sampler_samples = 0
stage_a_probe_test_scaler_samples = 0
```

如果后两者不是 0：

```text
FORMAL TRAINING HARD FAIL
```

---

## 5.4 Causal leakage sentinel test

新增一个强测试：

1. 构造一个 synthetic commodity；
2. latest contract group 为 ProbeTest；
3. 建立一个合法 ProbeFit/Dev training sample；
4. 将 ProbeTest raw bars 全部替换成极端 sentinel 值；
5. 重新构造同一个 earlier training sample；
6. 要求所有 model input tensors bitwise/equal within deterministic tolerance。

即证明：

\[
F(X_{\le t}, ProbeTestFuture)
=
F(X_{\le t}, ProbeTestFuture')
\]

ProbeTest 不能通过 history cache / weekly lineage / context 间接污染 earlier train samples。

---

# 6. 新增 Stage-A 核心命令

不要删除现有完整 evaluation。

新增：

```bash
python evaluate_market_jepa_v1_1.py stage-a \
  --checkpoint <Full-S-last.pt> \
  --output <stage_a_output> \
  --device cuda \
  --batch-size 8
```

Stage-A command：

- 只允许 `variant=Full`；
- 默认只用于 S；
- 不需要外部 manifest 参数；
- 从 checkpoint / training run 加载绑定的 `stage_a_temporal_split.json`；
- 自己建立 deterministic Stage-A evaluation anchor manifest；
- 不运行 A3；
- 不运行 source-removal diagnostics；
- 不要求 controls；
- 不读取 RB；
- 不要求 M/L/XL。

现有：

```text
train-domain
compare
freeze-rb
rb-dev
rb-test
```

保留，标成：

```text
extended/post-Stage-A evaluation
```

不要为了这次任务删除它们。

---

# 7. Stage-A evaluation anchor population

## 7.1 ProbeFit

从 split 中 `probe_fit_contracts` 选 anchor。

用途：

```text
Ridge X/Y normalizer
Ridge initial fit
Train-mean latent baseline
Block-shuffle source pool
```

继续保持：

```text
cap = 4096 anchors / commodity
no replacement
```

---

## 7.2 ProbeDev

从 `probe_dev_contracts` 选 anchor。

用途：

```text
Ridge alpha selection only
```

不能用于最终 metric。

---

## 7.3 ProbeTest

从 `probe_test_contracts` 选 anchor。

这是唯一 formal Stage-A outcome population：

```text
A0 latent health
A1 formal prediction
A2 formal frozen probe
```

所有 GO/NO-GO 都只根据 ProbeTest。

---

## 7.4 Sampling

继续：

```text
balanced / deterministic
no replacement
real-contract / contract-day aware
equal commodity aggregation
```

保留现有：

```text
bootstrap = 2000
block = (contract_uid, trading_date)
equal commodity weighting
```

---

# 8. A0：Latent Health / Collapse Sanity

当前低 cosine loss 很容易出现，因此必须先排除 trivial representation。

## 8.1 必须导出

在 ProbeTest 保存：

```text
belief Z_t
EMA target Z_{t+16}
EMA target Z_{t+64}
EMA target Z_{t+256}
```

不要只保存 cosine loss。

---

## 8.2 对 belief 和每个 target horizon 计算

对于矩阵：

\[
X\in R^{N\times d}
\]

计算：

```text
sample_count
dimension
finite_fraction
mean_vector_norm

per_dimension_std:
  min
  p10
  median
  mean
  max

centered_total_variance

covariance eigenvalues / singular spectrum

numerical_rank
effective_rank
effective_rank_ratio

participation_ratio

mean_raw_pairwise_cosine
mean_centered_pairwise_cosine
```

Effective rank：

\[
p_i=\lambda_i/\sum_j\lambda_j
\]

\[
r_{eff}
=
\exp\left(-\sum_i p_i\log p_i\right)
\]

Participation ratio：

\[
r_{PR}
=
\frac{(\sum_i\lambda_i)^2}
{\sum_i\lambda_i^2}
\]

pairwise cosine 可以 deterministic subsample，最多 4096 embeddings，避免 O(N²) 爆内存。

---

## 8.3 A0 hard gate

A0 只负责排除明显 trivial collapse，不用人为规定“优秀 latent 必须 rank=100”。

Hard FAIL：

```text
NaN / Inf
N < required minimum
centered_total_variance <= numerical epsilon
numerical_rank < 2
```

其中 numerical rank 使用 relative threshold，例如：

```text
lambda_i > lambda_max * 1e-6
```

以下只作为 warning，不作为 formal fail：

```text
effective_rank_ratio < 0.05
mean_raw_pairwise_cosine > 0.99
```

输出：

```text
a0_latent_health.json
```

状态：

```text
PASS
BLOCKED_TRIVIAL_COLLAPSE
```

---

# 9. A1：Strict-OOS Future Prediction

A1 必须只在 ProbeTest 上作为 formal metric。

当前 `horizon_persistence()` 的 V0 continuity 语义保留：

```text
H16  -> current trailing 16 bars
H64  -> current trailing 64 bars
H256 -> current trailing 256 bars
```

并继续使用相同 EMA target encoder 与 online same-origin IMC。

---

## 9.1 Primary baseline：Persistence

定义：

\[
Gain_H
=
1-
\frac{L_{JEPA,H}}
{L_{Persistence,H}}
\]

保持当前 paired contract-day bootstrap。

---

## 9.2 增加 Train-Mean baseline

为了排除 target latent 接近常量：

对每个 H：

```text
只使用 ProbeFit 的 EMA target
计算 mean target vector
```

然后作为 ProbeTest 的固定预测。

禁止使用 ProbeTest target 计算 mean。

报告：

\[
GainMean_H
=
1-
L_{JEPA}/L_{MeanTarget}
\]

---

## 9.3 增加 Block-Shuffle sanity baseline

从 ProbeFit target pool 中 deterministic 抽取与 ProbeTest anchor 无关的 target vector。

必须：

```text
different contract-day when possible
seed = 42
no ProbeTest target leakage
```

报告：

\[
GainShuffle_H
=
1-
L_{JEPA}/L_{Shuffle}
\]

这是 sanity，不是 representation geometry kNN。

---

# 10. A1 formal PASS 规则

Primary 仍然是 Persistence。

要求：

### Persistence

```text
Gain_ALL.ci95_low > 0
```

并且：

```text
H64 或 H256 至少一个 ci95_low > 0
```

同时：

```text
H64.ci95_high >= 0
H256.ci95_high >= 0
```

即不能在中长 horizon 上有显著负迁移。

H16：

```text
diagnostic only
```

不作为 GO 硬 gate。

### Mean / Shuffle sanity

要求 aggregate：

```text
GainMean_ALL.ci95_low > 0
GainShuffle_ALL.ci95_low > 0
```

否则说明 JEPA 可能没有超过 trivial/no-context baseline。

输出：

```text
a1_oos_prediction.json
a1_oos_prediction.csv
```

状态：

```text
PASS
FAIL
INCONCLUSIVE
```

定义：

```text
PASS:
  满足上述全部 formal requirements

FAIL:
  primary Gain_ALL.ci95_high <= 0
  或 Skill-style sanity baseline 显著不优

INCONCLUSIVE:
  point positive但 CI 跨 0，且没有显著负证据
```

不要因为单个 H16 negative point 就 FAIL。

---

# 11. A2：Strict-OOS Frozen Probe

保留当前 12 个 V0-exact outcomes：

```text
Return
MFE
MAE
RV

× H16/H64/H256
```

保持 exact historical V0 arithmetic/order。

---

## 11.1 Ridge protocol 保持

继续：

```text
ProbeFit:
  fit x/y normalizer
  initial Ridge fit

ProbeDev:
  select alpha independently per outcome

ProbeFit + ProbeDev:
  refit using frozen ProbeFit normalizers

ProbeTest:
  untouched final evaluation
```

Alpha：

```text
1e-4
1e-3
1e-2
1e-1
1
10
100
```

---

## 11.2 Primary baseline 不变

Primary baseline：

```text
MultiScaleSummary + causal context
```

其定义保持现有代码。

Secondary baselines：

```text
Unconditional mean
Context only
MinuteRaw24 + context
```

全部报告，但不作为 Stage-A primary gate。

---

## 11.3 Formal Skill

\[
Skill_j
=
1-
\frac{MSE_{Belief,j}}
{MSE_{MultiScaleSummary,j}}
\]

ProbeTest 上：

```text
equal commodity aggregate
contract-day paired bootstrap
2000 replicates
```

---

# 12. A2 formal PASS 规则

要求：

```text
Skill_ALL.ci95_low > 0
```

且：

```text
H64 或 H256 至少一个 ci95_low > 0
```

同时：

```text
H64.ci95_high >= 0
H256.ci95_high >= 0
```

H16 不作为 hard gate。

输出：

```text
a2_oos_probe.json
a2_probe_results.csv
a2_baselines.csv
a2_probe_hyperparams.json
```

状态：

```text
PASS
FAIL
INCONCLUSIVE
```

---

# 13. Strict Temporal OOS hard gate

Stage-A summary 必须明确检查：

```text
ProbeTest contract_uid 与 JEPA sampler population 交集 = empty
ProbeTest contract_uid 与 scaler population 交集 = empty
ProbeTest contract_uid 与 ProbeFit/ProbeDev 交集 = empty
stage_a_temporal_split_sha256 == checkpoint recorded hash
data manifest hash match
sampling population hash match
checkpoint fixed-budget final PASS
```

任何失败：

```text
STAGE_A_BLOCKED
```

而不是继续算分数。

---

# 14. Stage-A Final Decision

新增独立函数：

```python
stage_a_status(...)
```

输出只能有：

```text
STAGE_A_GO
STAGE_A_NO_GO
STAGE_A_INCONCLUSIVE
STAGE_A_BLOCKED
```

规则：

## GO

```text
temporal isolation PASS
A0 PASS
A1 PASS
A2 PASS
```

## NO_GO

以下任一出现明确负证据：

```text
A1 primary Gain_ALL.ci95_high <= 0
A2 Skill_ALL.ci95_high <= 0
A1/A2 formal baseline 明确显著失败
```

## INCONCLUSIVE

```text
A0/isolation 都正常
但 A1 或 A2 point > 0、CI 跨 0
没有明确负证据
```

即：

```text
证据不够
```

不是 GO，也不是 NO_GO。

## BLOCKED

```text
data leakage
split mismatch
checkpoint provenance mismatch
trivial collapse
nonfinite
protocol corruption
```

---

# 15. Stage-A Summary 输出

必须写：

```text
stage_a_summary.json
stage_a_summary.md
```

建议 JSON：

```json
{
  "question": "Does V1.1 learn a stable future-relevant predictive latent representation?",
  "status": "STAGE_A_GO",

  "strict_temporal_oos": "PASS",
  "latent_health": "PASS",
  "A1_future_prediction": "PASS",
  "A2_frozen_probe": "PASS",

  "Gain_ALL": {...},
  "Gain_H64": {...},
  "Gain_H256": {...},

  "Skill_ALL": {...},
  "Skill_H64": {...},
  "Skill_H256": {...},

  "formal_commodities": [...],
  "excluded_commodities": {...},

  "claim": "...",
  "not_claimed": [
    "identified Predictive State",
    "persistent HiddenMarketState",
    "RSSM value",
    "trading value",
    "cross-instrument generalization"
  ],

  "V0_comparison": "NOT_REQUIRED_FOR_STAGE_A",
  "A3_structure": "DEFERRED_UNTIL_STAGE_A_GO",
  "controls": "DEFERRED_UNTIL_STAGE_A_GO",
  "RB": "DEFERRED_UNTIL_STAGE_A_GO",
  "scaling": "DEFERRED_UNTIL_STAGE_A_GO"
}
```

---

# 16. V0 comparison 的地位

不要把旧 V0 comparison 放进 Stage-A GO。

原因：

旧 V0 和当前 V1.1 已经存在：

```text
不同数据组织
不同 IMC
不同 contract semantics
不同 multi-scale architecture
不同 training universe
```

直接比较旧历史数值不是干净的 single-variable experiment。

Stage A 先回答：

```text
Does V1 work in absolute terms?
```

只有 Stage A GO 后，才做：

```text
Why is V1 better / what architecture change matters?
```

这时：

```text
S-LateFusion matched control
```

比“直接拿旧 V0 数字比”更科学。

因此：

```text
V1 vs V0 = secondary historical reference
LateFusion vs Full S = later formal architecture attribution
```

---

# 17. kNN：不要恢复成 formal gate

原始路线曾把 latent kNN 当重要验证。

但后续已经证明：

\[
Z'=AZ
\]

可以保持 prediction 不变却改变欧氏/cosine neighborhood geometry。

所以：

```text
kNN may remain diagnostic
kNN MUST NOT decide Stage-A GO/NO-GO
```

本任务不要求实现 kNN。

如果以后加：

```text
predictive_geometry_diagnostic
```

必须明确：

```text
NON-GATING
```

---

# 18. A3 / controls / RB / scaling 怎么处理

不要删代码。

只从 `stage-a` command 中移除。

现有功能标记为：

```text
POST_STAGE_A
```

Stage A GO 后顺序才是：

```text
1. A3 Price/OI/Volume structure attribution
2. S LateFusion matched control
3. S MinuteOnly
4. S NoHistoricalWeekly
5. capacity S/M/L/XL
6. RB held-out
7. RSSM Stage B
```

注意：

真正原始主线里，RSSM 的价值问题仍在更后面。

---

# 19. Current `train-domain` command

为了 backward compatibility：

### 推荐方案

保留原：

```bash
train-domain
```

行为不动或重命名文档语义为：

```text
extended-train-domain
```

新增：

```bash
stage-a
```

作为当前正式 gate。

不要让 `stage-a` 隐式调用：

```text
structure_audit()
remove_sources()
RB
comparison
```

这样 Stage-A audit 应该明显更快、更简单。

---

# 20. W&B

Stage-A 可以记录 W&B，但本地仍 source of truth。

建议 run name：

```text
v1.1-S-stage-a-seed42
```

重点字段：

```text
stage_a/status

stage_a/a1_gain_all
stage_a/a1_gain_h64
stage_a/a1_gain_h256

stage_a/a2_skill_all
stage_a/a2_skill_h64
stage_a/a2_skill_h256

stage_a/belief_effective_rank
stage_a/target_h16_effective_rank
stage_a/target_h64_effective_rank
stage_a/target_h256_effective_rank
```

W&B failure nonfatal。

---

# 21. Required Code Changes

主要预期：

```text
market_jepa_v1_1/formal_training.py
market_jepa_v1_1/dataset.py
market_jepa_v1_1/sampler.py
market_jepa_v1_1/checkpoint.py

evaluate_market_jepa_v1_1.py

market_jepa_v1_1/evaluation/protocol.py
market_jepa_v1_1/evaluation/runner.py
market_jepa_v1_1/evaluation/metrics.py
market_jepa_v1_1/evaluation/probe.py

NEW:
market_jepa_v1_1/evaluation/stage_a.py
market_jepa_v1_1/evaluation/latent_health.py

configs/v1_1/*.yaml

tests/test_market_jepa_v1_1.py
tests/test_v11_formal_evaluation.py
NEW stage-a specific tests

docs/MARKET_JEPA_V1_1_STAGE_A_CORE_EVALUATION.md
```

不要修改：

```text
model architecture
IMC math
H16/H64/H256 objective
EMA tau
real-contract semantics
historical weekly lineage
Scheme-A 250K fixed budget
optimizer
BF16
RB semantics
```

---

# 22. Config 增加项

建议：

```yaml
evaluation:
  stage_a_temporal_oos:
    enabled: true
    version: stage_a_temporal_oos_v1
    split_unit: contract_uid
    split: [0.70, 0.15, 0.15]
    test_policy: latest_suffix
    min_contract_groups: 3
    probe_anchor_cap_per_commodity: 4096
```

正式 training hard gate：

```text
enabled MUST be true
```

注意：

不要让这个配置改变：

```text
max_optimizer_steps = 250000
```

---

# 23. Tests：P0 Acceptance Criteria

Codex 必须增加并通过以下测试。

## 23.1 Split chronology

Synthetic contracts：

```text
C1 C2 C3 C4 C5
```

要求：

```text
ProbeFit < ProbeDev < ProbeTest
ProbeTest is latest suffix
```

---

## 23.2 Contract atomicity

同一 contract_uid 有两个 re-main episodes：

```text
不得跨 partition
```

---

## 23.3 Test exclusion from sampler

对 full training sample stream：

```text
sampled contract_uid ∩ ProbeTest = empty
```

至少验证多个 sampling cycles。

---

## 23.4 Test exclusion from scaler

记录 scaler selected anchors：

```text
selected scaler contract_uid ∩ ProbeTest = empty
```

---

## 23.5 Resume split binding

checkpoint 保存后篡改：

```text
stage_a_temporal_split.json
```

resume 必须拒绝。

---

## 23.6 Causal sentinel leakage

修改 ProbeTest raw market values 后：

```text
earlier ProbeFit/Dev sample tensors unchanged
```

---

## 23.7 A1 formal population only test

构造：

```text
ProbeFit bad
ProbeDev bad
ProbeTest good
```

formal A1 必须只由 ProbeTest 决定。

---

## 23.8 A2 formal population only test

同上：

```text
ProbeTest only
```

决定 final Skill。

---

## 23.9 Mean baseline fit isolation

修改 ProbeTest targets：

```text
TrainMean baseline vector MUST NOT change
```

---

## 23.10 Block-shuffle isolation

Block-shuffle source只能来自 ProbeFit，且 deterministic seed42。

---

## 23.11 Latent collapse

constant belief：

```text
STAGE_A_BLOCKED
```

nonconstant full-rank-ish synthetic：

```text
A0 PASS
```

不要测试硬编码“effective rank 必须超过某个大数”。

---

## 23.12 Stage-A status

覆盖：

```text
GO
NO_GO
INCONCLUSIVE
BLOCKED
```

---

## 23.13 Stage-A does not execute deferred audit

mock：

```text
structure_audit
remove_sources
RB loader
```

运行 `stage-a` 时必须 zero calls。

---

## 23.14 RB untouched

运行 Stage-A：

```text
held_out RB bar files/model evaluation = NOT READ
rb_test_consumption ledger = NOT CREATED
```

---

## 23.15 Scheme A unchanged

确认：

```text
max_optimizer_steps=250000
warmup_optimizer_steps=12500
effective_batch=128
target_samples_seen=32000000
```

不因 temporal holdout 数量变化而改变。

---

## 23.16 Full suite

最后必须：

```text
new Stage-A tests PASS
existing V1.1 tests PASS
formal evaluation tests PASS
Scheme-A tests PASS
full pytest PASS
compileall PASS
CLI --help PASS
git diff --check PASS
```

---

# 24. Migration / Existing Checkpoints

旧 checkpoint：

```text
没有 stage_a_temporal_split_sha256
```

不能声称 strict Stage-A OOS。

允许：

```text
legacy diagnostic evaluation
```

但 `stage-a` formal command 必须拒绝：

```text
STAGE_A_BLOCKED_LEGACY_CHECKPOINT_NO_TEMPORAL_HOLDOUT
```

当前尚未开始的新 formal Scheme-A S 应使用新 split 从 scratch 训练。

已经训练中的旧 epoch-budget M：

```text
继续只作为 LEGACY_EPOCH_BUDGET_DIAGNOSTIC
```

不要迁移。

---

# 25. Codex 最终报告必须回答

完成后输出：

```text
1. 当前 Stage-A scientific question
2. temporal split exact policy
3. formal Stage-A commodities
4. excluded commodities + reasons
5. proof ProbeTest is absent from sampler
6. proof ProbeTest is absent from scaler
7. split SHA and checkpoint binding
8. A0 latent health implementation
9. A1 baselines and exact gate
10. A2 baseline and exact gate
11. STAGE_A_GO logic
12. deferred features not executed
13. Scheme A budget unchanged
14. files changed
15. tests executed
16. full pytest result
17. compileall result
18. git diff --check
19. exact formal S training command
20. exact Stage-A audit command
```

---

# 26. Reviewer Final Check

本需求经过以下逻辑复查：

## 26.1 是否仍然把 V1 > V0 当核心？

没有。

核心是 V1 absolute Stage-A evidence。

## 26.2 是否真的有 strict temporal OOS？

新设计有。

ProbeTest contract groups 在 JEPA pretraining 和 scaler fitting 中都不可见。

## 26.3 是否把 RB 提前？

没有。

RB 完全 deferred。

## 26.4 是否提前训练 controls / scaling？

没有。

Stage A 只要求 Full S。

## 26.5 是否用 kNN 作为 gate？

没有。

保持后续理论结论：geometry diagnostic only。

## 26.6 是否把 Predictive Representation 偷换成 Predictive State？

没有。

formal claim 被限制在：

```text
future-relevant predictive latent representation
```

## 26.7 是否破坏 Scheme A fixed sample budget？

没有。

仍然：

```text
250K optimizer updates
32M exposures
```

## 26.8 是否留下已有高级 audit 代码？

保留。

仅降级成 POST_STAGE_A，不删除。

---

# 27. Final Required Workflow

完成这次修改以后，实际研究顺序应变成：

```text
1. 生成/冻结 Stage-A temporal split
2. Full S 从 scratch 训练 250K steps
3. 运行 stage-a
4. 查看：
     A0 latent health
     A1 strict-OOS JEPA prediction
     A2 strict-OOS frozen probe
5. 得到：
     STAGE_A_GO
     或 NO_GO
     或 INCONCLUSIVE
6. 到这里停下来人工 review
```

只有 `STAGE_A_GO` 后，才讨论：

```text
A3
LateFusion
MinuteOnly
NoHistoricalWeekly
M/L/XL
RB
RSSM
```

---

# Reviewer Verdict

```text
REFOCUS REQUIRED
```

当前完整 evaluation framework 不必推倒重写，但当前正式入口必须新增一个真正的：

```text
Stage-A Core Gate
```

并且 **temporal ProbeTest 必须在训练前冻结和隔离**。

否则现有 `ProbeTest` 只能证明 Ridge probe 没见过测试标签，
不能证明 JEPA representation 在未见未来 contract episode 上泛化。
