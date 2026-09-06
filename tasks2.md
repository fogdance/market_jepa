

你现在要完成 **Market World Model V1.0 — Cross-Scale Conditional Representation** 的开发。

开始前必须完整阅读：

```text
MARKET_WORLD_MODEL_V0_TO_V1_ARCHITECTURE_RATIONALE.md
MARKET_INVARIANT_COORDINATES_MATHEMATICAL_DESIGN.md
```

然后检查现有仓库中：

```text
V0 / design_version 0.6.1
Minute Market Encoder
Minute Context GRU
Daily GRU
Weekly GRU
Fusion
EMA target encoder
H16/H64/H256 predictors
training/evaluation pipeline
```

现有实现。

## 0. 本次任务的边界

本次任务是：

$$
\boxed{\text{开发 V1 架构}}
$$

不是正式训练实验。

最终必须完成：

```text
architecture
config
forward path
EMA target compatibility
training compatibility
unit tests
integration tests
small smoke test
architecture report
self-review
```

但**不要启动正式多品种训练**。

不要读取或消耗未来正式 held-out RB Validation/Test 来调模型。

不要自动开始 V0 vs V1 benchmark。

---

# 1. V0 必须完整保留

V0 是正式 baseline。

禁止破坏：

```text
V0 architecture
V0 config
V0 checkpoint loading
V0 evaluation
V0 training behavior
design_version 0.6.1
```

优先通过：

```text
new model class
new config
shared reusable components
```

实现 V1。

不要为了代码“漂亮”大规模重构 V0。

如果共享代码可能改变 V0 numerical behavior，宁可保留少量重复代码。

要求现有 V0 tests 继续全部 PASS。

---

# 2. V1 的核心研究假设

V0：

$$
Minute\rightarrow Z_m
$$

$$
Context\rightarrow Z_c
$$

$$
Daily\rightarrow Z_d
$$

$$
Weekly\rightarrow Z_w
$$

然后：

$$
Fusion(Z_m,Z_c,Z_d,Z_w)
$$

属于：

$$
\boxed{
Encode
\rightarrow
Compress
\rightarrow
Interact
}
$$

V1 必须变成：

$$
\boxed{
Encode
\leftrightarrow
Condition
\leftrightarrow
Interact
\rightarrow
FinalCompression
}
$$

核心要求：

> Minute / Daily / Weekly 不能在看到其他尺度以前全部压成一个单向量。

---

# 3. V1.0 不追求“大模型”

本次不要同时研究 capacity scaling。

保持：

```text
d_model = 256
heads = 8
FFN = 1024
```

为核心尺度。

V1 可以因为 cross-scale blocks 比 V0 多一些参数，但目标不是占满 16GB 显存。

正式报告必须给出：

```text
V0 trainable parameter count
V1 trainable parameter count
difference
ratio
```

禁止擅自：

```text
d_model = 512/1024
12/24 transformer layers
100M parameter model
```

模型大小以后单独实验。

---

# 4. V1 输入原则

V1 必须兼容 IMC 输入。

架构本身不要硬编码特定 feature 数量。

使用 config 提供：

```text
minute_market_dim
minute_context_dim
daily_market_dim
daily_context_dim
weekly_market_dim
weekly_context_dim
```

IMC 数据准备属于 upstream preprocessing。

模型不负责重新计算 IMC。

---

# 5. 禁止 Commodity ID

V1.0 第一版禁止：

```text
commodity_id
commodity_embedding
commodity-specific encoder
commodity-specific head
commodity-specific normalization inside model
```

最终目标是：

$$
FG+SA+JM+SH+SP
\rightarrow RB
$$

所以模型不能依赖显式品种身份。

增加自动 assert / test，确认 model state_dict 中不存在 commodity embedding。

---

# 6. Minute Market × Context 提前交互

V0 的：

```text
Minute Market Transformer
Minute Context GRU
```

分离结构在 V1 中取消。

但为了保持 EMA target 的 market-only semantics，采用下面的设计。

## Minute token

分别投影：

$$
m_t=W_m x_t^{market}
$$

$$
c_t=W_c x_t^{context}
$$

在线 encoder token：

$$
\boxed{
u_t=m_t+c_t+e_{minute}
}
$$

也可以在代码内部使用等价的 projection + normalization，但不要使用一个完全无法关闭 context 的黑盒 concat MLP。

要求能够明确得到：

```text
market-only token
market + context token
```

因此：

### Online encoder

使用：

$$
m_t+c_t
$$

### EMA target

严格只使用：

$$
m_t
$$

即：

```text
future context must NOT enter target
```

保持 V0 的防 shortcut 原则。

---

# 7. Minute local encoder

V1 继续使用 Transformer 作为 minute local encoder。

冻结第一版：

```text
layers = 4
d_model = 256
heads = 8
FFN = 1024
```

保留：

```text
CLS token
```

但 V1 与 V0 最大区别是：

> 不允许只把 CLS 输出交给后续模块。

必须保留：

$$
\boxed{
MinuteTokenSequence
}
$$

供 cross-scale interaction 使用。

例如：

```text
[CLS, M1, M2, ..., M512]
```

经过 local encoder 后整条 sequence 都保留。

---

# 8. Daily / Weekly local encoder

第一版尽量复用 V0 的 GRU 思路，避免同时改变过多变量。

Daily：

```text
2-layer GRU
hidden = 128
```

Weekly：

```text
2-layer GRU
hidden = 128
```

但不要只返回 final hidden state。

必须返回：

$$
D_1,D_2,...,D_n
$$

以及：

$$
W_1,W_2,...,W_k
$$

整个 causal sequence。

然后分别投影：

$$
128\rightarrow256
$$

进入统一 cross-scale token space。

---

# 9. Scale embedding

加入三个 learned scale embeddings：

```text
minute
daily
weekly
```

每个：

$$
e_s\in R^{256}
$$

分别加到对应 scale token。

这是 scale identity，不是 commodity identity。

允许。

---

# 10. V1.0 Cross-Scale State Tokens

这是 V1 的核心。

定义：

```text
K = 8
```

个 learned cross-scale latent/state tokens：

$$
S^{(0)}
=
[s_1,\ldots,s_8]
$$

每个：

$$
s_i\in R^{256}
$$

其中：

```text
s1 = BELIEF token
s2..s8 = auxiliary state slots
```

最终：

$$
B_t
$$

由 `s1` 产生。

---

# 11. Cross-scale interaction 不使用全量 all-to-all attention

禁止把：

```text
512 minute
× daily
× weekly
```

全部做暴力 full pairwise cross-attention。

使用 latent/state tokens 作为信息交换媒介。

定义：

$$
T=
[
MinuteTokens;
DailyTokens;
WeeklyTokens
]
$$

---

# 12. Cross-Scale Block

冻结 V1.0：

```text
cross_scale_rounds = 2
```

每轮包含两个方向。

## Step A — State reads all scales

$$
S
\leftarrow
S+
MHA(
LN(S),
LN(T),
LN(T)
)
$$

然后：

$$
S
\leftarrow
S+
FFN(LN(S))
$$

---

## Step B — Scale tokens read shared state

$$
T
\leftarrow
T+
MHA(
LN(T),
LN(S),
LN(S)
)
$$

然后：

$$
T
\leftarrow
T+
FFN(LN(T))
$$

注意：

这里的 token-feedback block 可以对统一 `T` 使用 shared parameters。

不要为：

```text
minute
daily
weekly
```

各自复制一套巨大 feedback network。

scale embedding 已经提供尺度身份。

---

## 然后进入下一轮

第二轮：

$$
S\leftarrow Attend(S,T)
$$

此时的 \(T\) 已经包含其他尺度通过 \(S\) 反馈的信息。

因此形成：

$$
Minute
\leftrightarrow
State
\leftrightarrow
Daily
$$

以及：

$$
Minute
\leftrightarrow
State
\leftrightarrow
Weekly
$$

的间接双向信息交换。

---

# 13. 为什么使用两轮

一轮：

```text
S reads Minute/Daily/Weekly
```

只能形成汇总。

加入：

```text
T reads S
```

以后，高尺度信息能够反馈到 minute tokens。

再做第二轮：

```text
S reads conditioned T
```

才真正形成：

$$
\boxed{
Condition\ while\ representing
}
$$

而不是换一种形式的 late fusion。

不要把 `cross_scale_rounds=2` 擅自改成 6、12 等。

以后另做实验。

---

# 14. Final belief state

第二轮结束后：

$$
S^{final}
$$

取第一个 BELIEF token：

$$
s_1
$$

经过：

```text
LayerNorm
Linear(256 -> 256)
```

得到：

$$
\boxed{
B_t\in R^{256}
}
$$

不要 concatenate 所有 branch 的 final vector。

V1 不再使用 V0 的：

```text
544 -> 512 -> 256
```

late-fusion MLP 作为主路径。

---

# 15. JEPA objective 保持不变

这是非常重要的实验控制。

禁止这次加入：

```text
CDF
CME
RSSM
Actor
Critic
Reward
PnL loss
supervised future labels
classification heads
regime labels
```

仍然是：

$$
B_t
\xrightarrow{P_h}
Z^{EMA}_{t+h}
$$

其中：

```text
H16
H64
H256
```

保持原 JEPA predictors。

尽量复用 V0：

```text
256 -> 512 -> 256
```

predictor。

---

# 16. EMA target semantics 必须保持 V0 原则

这是重点 review 项。

EMA target：

$$
Z^{EMA}_{t+h}
$$

不得包含 future：

```text
clock
weekday
session context
daily future context
weekly future context
commodity ID
```

V1 target 仍然以：

$$
\boxed{
future market-only minute representation
}
$$

为目标。

优先复用/适配 V0 target semantics，而不是重新设计一个新 target task。

原因：

> V0 vs V1 要尽量只比较 representation architecture。

---

# 17. EMA 更新

保持：

```text
tau = 0.996
```

只 EMA target encoder 所需的 market-only minute path。

必须检查：

```text
target requires_grad = false
target gradients = None
```

---

# 18. Mask / Padding

Minute、Daily、Weekly sequence 长度可能不同。

所有：

```text
local encoder
cross-attention
state attention
```

都必须正确支持：

```text
padding mask
causal data construction
```

注意：

Cross-scale attention 本身处理的都是 anchor 时刻已经观测到的历史数据，所以不需要人为把过去 token 之间再限制成 decoder-style causal attention，除非现有 local encoder 本身的设计要求。

但：

$$
\boxed{\text{任何未来 token 都不能进入输入}}
$$

必须由 dataset/integrity tests 保证。

---

# 19. V1 architecture class

建议新增类似：

```text
MarketJEPAV1
CrossScaleStateEncoder
CrossScaleInteractionBlock
```

具体名字可以适应仓库风格。

config 建议：

```text
configs/market_jepa_v1.yaml
```

model version：

```text
design_version = "1.0"
```

不要覆盖：

```text
0.6.1
```

---

# 20. V1 config 至少需要显式记录

```yaml
design_version: "1.0"

d_model: 256
num_heads: 8
ffn_dim: 1024

minute_layers: 4

daily_gru_layers: 2
daily_gru_hidden: 128

weekly_gru_layers: 2
weekly_gru_hidden: 128

num_state_tokens: 8
cross_scale_rounds: 2

belief_dim: 256

use_commodity_embedding: false
```

不要把这些关键结构藏成代码 magic constants。

---

# 21. 必须增加 Architecture Information Flow Tests

普通 shape tests 不够。

必须验证 V1 确实解决了它声称解决的问题。

---

## Test A — Daily affects belief

固定：

```text
minute
weekly
context
```

只改变 Daily。

要求：

$$
B_t^{(1)}\neq B_t^{(2)}
$$

---

## Test B — Weekly affects belief

固定其他输入，只改变 Weekly。

要求：

$$
B_t^{(1)}\neq B_t^{(2)}
$$

---

## Test C — Context affects online belief

固定 market values，只改变 minute context。

要求：

$$
B_t^{online,1}
\neq
B_t^{online,2}
$$

---

## Test D — Context does NOT affect target

构造相同 future market data，不同 future context。

EMA target 必须：

$$
\boxed{
Z^{target,1}=Z^{target,2}
}
$$

在数值 tolerance 内完全一致。

这项必须 PASS。

---

# 22. Cross-scale feedback test

这是 V1 最重要的新 test。

第一轮 state read 完成后：

改变 Daily tokens。

检查第二轮进入 state 前的 Minute token representation 是否改变。

形式上必须验证：

$$
MinuteRepresentation^{after-feedback}
=
f(Minute,Daily,Weekly)
$$

而不是：

$$
f(Minute)
$$

也就是说：

> Daily/Weekly 能够在最终压缩之前影响 Minute representation。

这是 V1 与“换了个 latent pooling 的 late fusion”之间的核心区别。

---

# 23. Gradient connectivity test

构造一个：

$$
loss=B_t.sum()
$$

反向传播。

必须确认：

```text
non-CLS minute tokens
daily intermediate tokens
weekly intermediate tokens
minute context projection
cross-scale state tokens
```

全部存在 finite non-zero gradient。

目的不是测试训练效果。

只是确认信息路径真实存在。

---

# 24. Cross-scale ablation sanity test

支持 config：

```text
cross_scale_rounds = 0
```

仅用于 test/debug。

当：

```text
cross_scale_rounds = 0
```

模型不得偷偷调用 cross-scale block。

但正式 V1 config：

```text
cross_scale_rounds = 2
```

不要把 `0` 当正式模型。

---

# 25. Commodity shortcut test

搜索：

```text
state_dict keys
model constructor args
forward args
```

正式 V1 不允许：

```text
commodity
symbol_id
instrument_embedding
```

作为模型输入。

如果 dataset 中存在 commodity 字段，用于 sampling/logging 可以，但不得送进 model forward。

---

# 26. Parameter-count review

输出：

```text
V0 trainable params
V1 trainable params

Minute local encoder params
Daily encoder params
Weekly encoder params
Cross-scale params
Predictor params
EMA params separately
```

不要把 EMA 参数计入 trainable count。

如果 V1 trainable params 意外超过 V0 的 3 倍：

```text
STOP
```

先检查设计。

V1.0 首版不应该通过暴力扩大参数获得优势。

---

# 27. GPU smoke test

如果运行环境有 CUDA：

使用真实 feature dimensions + synthetic 或 Train-only small batch。

至少测试：

```text
batch = 2
batch = 8
```

执行：

```text
forward
JEPA loss
backward
optimizer step
EMA update
```

记录：

```text
peak allocated VRAM
peak reserved VRAM
forward time
backward time
```

不要为了“利用更多显存”提高 batch。

这里只验证能够稳定运行。

---

# 28. Small real-data smoke test

如果现有 Train 数据可直接读取：

只使用：

```text
training population
```

运行一个极小 smoke：

```text
100-500 optimizer steps
```

目的：

```text
loss finite
no NaN
no gradient explosion
EMA works
checkpoint save/load works
```

不要根据 smoke loss 判断模型好坏。

不要接触正式 RB Validation/Test。

---

# 29. Checkpoint compatibility

V1 checkpoint 必须包含：

```text
design_version
architecture config
feature dimensions
state token count
cross scale rounds
EMA state
optimizer state
training step
```

V0 checkpoint loader 继续正常工作。

V1 loader 对错误 design_version 必须给清晰错误，而不是 silent partial load。

---

# 30. 保存中间 representation 的 debug API

为了未来 scientific audit，V1 forward/debug 模式应能可选返回：

```text
minute_local_tokens
daily_local_tokens
weekly_local_tokens

state_tokens_after_round_1
tokens_after_feedback_round_1

state_tokens_after_round_2
final_belief
```

默认训练时不要保存这些大 tensor。

例如：

```text
return_intermediates=False
```

只在 audit/debug 打开。

这是以后验证 cross-scale 学到了什么的重要接口。

---

# 31. 不要把 attention weights 当成“解释”

可以提供 debug attention maps，但报告中明确：

```text
attention weight != causal importance
```

未来判断：

```text
OI use
Volume use
Price × OI × Volume
```

仍以 intervention / relation destruction 为主。

---

# 32. 测试

至少增加：

```text
test_market_jepa_v1_shapes
test_market_jepa_v1_no_future_context_target
test_market_jepa_v1_daily_changes_belief
test_market_jepa_v1_weekly_changes_belief
test_market_jepa_v1_context_changes_online
test_market_jepa_v1_cross_scale_feedback
test_market_jepa_v1_gradients
test_market_jepa_v1_no_commodity_embedding
test_market_jepa_v1_ema_frozen
test_market_jepa_v1_checkpoint_roundtrip
test_market_jepa_v0_regression
```

运行整个相关测试集。

不能只跑新 tests。

---

# 33. 开发前先生成 implementation plan

在改代码以前先输出：

```text
docs/MARKET_JEPA_V1_IMPLEMENTATION_PLAN.md
```

里面必须列：

### Existing V0 components

实际文件、class、function。

### Reused unchanged

哪些直接复用。

### New components

哪些新增。

### Modified shared components

为什么必须修改。

### V0 regression risk

可能影响 V0 的位置。

### Data contract

V1 forward 具体 tensor shape。

### Information-flow diagram

明确画出：

```text
Minute Market + Context
        ↓
Minute Local Tokens
        ↘
Daily Local Tokens
          → State Tokens ↔ Scale Tokens
Weekly Local Tokens
        ↗
        ↓
     Belief B_t
```

不要 plan 完就停止。

完成 review 后继续实现。

---

# 34. 开发完成后做一次独立 self-review

不要只说 tests PASS。

从研究目标重新检查代码：

### Review question 1

V1 是否真的：

$$
CrossScaleInteraction
\prec
FinalCompression
$$

还是实际上换一种方式做 late fusion？

### Review question 2

Daily/Weekly 是否能够在最终 belief 形成前改变 minute representation？

### Review question 3

Market × Time 是否在 minute local representation 阶段已经交互？

### Review question 4

EMA target 是否仍然 market-only？

### Review question 5

是否偷偷增加了 commodity shortcut？

### Review question 6

V1 是否无意中变成参数量远大于 V0 的模型？

### Review question 7

是否改变了 JEPA objective，从而破坏以后 V0/V1 因果比较？

发现问题必须先修复，再出报告。

---

# 35. 输出设计文档

开发完成后生成：

```text
docs/MARKET_JEPA_V1_ARCHITECTURE_IMPLEMENTED.md
```

必须包含：

1. 实际最终架构；
2. 每个 tensor shape；
3. local encoders；
4. state tokens；
5. 两轮 cross-scale interaction；
6. target encoder；
7. 参数量；
8. V0 vs V1 实际代码级区别；
9. 哪些设计与最初 proposal 有变化；
10. 为什么变化。

---

# 36. 输出开发报告

生成：

```text
artifacts/evaluation/v1_architecture_development/
    summary.md
    summary.json
    parameter_counts.json
    smoke_test.json
    test_results.txt
```

`summary.md` 必须直接回答：

1. V0 是否完全保留？
2. V1 Market×Context 是否 early interaction？
3. Minute/Daily/Weekly 是否在最终压缩前交互？
4. Daily 是否能反馈影响 Minute representation？
5. Weekly 是否能反馈影响 Minute representation？
6. target 是否严格 market-only？
7. 是否存在 commodity ID？
8. V0/V1 trainable parameters 分别多少？
9. CUDA smoke peak VRAM 多少？
10. 所有 tests 是否 PASS？
11. 是否具备进入正式 IMC + cross-commodity benchmark 的条件？

---

# 37. 本次开发成功条件

只有全部满足才可标记：

```text
V1_ARCHITECTURE_IMPLEMENTATION_PASS
```

必须满足：

$$
\boxed{\text{V0 regression PASS}}
$$

$$
\boxed{\text{Market×Context early interaction PASS}}
$$

$$
\boxed{\text{Cross-scale pre-compression interaction PASS}}
$$

$$
\boxed{\text{Daily/Weekly → Minute feedback PASS}}
$$

$$
\boxed{\text{Target market-only PASS}}
$$

$$
\boxed{\text{No commodity embedding PASS}}
$$

$$
\boxed{\text{Gradient connectivity PASS}}
$$

$$
\boxed{\text{Checkpoint roundtrip PASS}}
$$

$$
\boxed{\text{Smoke training PASS}}
$$

否则：

```text
V1_ARCHITECTURE_IMPLEMENTATION_FAIL
```

并明确列出失败项。

---

# 38. 不允许做的事情

本次禁止：

```text
正式训练 FG+SA+JM+SH+SP
读取 RB test 调架构
根据 Validation 调 cross_scale_rounds
根据结果修改 IMC
加入 RSSM
加入 CDF
加入 CME
加入 Actor/Critic
加入 trading reward
加入 PnL
加入 regime labels
加入人工 Price/OI 四象限标签
加入 commodity embedding
做大模型 scaling
自动启动 V0 vs V1 正式 benchmark
```

---

# 39. 最终停止位置

完成：

```text
implementation
tests
self-review
small smoke test
documentation
```

以后停止。

不要自动开始正式训练。

最终只告诉我：

```text
V1_ARCHITECTURE_IMPLEMENTATION_PASS / FAIL
```

并给出：

```text
V0 params
V1 params
peak VRAM
tests passed/failed
关键文件
任何未解决风险
```

---

这版指令我建议作为 **V1.0 最小科学版本**。它有一个很重要的好处：我们没有同时把 `d=256` 改成 `512`，所以以后 V1 如果赢 V0，更有资格把差异归因到：

$$
\boxed{\text{信息流和 cross-scale conditioning}}
$$

而不是简单归因于“V1 参数更多”。

另外，这里的两轮：

$$
S\leftarrow T
$$

$$
T\leftarrow S
$$

$$
S\leftarrow T
$$

我认为是这次最值得冻结的设计。它比单纯用几个 latent token 把 minute/day/week 汇总一下更重要，因为**只有 feedback 之后，高时间尺度的信息才真正有机会在最终压缩前改变分钟级 representation**。这才真正对应我们现在发现的 V0 问题。

