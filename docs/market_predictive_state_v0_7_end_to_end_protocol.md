# Market Predictive State V0.7 — End-to-End Final Frozen Protocol

**协议版本：0.7.1**  
**状态：Frozen for implementation**  
**最后更新：2026-09-04**  
**实验类型：End-to-End Predictive-State**  
**Final Test consumed：false**

## 修订记录

| 版本 | 日期 | 状态 | 变更 |
| --- | --- | --- | --- |
| 0.7.0 | 2026-09-04 | 已替代 | 首次整理 End-to-End Predictive-State 协议，进入评审。 |
| 0.7.1 | 2026-09-04 | Frozen for implementation | 冻结 RFF 公式、PCG64 调用顺序与 numerical regression；Market/Context 改为独立同 seed run；Gate 2 冻结 exact top-K 与 tie rule；补充 target-end、budget ceiling、clean-tree 和 claim 限制。 |

## 1. 唯一科学问题与 claim boundary

本实验只回答：如果整个 market encoder 从随机初始化开始，直接接受 Predictive-State Objective，是否能在 2023–2024 Development period 学到同时具有稳定 conditional-distribution information 和 predictive geometry 的 `Y-Predictive State Candidate`？

V0.7 不是 JEPA 实验。禁止加载旧 epoch49 权重，禁止 EMA target encoder、JEPA predictor、JEPA loss、RSSM、Actor/Critic、Reward/PnL、交易或实时系统。

若 Development GO，唯一允许的结论是：

> End-to-End Predictive-State Objective produced a Y-Predictive State Candidate with statistically positive aggregated 2023–2024 Development conditional-distribution information and predictive geometry beyond context-only and unconditional/random baselines under the frozen protocol.

不得声称已经证明完整 HiddenMarketState、predictive sufficiency、RSSM 有效性、策略有效性或盈利能力。若 NO-GO，只说明当前 12D Y、CME/RFF target、fixed-window encoder 和冻结训练协议未在 Development period 达到预注册要求，不能据此否定整个 Predictive-State theory。

## 2. 数据边界与因果样本

唯一行情源、feature definitions、trading-day inference、causal minute/daily/weekly snapshot、H16/H64/H256 outcome、anchor stride=1 和 future-target boundary 全部沿用 Market-JEPA V0.6.1；不得删除、修复、winsorize 或改写异常行情。

| Population | 日期 | 用途 |
| --- | --- | --- |
| Inner Fit | 2018-01-02–2021-12-31 | Inner preprocessing 与训练 |
| Inner Dev | 2022-01-01–2022-12-31 | 只做 epoch budget selection |
| Final Train | 2018-01-02–2022-12-31 | Final preprocessing 与从头训练 |
| Development Evaluation | 2023-01-01–2024-12-31 | Final checkpoints 完成后一次性评估 |
| Final Test | 2025-01-01–2025-12-02 | 本任务禁止读取、构造、导出或评估 |

定义：

```text
target_end_index = anchor_index + 256
```

一个 anchor 属于某 population，必须同时满足：

1. anchor 的 inferred `trading_day` 位于该闭区间；
2. `target_end_index` 在 source minute array 中存在；
3. `trading_day[target_end_index] <= population_end`。

这里的 `+256` 是 source minute row index 运算，不是 timestamp 加 256 个 wall-clock minutes。

历史 context 可以因果跨越 population 起点，所有 input source index 必须 `<= anchor_index`；未来 target 不能跨越 population 终点。H 表示后续真实 minute rows，不按 wall-clock 补齐。

### 2.1 Nested input preprocessing

Inner stage 只能使用 2018–2021 拟合：

- minute market normalizer；
- minute context normalizer；
- daily market/context normalizer；
- weekly market/context normalizer。

2022 只能使用上述统计量 transform。Final stage 在 epoch budgets 已锁定后重新使用 2018–2022 拟合全部 input normalizers；2023–2024 只能 transform。

Normalizer population 保持 V0 定义：minute/context 使用 fit range 内全部 minute rows；daily/weekly 分别使用 fit range 内 completed tokens 与该范围每根 minute 对应的 causal partial token。零方差维度 std 置 1。所有 normalizer 保存 feature names、mean、std、fit count 和 SHA-256。

Development Dataset 只能在两个 Final checkpoints 均已完成后构造。程序不得提供会构造 2025 Dataset 的本任务执行路径。

## 3. Future variable 与固定 CME/RFF target

每个 anchor 的 future variable 固定为：

```text
Y = concat(
  H16  [Return, MFE, MAE, RV],
  H64  [Return, MFE, MAE, RV],
  H256 [Return, MFE, MAE, RV]
) ∈ R12
```

每个 preprocessing stage 在自身 fit population 对 Y 逐维拟合 mean/std，并定义 `Y_std=(Y-mean)/std`。Inner 的 Y scaler 只用 2018–2021，Final 的 Y scaler 只用 2018–2022。

对标准化后的 joint 12D Y，固定使用三个等权 Gaussian kernels。所有 RNG 都显式构造为：

```python
np.random.Generator(np.random.PCG64(seed))
```

设 fit population 样本数为 `n`，pair sampling 必须按下列调用顺序执行，以保持与 V0.7-Frozen 已生成 target space 一致：

```python
pair_rng = np.random.Generator(np.random.PCG64(4242))
left = pair_rng.integers(0, n, size=100_000)
offset = pair_rng.integers(1, n, size=100_000)
right = (left + offset) % n
pairs = np.stack([left, right], axis=1)
```

因此 `left != right` 恒成立，且每个 left 对应的 right 在其他 `n-1` 个样本中均匀取值。禁止改成 rejection redraw 或其他虽然分布等价、但 RNG stream 不同的实现。

令三个 bandwidth 按 `[0.5*sigma0, 1.0*sigma0, 2.0*sigma0]` 排列。对第 `l` 个 bandwidth `sigma_l`：

\[
W_l\sim N(0,\sigma_l^{-2}I_{12}),\qquad
b_l\sim Uniform(0,2\pi),
\]

其中 `W_l` 的 shape 为 `[12,1024]`，`b_l` 的 shape 为 `[1024]`。单块特征定义为：

\[
\phi_l(y)=\sqrt{\frac{2}{1024}}\cos(yW_l+b_l).
\]

最终：

\[
\boxed{
\phi(y)=\frac{1}{\sqrt3}
[\phi_1(y),\phi_2(y),\phi_3(y)]
}
\in\mathbb{R}^{3072}.
\]

其内积近似：

\[
\phi(y_i)^T\phi(y_j)
\approx
\frac13\sum_{l=1}^{3}
\exp\left(-\frac{\|y_i-y_j\|_2^2}{2\sigma_l^2}\right).
\]

RFF RNG 必须按以下顺序使用一个 PCG64 stream：先按 bandwidth 顺序分别生成三个 `W_l`，再一次生成 shape `[3,1024]` 的全部 phases：

```python
rff_rng = np.random.Generator(np.random.PCG64(4243))
W = np.stack([
    rff_rng.normal(0.0, 1.0 / sigma_l, size=(12, 1024))
    for sigma_l in bandwidths
]).astype(np.float32)
b = rff_rng.uniform(0.0, 2.0 * np.pi, size=(3, 1024)).astype(np.float32)
```

数值路径同样冻结：Y mean/std 用 CPU float64 计算，标准化后的 Y 保存为 float32；pair distance 和 exact kernel 用 float64；W/b 与 `phi(Y)` 用 float32；audit 的 RFF inner product 累加前转换为 float64。

概念步骤固定为：

1. 使用上述 seed=4242 PCG64 算法抽取 100000 个非自身 pairs；
2. `sigma0` 为 pair Euclidean distance 中位数；
3. bandwidths 固定为 `0.5*sigma0`、`sigma0`、`2*sigma0`；
4. 每个 bandwidth 使用 1024 个 RFF，总维度 3072；
5. 使用上述 seed=4243 PCG64 stream；
6. 每块使用标准 RFF scaling，并额外乘 `1/sqrt(3)`。

每个 stage 的 `mu_phi` 是其 fit population 上 `phi(Y)` 的逐维均值。Inner Dev 和 Development 不得参与 Y scaler、kernel、RFF 或 `mu_phi` 的确定。

Inner 与 Final 均使用生成 kernel bandwidth 的同一组 100000 Train-only pairs 输出 exact mixture-kernel 与 RFF inner product 的 RMSE、MAE、correlation；non-finite 或确认的实现错误必须 HARD FAIL，不允许换 seed、m、bandwidth 或 kernel。

所有 array hash 统一定义为：

```text
SHA256(dtype.str ASCII || int64 shape bytes || C-contiguous array bytes)
```

每个 stage 保存 `pairs`、`W`、`b` 的 hash。Final 2018–2022 target construction 必须通过下列 deterministic regression：

```text
sigma0 reference       = 3.0125319812116054
RFF RMSE reference     = 0.011788527790449407
RFF MAE reference      = 0.009348590317651785
RFF correlation        = 0.998792590439722

Y mean hash            = 9158a8251a3b52c82fd2d9b88f979a95fdbcd0cb3b5321070af6c1e7858eea8e
Y std hash             = 58145f4c51bdd64e38c2c073b8e9b1a71dde12e46e8e3789289ffdd316143b96
pair-index hash        = d6d2951c3a2bb53b268ca516278f9b725e772309b7ed1d5b171a5489a7ae9990
W hash                 = 0f696e854d938331011b317b129f6568ff7b7960f3f8ad9bf2d118cfb5a2c50b
b hash                 = ef45a53d4f680e207e2b9ddcd0834f48b9c0161237be7bd3ed95e799c1790a6f
```

五个 array hash 必须完全相同；四个 scalar audit values 使用 `rtol=1e-9, atol=1e-12` 比较。任何一项失败都必须在 Final training 前 HARD FAIL 并人工检查；不得根据 Inner Dev 或 Development 调整 target construction。Inner 2018–2021 独立 fit，因此只保存自己的 hashes/audit，不与上述 Final reference 比较。

## 4. 模型与唯一 objective

### 4.1 Market model

模型从 seed42 随机初始化，只复用 V0 online encoder/fusion 架构：

```text
Minute Market Transformer
Minute Context GRU
Daily GRU
Weekly GRU
        ↓
      Fusion
        ↓
     B_t ∈ R256
        ↓
Linear(256,512)
        ↓
      GELU
        ↓
Linear(512,3072)
        ↓
     S_B(t)
```

State Head 使用与 V0.7-Frozen 相同的 deterministic initialization rule。输出禁止 LayerNorm、L2 normalization、cosine normalization 和 `mu_phi + deltaS` residual。

### 4.2 Context-only control

每个 training stage 的 Market 与 Context 模型必须从同一个完整 initial `state_dict` copy，架构、参数量、State Head、optimizer、scheduler 和 sampler/order 相同。两者必须作为两个**独立 deterministic runs** 执行，不得交错 Market/Context forward，也不得实现任何手工 capture/restore paired-RNG hack。

每个独立 run 开始前都重新设置完全相同的 Python、NumPy、Torch CPU 和 Torch CUDA seed=42，并使用相同 sampler seed、相同 batch order、相同 DataLoader 配置与相同 forward 调用结构。因此在相同 epoch 范围内，dropout RNG 序列自然一致。唯一输入差异是 Context forward 在 normalization 后强制：

- minute 的全部 market-feature channels 置 0；
- daily/weekly 的 market-feature channels 置 0；
- 保留 minute time/context observations；
- 保留 daily/weekly source-bar count、sequence length/progress；
- 保留现有 positional information。

### 4.3 Formal objective

唯一 loss 为：

\[
L_{state}=\frac{1}{N}\sum_t\|S_t-\phi(Y_t)\|_2^2.
\]

实现必须在 FP32 中执行：

```python
per_sample = sum((S.float() - phiY.float()) ** 2, dim=-1)
loss = mean(per_sample)
```

梯度必须贯穿 State Head、Fusion、Minute Market Transformer、Minute Context GRU、Daily GRU 和 Weekly GRU。不得加入任何 auxiliary loss。

## 5. 冻结训练协议

固定配置：

```text
seed                    = 42
deterministic algorithms = true
microbatch              = 64
gradient accumulation   = 2
effective batch         = 128
gradient clip           = 1.0
AMP                     = true
AdamW betas             = (0.9, 0.999)
AdamW eps               = 1e-8
warmup                  = 5%
scheduler               = cosine
scheduler horizon       = 100 epochs
```

AdamW 参数组：

| 参数 | LR | weight decay |
| --- | ---: | ---: |
| Encoder/Fusion | 3e-4 | 0.05 |
| Predictive-State Head | 1e-3 | 1e-4 |

### 5.1 Inner selection

Market 和 Context 分别在 2018–2021 训练完整 100 epochs，每 epoch 只在 2022 Inner Dev 的固定 `phi(Y)` space 计算 State MSE。各自选择 minimum Inner-Dev State MSE；完全相等时选更早 epoch。若 zero-based best epoch 为 `e`，Final budget 为 `N=e+1`。

不得 early-stop。Inner checkpoint 只选择预算，不能用于 Development evaluation。如果 best epoch 为 zero-based 99，则预算保持 `N=100`，禁止自动扩展搜索上限，并记录 `budget_hit_ceiling=true`；否则记录 false。

### 5.2 Final training

Inner selection 完成后：

1. 丢弃 inner model；
2. 重新拟合 2018–2022 的全部 preprocessing；
3. 重新 seed42 初始化；
4. Market/Context 从同一个 initial `state_dict` copy；
5. 分别训练各自固定 N epochs；
6. 保存 Final checkpoint。

Final scheduler 的 horizon 仍按完整 100 epochs 计算；即使 N<100，也只执行 100-epoch schedule 的前 N epochs，不压缩 cosine schedule。2023–2024 不参与 checkpoint selection。

Market 与 Context 的每个独立 run 在每个完整 epoch 后分别原子保存可恢复状态，包括 model、optimizer、scheduler、GradScaler、Python/NumPy/Torch/CUDA RNG、sampler epoch、global step、history 和 preprocessing hash。恢复训练不得改变 batch order 或随机轨迹。

## 6. Final checkpoint contract

Market 与 Context 分开保存，至少包含：

- model weights、zero-based final epoch 和 selected epoch budget；
- 完整 input normalizers 及 SHA-256；
- Y scaler；
- `sigma0`、bandwidths、RFF frequencies/phases、`mu_phi`；
- source-data SHA-256；
- ordered feature schema；
- split definitions；
- protocol version 和 protocol document SHA-256；
- git HEAD 与 dirty status；
- implementation manifest 和 SHA-256；
- `test_consumed=false`。

## 7. Development Gate 1 — Conditional Distribution Information

只在 Final checkpoints 完成后首次计算 2023–2024：

\[
e_B=\|S_B-\phi(Y)\|^2,\qquad
e_C=\|S_C-\phi(Y)\|^2,
\]

\[
S_U=\mu_\phi,\qquad e_U=\|S_U-\phi(Y)\|^2.
\]

Primary 和 sanity effects：

\[
\Delta_{info}=e_C-e_B,
\qquad
\Delta_{uncond}=e_U-e_B.
\]

Gate 1 PASS 要求两个 effect 的预注册 bootstrap 95% CI lower 均严格大于 0。

## 8. Development Gate 2 — Predictive Geometry

State distance 固定为原始 S-space squared Euclidean distance。KNN search 必须返回全体 Train candidates 中的 exact top-50，禁止 FAISS IVF、HNSW 或任何 approximate ANN，也禁止 cosine、PCA、post-hoc normalization、metric learning 或其他 K。允许 GPU FAISS `IndexFlatL2`、PyTorch chunked exact distance 或 CPU brute force。

距离完全相等时，按更小的 Train `anchor_index` 优先；实现必须对 top-K boundary ties 应用该规则，不能依赖 backend 未定义的 `topk` tie order。新增小规模 regression，要求 chunked/GPU 的 `(distance, anchor_index)` 排序结果与 full brute-force exact result 完全一致。

```text
K                  = 50
query              = 2023–2024 Development
candidate          = 2018–2022 Final Train only
causal assertion   = candidate_timestamp < query_timestamp
random seed        = 4244
random sampling    = without replacement
```

分别在 `S_B` 和 `S_C` 中寻找 Train neighbors。每组 neighbor 的预测为其 `phi(Y)` 均值，误差为该均值与 query `phi(Y)` 的 squared Euclidean distance。

\[
\Delta_{geometry}=Error_{ContextKNN}-Error_{MarketKNN},
\]

\[
\Delta_{random}=Error_{Random}-Error_{MarketKNN}.
\]

Gate 2 PASS 要求两个 effect 的预注册 bootstrap 95% CI lower 均严格大于 0。

## 9. Bootstrap 与 Development decision

四个正式 effects 均使用：

```text
block             = inferred trading_day 的 ISO year-week
resamples         = 10000
seed              = 42
estimand          = paired per-sample mean effect
```

每次 bootstrap 抽取 week 后使用：

```text
sum(sampled_block_effect_sums) / sum(sampled_block_sample_counts)
```

禁止对 weekly means 等权平均。

该 bootstrap 只衡量 2023–2024 evaluation samples/weeks 的不确定性，不衡量不同 neural-network initialization seeds 的训练不确定性，也不证明逐年复制。只有以下四项的 CI lower 全部大于 0：

```text
Market > Context information
Market > Unconditional information
Market geometry > Context geometry
Market geometry > Random
```

才输出 `V0.7 DEVELOPMENT_GO`；否则输出 `V0.7 DEVELOPMENT_NO_GO`。无论结果为何都必须停止，不自动调参，不进入 RSSM，不运行 Final Test。

## 10. 2025 Final Test embargo

本协议现在冻结未来 Test rule，但本任务禁止执行。只有 Development GO 后且用户另行明确授权，才允许使用同一套 2018–2022 Final checkpoints 和 preprocessing 对 2025 评估一次；禁止加入 2023–2024 retraining。

未来 Test 的最低成功条件冻结为四个 point estimates 与 Development 同方向：

```text
Delta_info > 0
Delta_unconditional > 0
Delta_geometry > 0
Delta_random > 0
```

Test bootstrap CI 可报告但不作为强制条件。当前以及本任务结束时都必须保持 `test_consumed=false`。

## 11. Artifact 与最终报告

使用独立目录，不覆盖 V0 或 V0.7-Frozen：

```text
artifacts/checkpoints/market_predictive_state_v0_7/
artifacts/evaluation/market_predictive_state_v0_7/
```

最终报告只包含：

```text
V0.7 End-to-End Predictive-State

Inner:
Market selected epoch
Context selected epoch

Final target audit:
sigma0
RFF RMSE/MAE/correlation

Gate 1:
Market/Context/Unconditional errors
Delta_info + CI + PASS/FAIL
Delta_unconditional + CI + PASS/FAIL

Gate 2:
Market/Context/Random kNN errors
Delta_geometry + CI + PASS/FAIL
Delta_random + CI + PASS/FAIL

Overall:
V0.7 DEVELOPMENT_GO / DEVELOPMENT_NO_GO

Test consumed = false
pytest = ...
```

## 12. Implementation traceability checklist

| 协议要求 | 实现门禁 | Regression evidence |
| --- | --- | --- |
| 旧 JEPA 完全禁用 | 独立 model/CLI，不接受旧 checkpoint 参数 | 模型 schema 不含 EMA/predictor；旧 checkpoint loader mock 未调用 |
| Inner preprocessing 仅 2018–2021 | 显式 fit range | 修改 2022 数据不改变 Inner statistics/hash |
| Final preprocessing 仅 2018–2022 | Final 独立 fit | 修改 2023–2024 不改变 Final statistics/hash |
| 2025 禁止 | 无 Test split CLI；数据截止 2024 | Test Dataset constructor mock 调用数为 0 |
| H256 不跨 population end | range-aware anchor builder | anchor/target-end boundary tests |
| Market/Context 唯一差异是输入 | 同一 initial state、batch 和 dropout RNG；market channels mask | bitwise initialization、parameter-count、batch-order、mask tests |
| FP32 squared Euclidean objective | autocast 外显式 float loss | dtype、公式和全分支 gradient tests |
| 固定 100-epoch scheduler | scheduler horizon 与实际 N 解耦 | N<100 时 LR trajectory regression |
| Inner model 不进入 Development | Final 强制重新初始化 | state/hash 与 stage provenance tests |
| Train-only Euclidean KNN | 固定 evaluator 和 causal assertion | distance、K、candidate boundary tests |
| sample-weighted week bootstrap | block sums/counts | 不等长 week regression test |
| exact resume | epoch 原子 checkpoint + 全 RNG | interrupted/resumed trajectory test |
| 最终停止 | Development-only command | GO/NO-GO 均不调用 Test/后续 stage |

## 13. 开发前冻结门禁

0.7.1 已完成科学和实现歧义评审，状态冻结为：

```text
Frozen for implementation
```

正式训练启动前必须满足 `git status --short` 无任何输出；artifacts/logs/cache 必须由 `.gitignore` 排除，禁止从 dirty worktree 启动。随后计算并记录本冻结文件的 SHA-256。实现不得与冻结文档冲突；任何科学协议变更必须提升版本并重新评审。
