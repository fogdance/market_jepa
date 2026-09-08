# Market-JEPA V1.1 固定样本预算训练方案（Scheme A）
## 基于当前代码的设计、迁移规范与 Codex 实现任务书

**状态：REVIEWED / APPROVED FOR CODEX IMPLEMENTATION**
**训练协议版本建议：`fixed_sample_budget_v1`**
**适用范围：V1.1 Full S/M/L/XL + S-LateFusion / S-MinuteOnly / S-NoHistoricalWeekly controls**
**不包含：Evaluation P0 修复、RB-Test 语义修改、模型架构修改。**

---

# 0. Executive Decision

当前 V1.1 正式训练把：

```text
samples_per_epoch = len(train_dataset)
total_steps = steps_per_epoch * max_epochs
warmup_steps = total_steps * warmup_ratio
checkpoint = epoch end only
```

绑定在一起。

在当前约 70 个有效 Train commodities 下，实际已经出现：

```text
optimizer steps / epoch = 191,850
max_epochs              = 100
total optimizer steps   = 19,185,000
warmup_ratio             = 5%
warmup optimizer steps  = 959,250
```

结果是：

```text
商品数量增加
    ↓
eligible anchors 增加
    ↓
epoch 自动变长
    ↓
总训练预算自动变大
    ↓
warmup 自动变长
    ↓
第一次 checkpoint 也自动推迟
```

这不适合当前：

```text
Commodity -> Contract -> Anchor
```

的随机层级采样式自监督预训练。

本方案改为业内成熟的 **Scheme A：固定 optimizer-step / sample exposure budget**：

\[
\boxed{
SamplingDistribution
\quad\perp\quad
TrainingDuration
}
\]

即：

- 商品/合约/anchor 的采样分布继续由 sampler 决定；
- 总训练量由固定 optimizer steps 决定；
- warmup 由固定 optimizer steps 决定；
- checkpoint 按 optimizer steps 保存；
- `len(dataset)` 只描述可采样数据空间，不再决定训练时长。

---

# 1. Industry Grounding

当前方案遵循大规模预训练常见的 step/sample-budget 设计，而不是依赖“完整 dataset 遍历一次”的 epoch 语义。

典型成熟实现具有以下共同点：

- Hugging Face Trainer 支持 `max_steps` 作为训练总时长，并覆盖 `num_train_epochs`；
- warmup 可以按明确的 optimizer steps 定义；
- logging / save 可以按 steps 定义；
- NVIDIA NeMo 大模型预训练使用 `trainer.max_steps`；
- NeMo 明确使用：
  `consumed_samples = global_step * global_batch_size`
  来描述数据消耗。

因此这里不是删除“epoch”这个词本身，而是：

\[
\boxed{
不再让 raw dataset size 决定 optimizer time scale
}
\]

---

# 2. 当前代码审计：问题具体在哪里

Review 基于当前代码 commit：

```text
24c5bbaa4dd5609473d630096575e58aaa55b4e8
```

核心相关文件：

```text
market_jepa_v1_1/training.py
market_jepa_v1_1/formal_training.py
market_jepa_v1_1/config.py
market_jepa_v1_1/sampler.py
market_jepa_v1_1/checkpoint.py
market_jepa_v1_1/wandb_logging.py
configs/v1_1/*.yaml
train_market_jepa_v1_1.py
train_market_jepa_v1_1_control.py
```

---

## 2.1 当前训练时长由 `len(train_dataset)` 隐式决定

当前 `V11Trainer.__init__`：

```python
samples = int(samples_per_epoch or len(train_dataset))
```

正式 runner 又明确传：

```python
samples_per_epoch=len(train_dataset)
```

然后：

```python
steps_per_epoch = ceil(ceil(samples / batch_size) / accumulation)
total_steps = steps_per_epoch * max_epochs
```

因此：

\[
len(dataset)\uparrow
\Rightarrow
total\_steps\uparrow
\]

这正是当前 5 commodities 扩大到约 70 commodities 后训练时间暴涨的主要来源。

---

## 2.2 Warmup 也被 dataset size 隐式拖长

当前：

```python
warmup = int(total_steps * warmup_ratio)
```

现在：

```text
191,850 steps/epoch
× 100 epochs
× 5%
=
959,250 warmup steps
```

也就是说 optimizer 要接近 **96 万次更新**才完成 warmup。

这导致：

> 改变商品数量，实际上同时改变了优化器的时间尺度。

这是不应该的 coupling。

---

## 2.3 Checkpoint 只在 epoch 结束保存

当前：

```python
train = self._run_epoch(...)
...
save_checkpoint(self._state(epoch), last.pt)
```

当前一个 epoch 约 40+ 小时。

因此如果训练在 step 54,800 停止：

```text
没有 mid-epoch checkpoint
→ 54,800 updates 全部丢失
→ 从 step 0 重来
```

这是当前最明显的工程风险之一。

---

## 2.4 Resume 语义也是 epoch-level

当前：

```python
self.start_epoch = state["epoch"] + 1
```

只支持：

```text
完成 epoch 0
→ resume epoch 1
```

不支持：

```text
step 50,000
→ resume step 50,001
```

---

## 2.5 Epoch 对当前 sampler 没有“完整遍历一次”的强语义

当前 sampler：

```text
uniform Commodity
    ↓
uniform eligible Contract
    ↓
uniform Anchor
```

是 replacement-style randomized hierarchical sampling。

相邻 minute anchors 又高度重叠：

```text
anchor t:
[t-511 ... t]

anchor t+1:
[t-510 ... t+1]
```

因此：

```text
1 epoch = len(all eligible anchors)
```

只是人为设置的抽样次数，不代表真正意义上的“每个独立训练样本完整看一次”。

所以 dataset length 不应继续承担 optimizer budget 的角色。

---

# 3. Scheme A 的科学目标

本方案固定：

\[
\boxed{
\text{optimizer updates}
}
\]

由于当前所有 S/M/L/XL 的：

```text
batch_size = 64
gradient_accumulation = 2
```

所以：

\[
EffectiveBatch=128
\]

因此固定 optimizer steps 等价于固定 sample exposure：

\[
SamplesSeen
=
GlobalStep\times128
\]

这正是 Scheme A。

---

# 4. 本次建议冻结的训练预算

## 4.1 Formal V1.1 Fixed-Budget V1

建议第一版冻结：

```yaml
max_optimizer_steps: 250000
effective_batch_size: 128
```

因此：

\[
250000\times128
=
32,000,000
\]

即：

```text
formal target sample exposures = 32,000,000
```

---

## 4.2 为什么先选择 250K，而不是继续 100 epochs

这个数字不是“行业魔法常数”。

它是当前项目的第一版 fixed-budget protocol，理由是：

1. 与 commodity count 完全解耦；
2. 32M exposure 已经略高于当前一个超长 epoch 的约 24.56M exposure；
3. 对高度重叠的 minute anchors，不需要机械重复 100 次所谓 dataset traversal；
4. 当前 M 实测约 0.8 sec / optimizer step：
   250K 大约 55 小时，而不是约 177 天；
5. 足够形成一个完整的 warmup + cosine schedule；
6. 可以为 S/M/L/XL 提供完全相同的数据 exposure budget；
7. 后续如果研究更长预算，必须另立：
   `fixed_sample_budget_v2_500k`
   等独立 campaign，而不是根据结果临时改变当前 formal endpoint。

重要：

\[
\boxed{
250K\ 是实验协议，不是“训练到完全收敛”的声明
}
\]

它回答的是：

> 相同数据 exposure budget 下，不同模型容量的 predictive ability 如何变化。

---

# 5. 新的正式 Training Config

建议正式 YAML 改成：

```yaml
training:
  protocol_version: fixed_sample_budget_v1

  seed: 42

  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.05
  betas: [0.9, 0.999]
  eps: 1.0e-8

  batch_size: 64
  gradient_accumulation: 2

  budget_mode: fixed_optimizer_steps
  max_optimizer_steps: 250000
  warmup_optimizer_steps: 12500

  checkpoint_every_optimizer_steps: 5000
  progress_every_optimizer_steps: 200

  scheduler: cosine
  gradient_clip_norm: 1.0
  ema_tau: 0.996

  amp: true
  amp_dtype: bfloat16

  lambda_var: 0.0
  lambda_cov: 0.0
  variance_floor: 1.0

  num_workers: 8
  gradient_checkpointing: <profile-dependent>

  checkpoint_dir: ...
```

Formal config 中：

```text
max_epochs
warmup_ratio
```

不再作为训练控制参数。

最好从正式 V1.1 step-budget config schema 中删除，避免存在两个互相冲突的 source of truth。

---

# 6. Warmup 与 Scheduler

当前 optimizer / scheduler shape 不需要重新发明。

继续保持：

```text
AdamW
base LR = 3e-4
cosine
5% warmup 的原意
```

但把 warmup 从：

```text
5% × dataset-dependent total steps
```

变成明确：

```text
12,500 optimizer steps
```

即：

\[
12500/250000=5\%
\]

这样商品数量变化不会改变：

- peak LR 到达时间；
- cosine schedule 长度；
- optimizer 的整体时间尺度。

---

## 6.1 Scheduler hard invariants

Codex 必须增加测试验证：

```text
total scheduler budget = 250000 successful optimizer updates
warmup budget          = 12500
```

并确保：

- scheduler 只在成功 optimizer update 后 advance；
- EMA 也只在成功 optimizer update 后 advance；
- `global_step` 与 scheduler step 一一对应；
- formal BF16 遇到 non-finite gradient 直接 FAIL，不允许 silently consume budget。

不要在本任务中改变现有 cosine 数学公式，只改变：

```text
total_steps source
warmup_steps source
```

以降低实验变量数量。

---

# 7. Sampler：保留核心设计，不要改 sampling distribution

当前：

\[
Commodity\rightarrow Contract\rightarrow Anchor
\]

是正确的。

不要改成：

```text
uniform over all raw anchors
```

否则数据量大的 commodity 会支配训练。

Scheme A 只改变：

```text
训练多久
```

不改变：

```text
训练看什么
```

---

# 8. 新的 sampler / training segment 设计

## 8.1 目标

训练 duration 与 dataset length 解耦，同时保证：

- deterministic；
- 可 resume；
- S/M/L/XL 使用同一 sample stream；
- controls 使用同一 sample stream。

---

## 8.2 Sampling cycle

推荐把：

```text
checkpoint_every_optimizer_steps = 5000
```

同时作为一个 sampling cycle。

每个 cycle：

\[
5000\times128
=
640,000
\]

sample selections。

对于 250K：

```text
50 sampling cycles
```

---

## 8.3 Deterministic cycle seeding

当前 sampler 已经使用：

```python
SeedSequence([seed, epoch])
```

新方案改语义为：

```python
SeedSequence([seed, sampling_cycle])
```

不要再把它叫 scientific epoch。

每个 cycle：

```text
cycle 0 → seed [42,0]
cycle 1 → seed [42,1]
...
cycle 49 → seed [42,49]
```

因此只要：

- dataset hierarchy 一致；
- seed 一致；
- cycle 一致；

S/M/L/XL 和 matched controls 会收到**完全相同的抽样序列**。

这是比当前“same epoch count”更强的科学匹配。

---

# 9. Exact Sample Stream Parity

正式 campaign 应增加一个：

```text
sampling_population_sha256
```

它至少 hash：

```text
ordered effective train commodities
for each commodity:
    ordered eligible contract episode keys
    anchor count per episode
data manifest sha256
history eligibility semantics
```

所有：

```text
S
M
L
XL
LateFusion-S
MinuteOnly-S
NoHistoricalWeekly-S
```

必须具有相同：

```text
sampling_population_sha256
sampler_seed
max_optimizer_steps
effective_batch_size
```

这样才能声称：

\[
\boxed{
same\ data\ exposure
}
\]

---

# 10. Checkpoint 设计

## 10.1 Periodic recovery checkpoint

每：

```text
5000 optimizer steps
```

atomic overwrite：

```text
checkpoints/last.pt
```

例如：

```text
5k
10k
15k
...
250k
```

当前 M 速度下，最大未保存训练量约为 1 小时左右，而不是 40+ 小时。

---

## 10.2 Official endpoint

仍保持：

```text
checkpoint_selection = fixed_budget_final
```

但官方 endpoint 从：

```text
epoch 99 last.pt
```

改为：

```text
global_step = 250000
samples_seen = 32000000
last.pt
```

Intermediate checkpoint 永远不能用于：

- checkpoint selection；
- best-loss selection；
- early stopping。

---

## 10.3 Checkpoint state 新增字段

新 checkpoint 至少加入：

```text
training_protocol_version
budget_mode

global_step
max_optimizer_steps

effective_batch_size
samples_seen
target_samples_seen

warmup_optimizer_steps
checkpoint_every_optimizer_steps

sampling_cycle
sampling_cycle_sample_offset
sampling_population_sha256

model
EMA
optimizer
scheduler
AMP scaler if applicable
RNG state

data manifest hash
implementation hash
config snapshot/hash
shared IMC scaler

W&B run id

skipped_optimizer_steps
gradient_connectivity
```

Formal BF16 正常情况下必须满足：

\[
samples\_seen
=
global\_step\times128
\]

---

# 11. Resume 设计

## 11.1 正常周期 checkpoint resume

例如：

```text
last.pt:
global_step = 50000
samples_seen = 6,400,000
```

下次：

```text
resume
→ directly continue toward 50001
```

而不是：

```text
start_epoch = old_epoch + 1
```

---

## 11.2 Hard crash

如果：

```text
last checkpoint = 50,000
process dies at 54,800
```

则 resume：

```text
50,000
```

最多损失 4,800 updates。

这是正常 periodic-checkpoint recovery semantics。

---

## 11.3 Graceful Ctrl+C（推荐实现）

推荐 Codex 同时实现：

```text
SIGINT / SIGTERM
→ set stop_requested
→ 完成当前 gradient accumulation / optimizer update
→ 保存 last.pt
→ clean exit
```

如果实现 exact mid-cycle checkpoint：

checkpoint 还要保存：

```text
sampling_cycle
sampling_cycle_sample_offset
```

Resume 时：

1. 重新初始化该 cycle 的 deterministic RNG；
2. 跳过已 committed sample offset；
3. 从下一个 sample 继续。

重要：

不要使用 DataLoader 当前 prefetch position 作为 source of truth。

source of truth 必须是：

```text
committed samples_seen
sampling_cycle
sampling_cycle_sample_offset
```

这样 num_workers/prefetch 不会破坏 resume reproducibility。

如果 Codex 判断 graceful exact resume 在当前 DataLoader 架构下会显著增加复杂度，可以：

- P0 先实现 5K periodic checkpoint；
- P1 再实现 graceful exact resume。

但不能继续维持 epoch-end-only checkpoint。

---

# 12. Trainer 主循环重构

当前：

```python
for epoch in range(...):
    sampler.set_epoch(epoch)
    run_epoch()
    save_checkpoint()
```

改成概念上：

```python
while global_step < max_optimizer_steps:
    cycle = global_step // checkpoint_every_optimizer_steps

    steps_remaining = max_optimizer_steps - global_step
    steps_this_cycle = min(
        checkpoint_every_optimizer_steps,
        steps_remaining,
    )

    samples_this_cycle = (
        steps_this_cycle
        * batch_size
        * gradient_accumulation
    )

    sampler.configure(
        cycle=cycle,
        num_samples=samples_this_cycle,
        start_offset=resume_offset,
    )

    run_training_segment(
        expected_optimizer_steps=steps_this_cycle
    )

    save_atomic(last.pt)
    append interval metrics
```

不要再根据：

```python
len(train_dataset)
```

计算 training duration。

---

# 13. Progress 语义

现在日志：

```text
epoch=0 step=54600/191850 28.5%
```

改成：

```text
global_step=54600/250000 21.8%
samples_seen=6988800/32000000
loss=...
h16=...
h64=...
h256=...
lr=...
elapsed=...
ETA=...
```

这才是真正有跨 dataset / model size 比较意义的进度。

---

# 14. Local Metrics

W&B 继续是 visualization。

本地文件仍然是 source of truth。

建议新增：

```text
step_metrics.jsonl
```

每 50 optimizer steps记录：

```text
global_step
samples_seen
loss
h16_loss
h64_loss
h256_loss
learning_rate
grad_norm
step_seconds
samples_per_sec
allocated_vram
reserved_vram
skipped_optimizer_steps
```

以及：

```text
interval_metrics.csv
```

每 5000 steps 一行：

```text
step_start
step_end
samples_start
samples_end
mean_loss
mean_h16
mean_h64
mean_h256
learning_rate_end
elapsed_seconds
samples_per_sec
peak_vram
```

不再把：

```text
epoch_metrics.csv
```

作为新的正式训练语义。

旧文件可以保留读取兼容，但新 campaign 不应再把它当 source of truth。

---

# 15. W&B 更新

Project 保持：

```text
market-jepa
```

建议新 Group：

```text
v1.1-formal-fixed-budget-v1
```

Run name：

```text
v1.1-S-8.26M-250Kstep-seed42
v1.1-M-22.09M-250Kstep-seed42
v1.1-L-45.52M-250Kstep-seed42
v1.1-XL-102.29M-250Kstep-seed42
```

W&B metadata 增加：

```text
training_protocol_version
budget_mode
max_optimizer_steps
target_samples_seen
effective_batch_size
warmup_optimizer_steps
checkpoint_every_optimizer_steps
sampling_population_sha256
```

W&B 横轴继续使用：

```text
global_step
```

并同步记录：

```text
samples_seen
```

不要再把 epoch 当主要 x-axis。

---

# 16. Formal Protocol / Summary 更新

当前 `protocol.json` 中：

```text
samples_per_epoch
epochs
official_endpoint = epoch ...
```

应改为：

```json
{
  "training_protocol_version": "fixed_sample_budget_v1",
  "budget_mode": "fixed_optimizer_steps",

  "max_optimizer_steps": 250000,
  "effective_batch_size": 128,
  "target_samples_seen": 32000000,

  "warmup_optimizer_steps": 12500,
  "checkpoint_every_optimizer_steps": 5000,

  "eligible_anchor_count": "...",
  "dataset_equivalent_exposure_ratio": "...",

  "sampler": "Commodity -> Eligible Contract -> Anchor",
  "sampling_population_sha256": "...",

  "checkpoint_policy": "fixed_budget_final",
  "official_endpoint": "global_step=250000"
}
```

`dataset_equivalent_exposure_ratio` 只是 diagnostic：

\[
target\_samples\_seen / eligible\_anchor\_count
\]

绝不能重新用于决定 training duration。

---

# 17. S/M/L/XL 的公平性

Scheme A 的核心实验控制：

```text
same commodities
same eligible contracts
same sampler
same sample sequence
same effective batch
same optimizer updates
same sample exposures
same LR schedule
same EMA schedule
same objective
same checkpoint endpoint
```

只改变：

```text
model capacity
```

因此：

```text
S  250K updates
M  250K updates
L  250K updates
XL 250K updates
```

全部看到：

```text
32M sample exposures
```

---

# 18. 这不是 Fixed Compute

必须明确：

Scheme A 固定的是：

```text
data/sample exposure
```

不是：

```text
FLOPs
```

因此 XL 会比 S 消耗更多计算资源。

本实验回答：

> 相同训练样本预算下，提高模型容量是否提升 predictive representation？

它不回答：

> 相同 FLOPs 下哪个模型最优？

后者属于单独 Scheme C / fixed-compute experiment。

不要混淆两个 claim。

---

# 19. 商品数量增加后的含义

固定 32M samples 后：

如果 effective Train commodities = 70：

\[
32M / 70
\approx457,143
\]

平均每个 commodity 的 expected selections。

增加 commodity 后：

```text
每个 commodity 平均 exposure 会下降
```

这是 Scheme A **有意的设计**，不是 bug。

因为 Scheme A 的问题就是：

> 相同总训练预算下，更多数据多样性是否有价值？

如果未来要研究：

> 每个 commodity 保持同样 exposure 时增加商品是否有价值？

那是另一种：

```text
fixed exposure per commodity
```

实验，不属于当前 campaign。

---

# 20. Controls 必须继承同一预算

以下 controls：

```text
S-LateFusion
S-MinuteOnly
S-NoHistoricalWeekly
```

必须使用：

```text
250000 optimizer steps
32M samples
same sample stream
same scheduler
same batch
same seed
same checkpoint cadence
```

Control checkpoint/protocol 必须记录：

```text
max_optimizer_steps
target_samples_seen
sampling_population_sha256
```

Evaluation comparison 必须 hard-gate：

```text
Full budget == Control budget
Full sample population hash == Control hash
Full effective batch == Control effective batch
```

否则不能称 matched control。

---

# 21. 当前正在训练的 M 如何处理

当前 M 使用旧 epoch-based scheduler：

```text
total_steps   = 19,185,000
warmup_steps  = 959,250
```

新的 Scheme A：

```text
total_steps   = 250,000
warmup_steps  = 12,500
```

两条 LR trajectory 完全不同。

因此：

\[
\boxed{
旧 M checkpoint 不能直接 resume 成 Scheme A formal run
}
\]

正确处理：

1. 当前 epoch0 跑完；
2. 确认旧 `last.pt` 保存；
3. 停止；
4. 将其标记为：
   `LEGACY_EPOCH_BUDGET_DIAGNOSTIC`;
5. 用于：
   - latent health；
   - A1 diagnostic；
   - loss behavior 分析；
6. **不得**作为 Scheme A 的初始化 checkpoint；
7. Scheme A formal campaign 必须 seed42 从 scratch 开始。

否则：

```text
前 191,850 steps 用旧 19.185M schedule
后面突然切 250K schedule
```

无法科学解释。

---

# 22. Backward Compatibility

新代码仍应能：

```text
load old checkpoint for evaluation
```

但必须拒绝：

```text
resume legacy epoch-budget checkpoint
into fixed_sample_budget_v1
```

建议错误：

```text
Cannot resume legacy epoch-budget checkpoint into
training_protocol_version=fixed_sample_budget_v1.
Legacy checkpoints remain evaluation-only.
```

这样不会破坏我们现有 epoch0 checkpoint 的诊断价值。

---

# 23. Config Validation

`validate_v11_config()` / `validate_formal_training_config()` 必须新增 hard gates：

```text
protocol_version == fixed_sample_budget_v1
budget_mode == fixed_optimizer_steps

max_optimizer_steps == 250000
warmup_optimizer_steps == 12500
checkpoint_every_optimizer_steps == 5000

batch_size == 64
gradient_accumulation == 2
effective_batch_size == 128
```

并冻结原有：

```text
AdamW
3e-4
wd 0.05
betas
eps
cosine
clip 1
EMA .996
BF16
objective
```

新 formal config 中不要同时保留：

```text
max_epochs
warmup_ratio
```

作为可生效参数。

---

# 24. Checkpoint Completion Gate

旧 completion：

```python
len(history) == max_epochs
state["epoch"] == final_epoch
```

必须改成：

```python
state["global_step"] == max_optimizer_steps
state["samples_seen"] == target_samples_seen
```

以及：

```text
finite metrics
gradient connectivity PASS
data hash match
implementation hash match
scaler match
sampling population hash match
checkpoint roundtrip PASS
```

最终状态：

```text
V1_1_FIXED_SAMPLE_BUDGET_TRAINING_PASS
```

---

# 25. CLI 语义

现有命令保持尽量不变：

```bash
python train_market_jepa_v1_1.py \
    --config configs/v1_1/market_jepa_v1_1_m.yaml \
    --output artifacts/training/v1_1_fixed_budget_m
```

Resume：

```bash
python train_market_jepa_v1_1.py \
    --config configs/v1_1/market_jepa_v1_1_m.yaml \
    --output artifacts/training/v1_1_fixed_budget_m \
    --resume artifacts/training/v1_1_fixed_budget_m/checkpoints/last.pt
```

不要要求用户手工提供：

```text
start_step
samples_seen
cycle id
```

全部从 checkpoint 恢复。

---

# 26. Tests — Codex 必须实现

## 26.1 Budget independent of dataset length

构建两个 fixture：

```text
small dataset
large dataset
```

配置相同：

```text
max_optimizer_steps = 20
effective batch = same
```

要求两个 trainer：

```text
final global_step == 20
same scheduler total steps
same warmup steps
```

dataset length 不得改变训练 duration。

---

## 26.2 Commodity count does not change max steps

5 commodities 与 20 commodities：

```text
same fixed budget
```

最后：

```text
same optimizer updates
same target samples_seen
```

仅 sampler distribution 不同。

---

## 26.3 Sample-stream parity across model sizes

同一 dataset + seed：

```text
S
M
```

前 N 个 sampled global indices 必须完全一致。

Control 也必须一致。

---

## 26.4 Periodic checkpoint

用 tiny debug：

```text
max_steps = 12
checkpoint_every = 5
```

验证：

```text
step5  -> last saved
step10 -> last saved
step12 -> final saved
```

最终：

```text
global_step=12
```

---

## 26.5 Resume

至少测试：

```text
uninterrupted 20 steps
```

vs：

```text
run 10
save
resume
run to 20
```

最终要求：

- global_step same；
- samples_seen same；
- optimizer state same；
- scheduler state same；
- EMA same；
- model parameters same within deterministic tolerance；
- next sampled indices same。

---

## 26.6 Legacy resume rejection

旧 checkpoint：

```text
epoch-budget
```

必须：

```text
model load/evaluation = PASS
formal fixed-budget resume = REJECT
```

---

## 26.7 Scheduler golden tests

验证：

```text
max steps = 250000
warmup = 12500
```

以及 tiny equivalent config：

- warmup start；
- warmup end；
- mid cosine；
- final step。

不要出现 dataset-size-dependent LR。

---

## 26.8 Samples seen

正式 BF16 path：

\[
samples\_seen
=
global\_step
\times
batch\_size
\times
gradient\_accumulation
\]

必须成立。

---

## 26.9 Formal controls

验证：

```text
Full S
LateFusion S
MinuteOnly S
NoHistoricalWeekly S
```

全部 budget/protocol parity。

---

## 26.10 Existing tests

本任务完成后必须：

```text
new tests PASS
existing V1.1 tests PASS
full repository tests PASS
git diff --check PASS
```

不接受只跑新增 20 个 tests。

---

# 27. Files Expected to Change

主要：

```text
market_jepa_v1_1/config.py
market_jepa_v1_1/training.py
market_jepa_v1_1/sampler.py
market_jepa_v1_1/formal_training.py
market_jepa_v1_1/checkpoint.py
market_jepa_v1_1/wandb_logging.py

configs/v1_1/market_jepa_v1_1.yaml
configs/v1_1/market_jepa_v1_1_s.yaml
configs/v1_1/market_jepa_v1_1_m.yaml
configs/v1_1/market_jepa_v1_1_l.yaml
configs/v1_1/market_jepa_v1_1_xl.yaml

train_market_jepa_v1_1_control.py

tests/test_market_jepa_v1_1.py
tests/test_market_jepa_v1_1_capacity.py
tests/test_market_jepa_v1_1_wandb.py
new step-budget tests as appropriate

docs/MARKET_JEPA_V1_1_FIXED_SAMPLE_BUDGET_TRAINING.md
```

如果 evaluation comparison 需要 budget provenance gate，可最小修改：

```text
market_jepa_v1_1/evaluation/comparison.py
```

但不要在本任务顺带重写 evaluation semantics。

---

# 28. Explicit Non-Goals

本任务不要修改：

- V1.1 architecture；
- IMC；
- Contract Episode semantics；
- same-delivery-month lineage；
- Daily/Weekly causal partial construction；
- Train commodity filtering；
- shared scaler；
- JEPA target；
- H16/H64/H256；
- objective；
- EMA tau；
- Price/OI/Volume semantics；
- held-out RB；
- formal evaluation outcome definitions；
- LateFusion fairness P0 等其它 evaluation review 项目（另一个任务）。

只处理：

\[
\boxed{
TrainingBudget
+
StepScheduler
+
StepCheckpoint
+
StepResume
+
ReproducibleSamplingBudget
}
\]

---

# 29. Migration Plan

## Phase 0 — 当前运行

当前生产路径的 M：

```text
继续 epoch0
→ 保存旧 last.pt
→ 停止
```

不要在正在运行的 production working tree 改代码。

---

## Phase 1 — 副本代码实现 Scheme A

Codex 在副本实现。

跑完整 test suite。

---

## Phase 2 — Review

人工 review：

- config；
- scheduler；
- sampler；
- checkpoint；
- resume；
- formal protocol；
- W&B；
- control budget parity。

---

## Phase 3 — Merge 到 production

只有当前 old M 已停止后 merge。

---

## Phase 4 — Fixed-budget campaign

建议正式顺序：

```text
S 250K
→ Train-domain diagnostics
→ S controls
→ M 250K
→ L 250K
→ XL 250K
```

RB-Test 仍按 Evaluation Protocol 最后一次性打开。

---

# 30. Codex Implementation Acceptance Report

Codex 完成后必须报告：

```text
1. files changed
2. exact config diff
3. old vs new training-budget semantics
4. max_optimizer_steps
5. target_samples_seen
6. warmup_optimizer_steps
7. checkpoint cadence
8. resume semantics
9. sampler determinism mechanism
10. sampling_population_sha256 implementation
11. old checkpoint compatibility
12. control parity
13. W&B changes
14. tests executed
15. full repository test result
16. git diff --check
17. exact start/resume commands
18. any remaining assumptions/blockers
```

并且：

```text
DO NOT start formal training automatically.
DO NOT run RB-Test.
```

---

# 31. Reviewer Re-check

本方案在交付前重新检查了以下潜在问题。

## 31.1 是否把“fixed samples”误写成 fixed compute？

没有。

明确：

```text
same samples / steps
different FLOPs
```

因此适用于 Scheme A，不冒充 compute-optimal scaling。

---

## 31.2 商品增加后每商品 exposure 是否下降？

会。

这是 Scheme A 的定义，不是错误。

文档已经明确区分未来的：

```text
fixed exposure per commodity
```

实验。

---

## 31.3 是否破坏现有 balanced sampler？

没有。

保留：

```text
Commodity -> Contract -> Anchor
```

只把 `epoch` 改为固定 sampling cycle。

---

## 31.4 是否仍会让 dataset size 改变 warmup？

不会。

`warmup_optimizer_steps=12500` 显式冻结。

---

## 31.5 是否还会 40 小时没有 checkpoint？

不会。

正常 recovery checkpoint 每 5K updates。

---

## 31.6 Resume 是否仍依赖 epoch？

新协议不依赖。

以：

```text
global_step
samples_seen
sampling cycle/offset
```

作为 source of truth。

---

## 31.7 S/M/L/XL 是否真正 same sample budget？

是。

并进一步要求：

```text
same sampler seed
same population hash
same sample stream
```

比原 epoch-based 方案更强。

---

## 31.8 当前旧 M checkpoint 能否无损切换？

不能，也不应该。

Scheduler history 已经不同。

旧 checkpoint 只用于 diagnostic。

---

## 31.9 250K 是否被误称为“收敛要求”？

没有。

它被定义为 formal fixed-budget endpoint。

后续更长预算必须另立 campaign。

---

## 31.10 是否扩大了本任务范围？

没有。

明确禁止修改 architecture / evaluation / RB 等非训练预算语义。

---

# 32. Final Review Verdict

当前方案解决了现有训练协议最核心的 coupling：

旧方案：

\[
DatasetSize
\rightarrow
EpochLength
\rightarrow
TotalSteps
\rightarrow
Warmup
\rightarrow
CheckpointTime
\]

新 Scheme A：

\[
Dataset
\rightarrow
SamplingDistribution
\]

与：

\[
FixedStepBudget
\rightarrow
Scheduler
\rightarrow
Checkpoint
\]

分离。

最终训练科学语义变成：

\[
\boxed{
70\ commodities
+
balanced\ hierarchical\ sampler
+
250K\ optimizer\ updates
+
32M\ sample\ exposures
}
\]

而不是：

\[
\boxed{
70\ commodities
\Rightarrow
191850\ steps/epoch
\Rightarrow
100\ epochs
\Rightarrow
19.185M\ updates
}
\]

**Reviewer Verdict：**

```text
APPROVED FOR CODEX IMPLEMENTATION
```

在 Codex 完成并通过完整测试与第二轮代码 review 以前：

```text
NOT APPROVED FOR FORMAL TRAINING
```
