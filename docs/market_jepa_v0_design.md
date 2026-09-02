# Market-JEPA V0 设计文档

**文档版本：0.6.1**
**状态：Frozen — Approved for V0 implementation**
**最后更新：2026-09-02**

## 修订记录

| 文档版本 | 日期 | 状态 | 主要变化 |
| --- | --- | --- | --- |
| 0.1.0 | 2026-09-02 | 已替代 | 初版：三分支 Transformer、future market target、基础 shuffled control。 |
| 0.2.0 | 2026-09-02 | 已替代 | future target 改为纯 EMA minute；加入 persistence/train-mean/block-shuffle；公平 raw window baseline；删除 metadata；冻结日/周聚合 contract；validation/test 只做尾部 purge；日/周改用 GRU；VICReg 默认关闭；加入预注册 GO/NO-GO。 |
| 0.3.0 | 2026-09-02 | 已替代 | minute CSV 成为唯一行情源；Daily/Weekly context 改为 completed history + anchor 时点实时 partial bar；冻结 partial 聚合、特征、normalization 与 source-index provenance contract；补充 8 项 partial snapshot 防泄漏测试。 |
| 0.4.0 | 2026-09-02 | 已替代 | 增加 causal 时间 observation 与 source bar count；预注册实际 split 日期和 outcome 公式；增加 partial/completed 分布审计、训练前异常 preflight、分 horizon EMA target diagnostics；清除遗留项目引用。 |
| 0.5.0 | 2026-09-02 | 已替代 | Future/Persistence EMA target 改为 market-only；时间/count 限定为 current-context-only；weekday 改为 5 日周期；模型删除 raw delta；公平 baseline 获得相同 observation；冻结完整训练与 checkpoint-selection protocol。 |
| 0.6.0 | 2026-09-02 | 已替代 | Minute market 与时间 observation 改为两个独立 encoder；online/EMA Minute Market Transformer 输入语义完全同构；冻结最终 V0 架构。 |
| 0.6.1 | 2026-09-02 | Frozen — Approved for V0 implementation | 修正 trading-day block bootstrap 的逐样本 estimand 和 calendar-month shuffle；checkpoint 增加完整 RNG state；补充端到端集成测试；行情标识改为品种与来源序列两级 metadata。 |

## 1. 目标与边界

本实验只验证一个问题：历史市场状态的可学习 latent 是否包含可泛化的未来市场分布信息。V0 不实现交易动作、持仓、reward、PnL、RL、新闻、LLM、order book 或聚类策略。

成功标准不是训练 loss 下降，而是完全 OOS 数据上的以下证据：

1. 真实未来 minute latent 的预测误差优于 persistence、train-mean 与 block-shuffled controls；
2. train-only latent kNN 的未来分布比随机历史邻居更一致；
3. frozen latent 的线性 probe 优于 raw engineered feature baseline；
4. 上述现象跨 validation/test 时间段稳定。

## 2. 已观察到的仓库与数据事实

- 仓库当前没有既有 Dataset 或交易日历。V0 明确以 minute CSV 为唯一权威行情源；Daily/Weekly 是按 anchor 实时生成的 causal snapshot，不依赖外部最终日/周线文件。
- 输入文件为 `8Y_DCE_JM2601_1m.csv`，列为 `Date, Open, High, Low, Close, Volume, OpenInterest`。
- 输出 metadata 固定使用 `symbol=JM`、`series_id=8Y_DCE_JM2601`，只标识品种和来源文件序列，不声称它是真实单合约 JM2601。
- 文件有 655,099 条数据，范围为 2018-01-02 09:01:00 至 2025-12-02 15:00:00；时间严格递增，六个数值字段没有缺失值。
- 数据同时包含 09:00--15:00 日盘和 21:00--23:00 夜盘，但没有显式 `trading_day`、上市日、到期日或交易所日历字段。

V0 不提供任何 contract metadata。文件第一日只能定义 `days_since_file_start`，不能证明是真实 `days_since_listing`；它还会成为日历年份/市场 regime 的强 proxy。`days_to_expiry` 同样无法由现有数据可靠得到。

## 3. 数据语义

### 3.1 Trading day

不得用 `timestamp.date()` 直接聚合。当前文件缺少显式交易日，采用下列可复现薄适配规则：

- 18:00 之前的 bar 属于其自然日期对应的日盘交易日；
- 18:00 及之后的夜盘 bar 属于数据中其后第一个实际出现的日盘日期；
- 文件尾部若存在找不到后续日盘的夜盘，则视为未确认交易日并排除。

该规则由 minute CSV 中实际存在的日盘日期驱动，因此周五夜盘、长周末和节假日前夜会映射到下一个实际交易日，而不是简单加一个自然日。V0 不从外部数据覆盖此映射；生产研究前必须把推断结果与权威交易所日历逐日核对。

周线按 trading day 所属 ISO year-week 分组，而不是 timestamp 的自然周分组。周五夜盘若被 trading-day 规则映射到下周一，则从该夜盘第一根开始属于下一 trading week。

### 3.2 Anchor-time Daily / Weekly snapshot contract

对任意 anchor `t`，三个 timeframe 的所有源 minute 必须满足 `source_timestamp <= t`，等价地满足 `source_index <= anchor_index`。

Daily context 由两部分组成：

1. `D(t)` 之前所有 completed Daily bars；
2. `D(t)` 的一根实时 partial Daily(t)。

partial Daily(t) 只从 `trading_day == D(t) AND timestamp <= t` 聚合：

- `Open` = 当前 trading day 第一根已出现 minute `Open`
- `High` = 截至 anchor 的 `max(minute High)`
- `Low` = 截至 anchor 的 `min(minute Low)`
- `Close` = anchor 时最后一根可见 minute `Close`
- `Volume` = 截至 anchor 的 `sum(minute Volume)`
- `OpenInterest` = anchor 时最后一根可见 minute `OpenInterest`

夜盘第一根出现时即创建该 trading day 的 partial Daily。夜盘结束后它保留夜盘累计状态；下一自然日日盘开始时继续更新**同一个 trading-day Daily token**，不能新建 token。只有进入下一个 trading day 后，前一个 partial 才成为 completed Daily。

Weekly context 同样由两部分组成：

1. 当前 ISO trading week 之前所有 completed Weekly bars；
2. 当前 trading week 的一根实时 partial Weekly(t)。

partial Weekly(t) 等价于“本周此前 completed Daily + 当前 partial Daily(t)”的聚合，固定规则为：

- `Open` = 本 trading week 第一根已出现 Daily/partial Daily `Open`
- `High` = 截至 anchor 的最大 `High`
- `Low` = 截至 anchor 的最小 `Low`
- `Close` = 当前 partial Daily(t) `Close`
- `Volume` = 本周截至 anchor 的累计 `Volume`
- `OpenInterest` = 当前 partial Daily(t) `OpenInterest`

实现允许直接在同一 trading-week 的可见 minute 上做等价的累计聚合，但测试必须证明结果与“completed Daily + partial Daily”完全一致。Daily/Weekly 的每个 token 都保留 `source_min_index` 和 `source_max_index`；当前两个 partial token 的 `source_max_index` 必须等于 anchor index，所有历史 token 的 `source_max_index` 必须更小。

为避免全量数据预聚合造成隐式泄漏，completed bar 即使可离线缓存，也只能按 key 截取严格早于当前 trading day/week 的记录；当前 key 的最终 completed bar绝不能进入 snapshot。

completed Daily/Weekly 不采用另一套聚合逻辑：它就是该 trading day/week 最后一根 minute anchor 对应的 partial snapshot，之后冻结。实现按 minute 原始顺序在 `trading_day` 和 ISO trading-week group 内维护 cumulative first/max/min/last/sum，并为每一行生成当时的 partial raw snapshot；取样时再 concat“严格更早 key 的 completed prefix + anchor 行 partial”。该算法避免每个 anchor 重扫全部 minute，同时保持逐行 source provenance。任何缓存都必须由上述逐行 causal 结果生成。

### 3.3 Market features 与 current-context-only observation

minute、daily、weekly 各自从 OHLCV/OI 生成下列共享的紧凑特征：

- `log_open` / `log_high` / `log_low` / `log_close`（保留价格基础信息并减小尺度跨度）
- `close_log_return`
- `open_to_prev_close`
- `high_to_prev_close`
- `low_to_prev_close`
- `normalized_range`
- `log1p_volume`
- `volume_log_change`
- `log1p_open_interest`（与 `log1p_volume` 一起保留成交量/OI 基础信息）
- `open_interest_log_change`
- `realized_volatility`（只含当前及过去 return 的 rolling RMS）

以上字段定义为 **market features**，可进入 current context 和 future/persistence EMA target。除此之外，current Market Encoder 可见以下 **context-only observation**。

Minute current-context token 的时间 observation 固定定义为：

- `time_of_day_sin = sin(2π * minute_of_day / 1440)`
- `time_of_day_cos = cos(2π * minute_of_day / 1440)`，其中 `minute_of_day = timestamp.hour * 60 + timestamp.minute`
- `day_of_week_sin = sin(2π * trading_day_weekday / 5)`
- `day_of_week_cos = cos(2π * trading_day_weekday / 5)`，其中 Monday=0、...、Friday=4，夜盘使用推断后的 trading day weekday
- `log1p_delta_minutes = log(1 + delta_minutes)`，其中 `delta_minutes = (timestamp_i - timestamp_{i-1}) / 1 minute`；文件第一根置 0

`delta_minutes` 原值只保留在 audit/provenance 中，不进入模型。模型仅接收其 log1p 值，使 session break、隔夜、周末/节假日不被 Transformer token position 隐藏，同时避免 raw gap 的重尾尺度。模型不接收绝对日期、年份或 `days_since_file_start`。

每个 Daily/Weekly current-context token 额外包含 context-only `source_bar_count`，定义为该 token 实际聚合的 minute 行数。completed token 保存完整周期 count，current partial token 保存截至 anchor 的 count；该值和 partial OHLCV/OI 一样每分钟 causal 更新，使模型能区分周期刚开始和接近完成。`source_bar_count` 随各 timeframe 的其余数值一起使用 train-only mean/std 标准化。

Feature schema 必须分别保存有序的 `market_feature_names` 和 `context_observation_names`，不能把两类字段合成一个没有语义边界的数组。Future target Dataset 不得返回 future time/calendar/gap observation；target forward API 只接受 market feature tensor，从接口上禁止误传。

所有 shift/rolling 只向后看。无有效历史的首项填零；价格分母使用 epsilon 防御。Daily/Weekly 的当前 partial token 使用前一 completed token 作为 `previous close/volume/OI` 参照；其 rolling realized volatility 由此前 completed returns 加上“当前 partial close 相对前一 completed close”的当前项计算。随着每根 minute 更新，partial token 的 raw values 和 derived features同步更新。

三个 timeframe 分别拟合 normalization bundle；每个 bundle 内又分别保存 market features 与 context-only observation 的 mean/std。Future/persistence target 只使用 `minute.market` statistics，不存在 future context-observation statistics。fit population 固定如下，避免随 `anchor_stride` 或 batch sampling 改变：

- minute：train 日期范围内的每根 minute feature，计一次；
- daily：train 日期范围内每个 completed Daily，计一次；再加该范围内**每根 minute anchor 对应的 partial Daily snapshot**，各计一次；
- weekly：train 日期范围内每个 completed Weekly，计一次；再加该范围内**每根 minute anchor 对应的 partial Weekly snapshot**，各计一次。

任何 validation/test snapshot 都不能参与 statistics fit。零方差维度的 std 设为 1；checkpoint 保存各 schema group 的统计量、fit count 和有序 feature names。

preflight 数据报告必须把 Daily/Weekly 每个数值 feature（包含 `source_bar_count`）分别打印以下 train-only mean/std、样本数和分位数：

- completed token；
- all partial token；
- early-session partial：按该 trading day/week 最终 minute count 计算，`source_bar_count / final_source_bar_count <= 0.25`；
- late-session partial：`source_bar_count / final_source_bar_count > 0.75`。

上述 early/late 比例只用于离线数据审计，不作为模型 feature，也不参与 normalization。V0 仍只使用一套 Daily normalizer 和一套 Weekly normalizer；若各组分布差异异常，只在后续新文档版本中预注册变更，不能根据 test 表现临时拆分 normalizer。

### 3.4 样本定义

anchor `t` 是最后一根已完成的当前 minute bar。默认样本包含：

- minute context：`[t-511, ..., t]`，固定 512 根，每个 token 包含 market features + current-context-only time observation；
- daily context：文件历史中 `D(t)` 之前的全部 completed Daily + 当前 `partial Daily(t)`；
- weekly context：文件历史中当前 trading week 之前的全部 completed Weekly + 当前 `partial Weekly(t)`；
- target H：`[t+1, ..., t+H]`，H 为 16、64、256，**只包含 market features**；
- timestamp/index：anchor timestamp 和原始行 index。

Daily/Weekly **只属于当前上下文，绝不进入 future target**。future time-of-day、weekday、delta/gap 同样绝不进入 target。三个 target 分别是 EMA minute market-only encoder 对纯未来 market-feature 窗口 `[t+1, ..., t+H]` 的 stop-gradient 编码：

```text
Z_market(t) -> Predictor_H -> EMA Minute Market Encoder(X_market[t+1:t+H])
```

target 保留相对 sequence position 和已知 horizon length，但不含绝对/交易日历 clock。这样任务只回答“当前多尺度市场状态能否预测未来 minute market behavior representation”，同时消除复制 Daily/Weekly latent 或推演未来时钟就能降低 loss 的捷径。

### 3.5 Future outcome contract

评估使用的 future outcomes 固定如下，所有索引均指原始 minute 行，不能按 wall-clock 补齐不存在的分钟。令 `C_t` 为 anchor close，`H ∈ {16, 64, 256}`：

\[
Return_H = \frac{C_{t+H}}{C_t} - 1
\]

\[
MFE_H = \max_{j=1,\ldots,H}\left(\frac{High_{t+j}}{C_t} - 1\right)
\]

\[
MAE_H = \min_{j=1,\ldots,H}\left(\frac{Low_{t+j}}{C_t} - 1\right)
\]

\[
r_{t+j} = \log\left(\frac{C_{t+j}}{C_{t+j-1}}\right), \qquad
RV_H = \sqrt{\frac{1}{H}\sum_{j=1}^{H}r_{t+j}^2}
\]

`RV_H` 不做年化。Return/MFE/MAE 是相对 anchor close 的简单收益率；RV 的每一项是相邻观测 minute close-to-close log return，跨 session 的第一项因此包含真实隔夜/休市 gap。

### 3.6 训练前数据 preflight audit

每次正式训练前必须针对输入文件生成机器可读 CSV/JSON audit，至少包含 source SHA-256、行数、时间范围、trading-day/week 数量、缺失/非有限值、重复/非单调 timestamp，并输出以下异常 Top 100（降序、固定 tie-break 为 source index）：

1. 最大绝对 1m return：`abs(log(C_i / C_{i-1}))`；
2. 最大跨 observation gap：对 `delta_minutes > 1` 的行按 `abs(log(Open_i / C_{i-1}))` 排序，同时记录前后 timestamp 和 delta minutes；
3. 最大 Volume 跳变：`abs(log1p(Volume_i) - log1p(Volume_{i-1}))`；
4. 最大 OI 跳变：`abs(log1p(OI_i) - log1p(OI_{i-1}))`。

每项记录 rank、当前/前一 source index、timestamp、原始值和计算值。preflight 还包含 3.3 节规定的 completed/partial/early/late normalization 分布报告。非单调时间、重复时间、非法 OHLC、负 Volume/OI、NaN/Inf 属于 hard failure；Top 100 异常只报告，不自动删除、winsorize、复权或修复。

训练入口默认先运行 preflight，并把 audit 路径、source SHA-256 和 audit 配置写入 checkpoint。恢复训练时 source hash 不一致必须停止；smoke run 可以减少训练样本，但不能跳过结构性数据校验。

## 4. Chronological split 与 purge

V0 预注册并冻结以下闭区间，以推断后的 `trading_day` 判定；范围中无行情的自然日自动跳过：

| Split | Start | End | 用途 |
| --- | --- | --- | --- |
| Train | 2018-01-02 | 2022-12-31 | encoder/predictor、normalization、ridge 与 train reference |
| Validation | 2023-01-01 | 2024-12-31 | 模型/配置选择与 Validation bootstrap |
| Test | 2025-01-01 | 2025-12-02 | 一次性最终 OOS，不参与选择 |

变更这些日期必须升级设计文档版本并视为新实验，不能覆盖 V0 结果。日期范围定义 anchor/target 的归属；历史 context 可以来自 split start 之前。样本进入某 split 必须同时满足：

- anchor 位于该 split；
- 最大 horizon 的最后一根不晚于 split end。

因此只在 split 尾部 embargo 256 根，不丢弃 validation/test 开头已有足够全局历史的 query。任何 train target 都不会进入 validation/test，任何 validation target 都不会进入 test；validation/test context 可以使用更早 split 的历史，这是实际 causal inference 可用的信息。Daily/Weekly snapshot 可包含 anchor 当时已形成的 current partial，但绝不能包含 anchor 后才形成的最终 high/low/close/volume/OI。

自动化测试将直接比较每个样本的 `anchor_index`、`target_end_index` 与 split 原始边界，并单独确认 validation 首个样本的 minute context 可以跨到 train。normalization fit row mask 仍严格位于 train 日期范围，不含 validation/test；更早的 causal context 使用这组 train statistics 变换，但不参与 statistics fit。

## 5. 模型

### 5.1 Market encoder

- Minute Market Encoder：**只接收 market features**，使用 4 层 Temporal Transformer，`d_model=256`，8 heads，FFN 1024，输出 `Z_minute_market ∈ R^256`；
- Minute Context Encoder：独立接收完整 512-token context-observation sequence `[time_of_day_sin/cos, day_of_week_sin/cos, log1p_delta_minutes]`，使用 1 层 GRU、hidden size 32，输出最后 hidden state `Z_time ∈ R^32`；
- Daily encoder：2 层 GRU，hidden size 128，以最后有效 hidden state 输出 `Z_daily`；
- Weekly encoder：2 层 GRU，hidden size 128，以最后有效 hidden state 输出 `Z_weekly`；
- Minute Market Transformer 使用 learnable CLS token 与 sinusoidal sequence-local position（每个输入窗口从 position 0 开始），支持长度 16/64/256/512；Context/Daily/Weekly GRU 使用 packed variable-length sequence。即使文件首个 trading day/week 尚无 completed history，Daily/Weekly 序列仍含当前 partial token，因而长度至少为 1；
- fusion concat `Z_minute_market/Z_time/Z_daily/Z_weekly`（总维度 544），经 `544 -> 512 -> 256` MLP 输出 `Z_market`，不接收 contract metadata。

```text
past minute market ── Minute Market Transformer ── Z_minute_market ─┐
past time/gap seq  ── 1-layer GRU (hidden 32) ──── Z_time ──────────┤
Daily market+count ── 2-layer GRU (hidden 128) ─── Z_daily ─────────┼─ Fusion ─ Z_market
Weekly market+count ─ 2-layer GRU (hidden 128) ─── Z_weekly ────────┘

future minute market ─ EMA copy of Minute Market Transformer ─ Z_future_market
```

Minute Market Encoder 与 Minute Context Encoder 是参数和 forward API 完全独立的模块；market token 与 context-observation token 不相加、不 concat 后送入同一个 Transformer，也不在 attention 前发生任何交互。两路信息第一次交互只能发生在 Fusion。

GRU 把 full-history branch 的单样本复杂度从 self-attention 的 O(D²) 降为 O(D)。Dataset/Collate 保留 daily/weekly length。相同 trading day 内历史 completed prefix 相同，但末尾 partial token 每分钟变化；任何去重/缓存只能复用 completed prefix，不能缓存或复用未来时点的 partial state。

Ablation mode 为 `minute`、`minute_daily`、`minute_daily_weekly`。`minute` 固定包含 `Z_minute_market + Z_time`；后两种依次增加 Daily、Weekly。禁用分支用同维度零向量占位，从而保持 fusion 和实验接口一致；训练和 evaluation 必须使用 checkpoint 中相同 mode。

### 5.2 EMA target 与 predictor

target Minute Market Encoder 是 online Minute Market Encoder 的完整同构 EMA 副本：二者具有相同 class、相同 market-feature input schema、相同 projection/CLS/Transformer/output norm 和 forward 语义。唯一数据差异是 online 输入过去 market window，target 输入未来 market window。target 不持有也不调用 Minute Context Encoder。

target 参数 `requires_grad=False` 且不传给 optimizer。Minute Context、Daily、Weekly 和 Fusion 没有 target 副本。每个 optimizer step 后，对 online/target Minute Market Encoder 的全部一一对应参数执行：

`target = tau * target + (1 - tau) * context`

三个独立 predictor head 均为 `256 -> 512 -> 256`，分别对应 H16/H64/H256。target minute forward 位于 `torch.no_grad()` 内，其函数签名只接受 market features。测试必须在 Dataset schema 和 model API 两层确认 future clock/calendar/gap tensor既未返回也无法传入，并确认 online/target Minute Market Encoder 的 state-dict keys 与 tensor shapes 完全一致。

### 5.3 Loss 与 collapse diagnostics

对 predictor 和 stop-gradient target 先逐样本 L2 normalize，JEPA prediction loss 为三个 horizon cosine distance 的均值。V0 默认正式目标只有该 prediction loss：`lambda_var=0`、`lambda_cov=0`。

VICReg 风格项仅作为确认 collapse 后可显式开启的 fallback：

- variance：惩罚每维 batch std 低于配置阈值；
- covariance：惩罚中心化 latent covariance 的非对角项；
- total = prediction + `lambda_var * variance + lambda_cov * covariance`；默认两个 lambda 都为 0。

无论正则是否开启，每个 train/validation epoch 都分别记录：

- `prediction_loss_H16/H64/H256` 及总 prediction/total loss；
- current `Z_market` 的 latent mean std、effective rank、低于 std 阈值的维度比例、covariance off-diagonal RMS；
- `target_H16`、`target_H64`、`target_H256` **各自独立**的 latent mean std、effective rank、低 std 维度比例和 covariance off-diagonal RMS。

effective rank 固定为 covariance eigenvalue probability 的 `exp(entropy)`。任一 horizon 的 target mean std 低于 `collapse_threshold`，都必须单独标记该 horizon collapse，不能被三 horizon 平均值掩盖，也不能仅根据总 loss 宣称训练有效。是否启用 VICReg fallback 是 validation 阶段的预先记录实验变更，不能根据 test 结果决定。

### 5.4 冻结的 V0 default training protocol

以下配置是唯一可称为“V0 default experiment”的正式协议：

| 项目 | 冻结值 |
| --- | --- |
| seed | 42 |
| optimizer | AdamW，`betas=(0.9, 0.999)`，`eps=1e-8` |
| learning rate | `3e-4` |
| weight decay | `0.05`，应用于全部 optimizer 参数 |
| micro batch size | 64 |
| gradient accumulation | 2（effective batch 128） |
| max epochs | 50 |
| warmup | 总 optimizer steps 的前 5%，linear `0 -> 3e-4` |
| scheduler | warmup 后 cosine decay `3e-4 -> 0`，无 restart，per optimizer step |
| gradient clipping | global norm 1.0，在 AMP unscale 后、optimizer step 前 |
| EMA tau | 常数 0.996 |
| AMP | CUDA float16 autocast + GradScaler；CPU smoke 关闭 |
| anchor stride | 1，不下采样 train/validation/test anchor |
| early stopping | 禁用 |

optimizer 只包含 online Minute Market Encoder、Minute Context Encoder、启用的 Daily/Weekly encoder、Fusion 和三个 predictors；EMA target 的任何参数都不得进入 optimizer。训练 sampler 每 epoch 用 `seed + epoch` 确定性 shuffle，只保留完整的 128-sample effective batch；每个 effective batch 拆成两个 64-sample micro batch，loss 各除以 2 后 backward。`steps_per_epoch = floor(num_train_samples / 128)`，`total_optimizer_steps = steps_per_epoch * 50`，warmup steps 取 `ceil(0.05 * total_optimizer_steps)`。EMA 只在实际 optimizer step 后更新一次，不在 micro batch 后更新。

Python、NumPy、PyTorch CPU/CUDA 和 DataLoader generator 都设 seed=42；`cudnn.benchmark=False`、`torch.use_deterministic_algorithms(True)`。如果某算子在目标硬件上不支持 deterministic mode，正式训练 hard fail，不静默降级。Validation 每 epoch 在全部 validation anchors 上、`eval()`/无 shuffle/无梯度执行。

正式 V0 的 `lambda_var=0`、`lambda_cov=0`，三个 horizon prediction loss 等权平均。发现 collapse 后开启 VICReg、改变 seed/lr/tau/stride/batch/epoch 或任何冻结值，都必须使用新 experiment id 并在运行 **Test 之前**登记为非 default follow-up；不能覆盖 default run，也不能把 follow-up 结果称为预注册 V0 default。

### 5.5 Checkpoint selection 与 Test 使用

- 每 epoch 保存可恢复的 `last` checkpoint，但它不用于最终 Test 选择；
- `best` 的唯一选择指标是全量 Validation 上的 **H64 learned-to-target mean cosine error**，取最小值；完全相等时保留更早 epoch；
- 不查看 shuffled、persistence、probe、kNN 或 Test 指标来选择 checkpoint；
- 50 epochs 完成后冻结 best checkpoint 文件和 SHA-256；latent export、四类最终 evaluation 与三种 ablation 的 Test 只能使用各自按同一规则选出的 best checkpoint；
- Test 对 best checkpoint 只运行一次，evaluation manifest 记录 checkpoint hash、source hash、运行时间和输出 hash。任何再次运行或代码/数据/hash 变化必须标记为 rerun，不能替代首次 Test 记录。

Smoke run 可通过独立 `--smoke` profile 缩小模型/样本/epoch 以验证链路，但产物目录、experiment id 和 manifest 必须带 `smoke`，不得参与 checkpoint selection、GO/NO-GO 或正式报告。

## 6. 工程结构与命令

```text
market_jepa/
  config.py
  data/{features,trading_day,dataset}.py
  model/{encoder,jepa}.py
  train/trainer.py
  eval/{metrics,pipeline}.py
configs/market_jepa_v0.yaml
train_market_jepa.py
export_market_latents.py
eval_market_jepa.py
tests/
```

使用 `uv` 管理 `.venv`、lockfile 和命令：

```bash
uv sync
uv run pytest
uv run python train_market_jepa.py --config configs/market_jepa_v0.yaml
uv run python export_market_latents.py --config ... --checkpoint ... --split test
uv run python eval_market_jepa.py --config ... --checkpoint ...
```

CLI 只允许在显式 `--smoke` profile 下限制 samples/epochs/model size。正式 V0 的 anchor stride 固定为 1，不能被 CLI 覆盖；三种 ablation 必须使用完全相同的 anchor indices。

## 7. Checkpoint 与导出

checkpoint 保存 online Minute Market Encoder、Minute Context Encoder、Daily/Weekly encoder、Fusion、EMA target Minute Market Encoder、三个 predictors、optimizer/scheduler/GradScaler、Python/NumPy/Torch CPU/Torch CUDA RNG state、epoch/global step、完整配置、三个 timeframe normalization、feature definitions（含 minute 时间 observation 与 Daily/Weekly `source_bar_count`）、固定 split ranges、symbol、series id、source SHA-256 和 preflight audit metadata。恢复时校验 source hash、feature 顺序、online/target market-encoder 同构性和 ablation mode，并在继续下一 epoch 前恢复全部 RNG state。

latent 导出为 `.npz`，包含 timestamps、trading days、symbol、series id、split、anchor indices、Daily/Weekly partial 的 source-index bounds、`Z_market`、每个 horizon 的 predictor/target/persistence latent、fair raw baseline features，以及 H16/H64/H256 的 return、MFE、MAE、realized volatility。evaluation 只消费导出文件，无需重新训练。

## 8. Evaluation

- Future prediction：每个 horizon 报告 learned predictor 对 market-only true target 的 cosine/MSE，并与三个 control 比较：
  - persistence：用同一 EMA minute market-only encoder 编码紧邻 anchor、长度同为 H 的过去 **market-feature-only** 窗口 `[t-H+1, ..., t]`；不传时间/calendar/gap，确保与 future target 位于同一表示空间；
  - train mean：train market-only target latent 的逐维均值，对 validation/test 固定不变；
  - block shuffle：只在相同 calendar month 与 session（日盘/夜盘）stratum 内循环置换 target，固定 seed；不足两个样本的 stratum 不计该 control。
- kNN：query 仅从 validation/test，候选仅从 train；额外应用 `candidate_timestamp < query_timestamp` 断言。K=20/50/100，比较 latent neighbors 与固定 seed random historical samples的 future return/MFE/MAE/vol 分布误差及邻居统计。
- Frozen linear probe：只用 train latent 拟合 ridge closed-form 回归，分别在 validation/test 测 H16/H64 return/vol；encoder 不参与该过程。公平 raw baseline 使用与对应 ablation 完全相同的可见信息：
  - minute 最近 16/64/256/512；daily 最近 5/20/60/all；weekly 最近 4/13/26/all；daily/weekly 各窗口最后一项均为 anchor 时点的 current partial token；
  - 上述窗口统计只对 market features 计算 last/mean/std/min/max/slope；slope 是 feature 对归一化位置 `[-1, 1]` 的 OLS 系数；
  - 另外原样加入 anchor 当前的 `time_of_day_sin/cos`、`trading day_of_week_sin/cos`、`log1p_delta_minutes`、current partial Daily `source_bar_count`、current partial Weekly `source_bar_count`，不对这些 context-only observation 再做窗口统计；
  - 历史短于窗口时用全部已有历史；由于 current partial 永远存在，daily/weekly 序列长度至少为 1，并输出显式 token count；
  - minute-only ablation 的 raw baseline 不含 Daily/Weekly count，minute+Daily 只增加 Daily count，三分支才增加 Weekly count；
  - raw statistics 与 latent 使用相同 train rows、相同 ridge alpha、相同目标和相同 OOS rows。
- OOS stability：所有指标分别打印 train 日期、validation 日期和 test 日期；test 不参与超参数选择。

三种 ablation 必须分别训练 checkpoint/导出 latent，再由 evaluation 汇总相同 split 和指标。V0 提供 mode 支持与单实验报告，不把未经运行的 ablation 结果写成结论。

### 8.1 预注册 GO / NO-GO

为避免看完 test 再定义成功，主 endpoint 预注册为 H64：

1. prediction endpoint：逐样本 `persistence cosine error - learned cosine error`；
2. kNN endpoint：query 的实际 future outcome 向量与 K=50 邻居 outcome 均值的误差，相对 random historical K=50 的改善；outcome 为 H64 return/MFE/MAE/realized-volatility 经 train scale 标准化后的四维向量；
3. probe endpoint：H64 return 与 realized-volatility 的 train-target-variance-normalized MSE 平均值，`raw baseline - latent probe`。

Validation 对 trading day 做 block bootstrap（固定 seed，10,000 次），但 point estimate 始终是全部逐样本 effect 的算术平均。每次 bootstrap 有放回抽取 trading-day block；抽中某天时带入该日全部 sample，并用所有抽中 block 的 `sum(effect) / count(sample)` 得到一次统计量，不能先对每日均值再等权平均。每项改善的 95% percentile CI 下界必须大于 0。Test 不再选模型/alpha/lambda，三项 point estimate 必须与 Validation 同方向；考虑单年样本统计功效，Test CI 不强制显著。只有三项都满足才标记 GO，否则标记 NO-GO/INCONCLUSIVE，并完整报告 effect size 和 CI。H16/H256、K=20/100、shuffle/train-mean control 是预注册 secondary endpoints，不替代失败的 H64 主 endpoint。

## 9. 验证计划

1. 数据与因果性单元测试：
   - 夜盘第一根 minute 后，当前 trading day 的 partial Daily 已存在；
   - 夜盘结束后的 partial Daily OHLCV/OI 等于截至该时刻的可见 minute 聚合；
   - 次日日盘第一根只更新同一个 trading-day Daily，不新增 Daily token；
   - 当前 partial Daily High/Low 不得看到 anchor 后的价格；
   - 当前 partial Weekly 必须包含当前 partial Daily；
   - 周中相邻 minute anchor 的 partial Weekly 会按新数据正确更新；
   - 周五夜盘若归属下一 trading week，partial Weekly 从下一周开始；
   - 任意 Daily/Weekly snapshot token 的 `source_max_index <= anchor_index`，两个 current partial 的 `source_max_index == anchor_index`。
2. 其余单元测试：minute `/5` weekday 与其他时间 observation 数值、raw `delta_minutes` 不在 model schema、跨 session `log1p_delta_minutes`、`source_bar_count` 更新、固定 split 尾部 purge/开头可用历史、四个 outcome 公式、train-only normalization（含 partial fit population）、Minute Market/Context 两个 encoder 参数与 forward 隔离、online/EMA Market Encoder class/schema/state-dict 同构、EMA 初始化/更新、target 无梯度且不在 optimizer、三个 future Dataset item 不含 clock/calendar/gap、target API 拒绝 context observation、predictor/target shape、checkpoint round-trip、latent timestamp、raw baseline 获得对应 ablation 的同等 current observation、kNN past/train-only、controls 和 bootstrap 可复现。
3. Preflight 测试：Top 100 排序/tie-break、异常公式、hard-failure 条件、source hash、completed/partial/early/late 报告完整性与可复现。
4. 小型合成/真实切片 CPU smoke：完整执行 preflight -> train -> resume/load -> export -> 四类 evaluation。
5. 正式训练前必须在目标 RTX 4060 Ti 上运行 CUDA smoke，真实执行 Minute Market Transformer、Minute Context/Daily/Weekly GRU、Fusion、三个 target/predictor 的 forward、AMP backward、GradScaler、clip、optimizer、scheduler 和 EMA；同时启用 deterministic 配置。任一步失败都不得启动 50-epoch 正式训练。
6. 检查训练日志确认 loss 为有限值、至少在 smoke optimization steps 上下降，并确认 H16/H64/H256 target diagnostics 分别存在且未被平均掩盖。
7. 自审 git diff，确认无数据文件改写、无调试日志、无未来泄漏路径。

## 10. 已知限制

- 原始文件没有权威 trading day；当前映射是可测试的推断规则，生产研究前应与权威交易所日历或用户提供的 trading_day 数据逐日核对。
- 原始文件名虽含 `JM2601`，却覆盖 2018--2025；无法由现有证据判断它是单合约回填、连续序列还是复权序列。这会影响合约元数据解释，V0 不擅自推断。
- 没有可靠 listing/expiry 日期，因此 V0 不提供任何 contract metadata。
- 单个合约/序列不能证明跨品种泛化；GO/NO-GO 结论只覆盖本数据和明确 OOS 区间。
- GRU 已把 daily/weekly 单样本编码降为 O(D/W)，但相同交易日内重复历史仍有计算冗余；全量训练资源成本需要在 smoke 通过后评估，必要时只做不改变样本语义的按日去重缓存。
