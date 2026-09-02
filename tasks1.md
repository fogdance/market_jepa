## Market-JEPA V0 数据输入修订：多时间尺度

V0 不再只使用 1 分钟数据。

Market state 必须同时包含：

1. 1 分钟局部行情
2. 当前合约从上市以来的完整历史日线
3. 当前合约从上市以来的完整历史周线

例如对 JM2701 某个 1 分钟 anchor timestamp `t`：

### Minute Context

使用 `t` 之前已经完成的最近 512 根 1 分钟 bar。

字段基础为：

* open
* high
* low
* close
* volume
* open_interest

默认：

`minute_context_length = 512`

### Daily Context

使用该合约从上市日起，到 anchor `t` 的“上一完整交易日”为止的全部日线数据。

不要固定截取最近 N 天。

如果 JM2701 截至该时间已经存在 183 个完整交易日，则输入完整 183 根日线。

禁止使用 anchor 所在交易日最终形成的日 OHLCV/OI，因为其中包含 anchor 之后的数据。

V0 不生成 partial current-day daily bar。

当前交易日截至 anchor 的所有信息由 1 分钟 branch 表达。

### Weekly Context

使用该合约从上市以来，到 anchor `t` 的“上一完整交易周”为止的全部周线数据。

禁止在周中使用本周最终形成的周 OHLCV/OI。

V0 不生成 partial current-week weekly bar。

### Trading Day Semantics

中国期货夜盘必须按照交易所 trading_day 语义处理，不得简单按照自然日期聚合。

例如夜盘 21:00 数据可能属于下一交易日。

优先复用现有 DreamerV3 数据里的 trading_day / contract calendar 逻辑。

如果已有独立可靠的 daily / weekly 数据源，应直接使用，不要为了 Market-JEPA 重新从 1m resample。

如果确实必须从 1m 聚合，则必须基于 trading_day，而不是 timestamp.date()。

### Features

三个 timeframe 都保留 OHLCV + OI 基础信息。

允许生成少量 causal derived features，例如：

* close log return
* open relative to previous close
* high relative to previous close
* low relative to previous close
* normalized high-low range
* log1p(volume)
* volume change
* log1p(open_interest)
* open_interest change
* realized volatility

任何 rolling feature 只能使用该时间点及之前的数据。

不同 timeframe 使用各自的 normalization statistics。

Normalization 只能根据 TRAIN 数据计算。

### Contract Metadata

加入：

* days_since_listing
* days_to_expiry

如果现有数据中能够可靠提供。

不得使用任何只有事后才能知道的“主力合约状态”等未来信息。

## 模型结构

使用三个独立 encoder：

Minute Encoder:

* input: recent 512 × minute_features
* 4-layer Temporal Transformer
* d_model = 256
* output: Z_minute ∈ R^256

Daily Encoder:

* input: all completed daily bars from listing to t
* variable-length sequence + padding mask
* 2-layer Temporal Transformer
* d_model = 128
* output: Z_daily ∈ R^128

Weekly Encoder:

* input: all completed weekly bars from listing to t
* variable-length sequence + padding mask
* 2-layer Temporal Transformer
* d_model = 128
* output: Z_weekly ∈ R^128

Fusion:

concat(
Z_minute,
Z_daily,
Z_weekly,
contract_metadata
)

→ MLP

→ Z_market ∈ R^256

V0 不使用 cross-attention，不做复杂 hierarchical transformer。

## Prediction Horizons

V0 使用至少三个 future horizon：

* H16
* H64
* H256

Market-JEPA 的目标仍然不是直接预测 future price。

目标是从当前多周期 market representation 预测未来 representation / future market behavior representation。

## Ablation

Evaluation 必须支持三种输入模式：

A. 1m only

B. 1m + Daily

C. 1m + Daily + Weekly

三种实验使用相同数据切分、训练和 evaluation 方法。

最终比较：

* true future prediction vs shuffled control
* latent kNN future distribution
* frozen linear probe
* OOS stability

用实验验证 daily / weekly context 是否真正增加信息，而不是先假设其一定有效。
