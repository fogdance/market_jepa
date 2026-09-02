实现一个独立的 `Market-JEPA V0` 实验，用于验证：

“市场历史中是否存在一个可学习的 latent state Z，使得相似 Z 对应更相似的未来市场分布。”

## 1. 最重要的约束

优先复用项目现有 DreamerV3 数据读取流程和数据格式。

现有基础数据为：

* 1分钟 OHLCV
* OI / Open Interest

不要重新设计行情存储格式，不要另外建立新的数据转换体系。

如果现有 Dataset/DataLoader 可以复用，直接复用；如果耦合较重，可以做薄适配层，但不要复制整套数据逻辑。

不要破坏现有 DreamerV3 训练代码。

Market-JEPA 应作为独立实验模块存在。

---

## 2. V0 暂时不要实现

不要加入：

* BUY / SELL / HOLD
* position
* Actor
* Critic
* Reward
* PnL optimization
* transaction cost
* RL
* Dreamer imagination
* 新闻
* LLM
* order book
* 多模态
* 聚类交易策略

V0 只验证 representation learning。

---

## 3. 数据输入

使用连续的 1 分钟行情窗口。

基础原始字段：

* open
* high
* low
* close
* volume
* open_interest

允许从这些字段计算少量基础、无未来信息的 normalized features，例如：

* log_return
* open-close return
* high-low range
* volume change / normalized volume
* OI change
* rolling realized volatility
* normalized range
* time-of-day encoding

不要加入大量 TA-Lib 技术指标。

所有 rolling feature 必须只使用当前时刻及以前的数据。

Normalization 参数只能使用 train split 计算。

严禁使用 validation/test 数据计算 mean/std/quantile 等统计量。

---

## 4. Dataset

默认：

context_length = 256

即使用过去 256 根 1m bar：

X[t-255:t]

预测未来 latent。

第一版使用两个 horizon：

* H=16
* H=64

Target windows 分别来自：

X[t+1:t+16]

X[t+1:t+64]

注意处理边界和 overlap。

Dataset 返回至少：

context
target_h16
target_h64
timestamp/index

---

## 5. 时间切分

严禁 random split。

必须 chronological split。

实现：

train
validation
test

并支持配置日期范围。

Train / validation / test 边界必须考虑：

context window
future target window

需要 embargo / purge，避免 target window 跨越 split boundary。

增加自动化测试验证：

任何 train sample 的 context 或 target 都不能进入 validation/test 时间范围。

任何 validation sample 的 target 都不能进入 test。

---

## 6. 模型结构

### Context Encoder

第一版使用小型 Temporal Transformer。

建议默认：

d_model = 256
num_layers = 4
num_heads = 8
ffn_dim = 1024
dropout = 0.1

输入：
[B, T, F]

输出一个固定维度 latent：

Z_now
[B, 256]

可以使用 CLS token 或经过验证的 temporal pooling。

---

### Target Encoder

架构与 Context Encoder 相同。

Target Encoder 不通过 optimizer 直接更新。

使用 Context Encoder 参数 EMA 更新：

target_param =
tau * target_param

* (1 - tau) * context_param

默认 tau 可配置，例如：

0.996 / 0.999

Target Encoder forward 必须 stop-gradient。

---

## 7. Predictor

从 Z_now 分别预测：

Z_future_16
Z_future_64

建议两个独立 predictor head。

可以先使用简单 MLP：

256
→ 512
→ 256

不要一开始设计复杂 dynamics model。

---

## 8. Loss

基础目标：

pred_h16 ≈ target_encoder(target_h16)

pred_h64 ≈ target_encoder(target_h64)

需要防止 representation collapse。

请研究并实现一个简洁可靠的 JEPA/self-supervised loss。

至少监控：

* JEPA prediction loss
* latent variance
* latent std per dimension
* feature covariance / collapse indicators

如果全部 latent 变成常数，训练必须能够检测出来。

不要只看 loss 是否下降。

---

## 9. checkpoint

保存：

* context encoder
* target encoder
* predictors
* optimizer
* training config
* normalization statistics
* feature definitions
* train/val/test ranges

保证 checkpoint 可完全复现实验。

---

## 10. latent export

实现单独命令，把 frozen encoder 应用于指定数据范围：

timestamp → latent vector

例如输出：

parquet / npz

至少包含：

timestamp
symbol
Z[0:256]

后续 evaluation 不应需要重新训练模型。

---

## 11. 必须实现的四类 Evaluation

### A. Future prediction vs shuffled control

计算真实：

Z_now → true future Z

的 prediction metric。

然后把 future target 随机打乱：

Z_now → shuffled future Z

比较结果。

如果真实未来和 shuffled target 差异不明显，则说明模型没有学到有效 temporal predictive structure。

---

### B. Latent kNN

对于每一个 OOS query latent：

找到历史 TRAIN latent 中最近的 K 个邻居。

严禁从未来数据找邻居。

默认：

K = 20 / 50 / 100

统计这些邻居之后真实市场的：

* future return H16
* future return H64
* MFE
* MAE
* future realized volatility

输出：

mean
median
quantiles
positive probability
negative probability

同时做 random-neighbor control。

比较：

latent nearest neighbors

vs

随机历史样本

的未来分布是否明显更集中或更相似。

---

### C. Frozen Linear Probe

Market-JEPA 完成以后：

冻结 encoder。

禁止 fine-tune。

用 Z_t 训练非常简单的 linear model，预测：

* H16 return
* H64 return
* H16 realized volatility
* H64 realized volatility

可以包含 classification 和 regression。

必须与至少一个 baseline 比较：

raw engineered features
→ linear model

也就是说比较：

Z → Linear

vs

Raw Features → Linear

如果 JEPA latent 没有带来任何 OOS 信息增益，需要明确报告。

---

### D. Cross-time OOS evaluation

训练只能使用过去。

所有最终报告必须区分：

Train
Validation
Test

禁止把 test 结果用于调参后再次称作 test。

Evaluation 输出中明确打印数据日期。

---

## 12. GO / NO-GO

V0 不以交易盈利作为验收标准。

关注以下问题：

1. True future latent prediction 是否显著优于 shuffled control？
2. Latent nearest neighbors 的未来市场分布是否比 random neighbors 更一致？
3. Frozen latent linear probe 是否优于 raw-feature linear baseline？
4. 这些现象是否在完全 OOS 时间段仍然存在？

仅有：

* train loss 下降
* validation loss 下降
* PCA 图漂亮
* 某几个案例看起来不错

都不能认为成功。

---

## 13. 项目结构

优先保持简单，例如：

market_jepa/
data/
model/
train/
eval/
tests/

configs/
market_jepa_v0.yaml

train_market_jepa.py
eval_market_jepa.py
export_market_latents.py

不要建立复杂框架和过度抽象。

如果已有项目结构适合，则服从现有结构。

---

## 14. 测试

重点写测试验证：

* 时间窗口没有未来泄漏
* split 没有 overlap
* normalization 没使用 test
* EMA target encoder 正确更新
* target encoder 没有 gradient
* predictor tensor shape 正确
* checkpoint save/load 一致
* latent export 时间戳正确
* kNN 不会检索 query 之后的数据

金融数据泄漏问题优先级高于模型功能。

---

## 15. 第一阶段完成标准

首先做到：

1. 单元测试通过
2. 小数据集可完整跑通
3. loss 正常下降且 latent 不 collapse
4. checkpoint 可恢复
5. latent 可以导出
6. evaluation 四套流程可以完整运行

不要先追求模型效果。

完成后整理 README，给出：

训练命令
评估命令
latent export 命令
数据要求
配置说明

并记录当前所有已知限制。
