# Market World Model 项目总纲（原始设计主线版）

**文档定位：整个项目的最高层设计基准与防跑偏文档**  
**作用：回答“我们最初到底想做什么、为什么这样做、每一阶段为什么存在、什么时候才允许进入下一阶段”。**  
**优先级：高于任何具体 V0.x 设计文档、实现细节、性能优化方案和单次实验结果。**

---

# 0. 一句话定义整个项目

我们最终想做的不是一个“预测下一根 K 线”的模型，也不是一个直接输出 BUY / SELL 的强化学习 Agent。

真正目标是：

\[
\boxed{
ObservedMarketHistory
\rightarrow
HiddenMarketState
\rightarrow
FutureMarketBehaviorDistribution
}
\]

然后才有一个**独立于世界模型**的决策层：

\[
\boxed{
FutureDistribution
+
RiskRules
\rightarrow
Position
}
\]

整个项目的核心思想是：

> **先建立市场世界模型，再考虑如何利用这个世界模型交易。**

---

# 1. 最初为什么从 DreamerV3 转向 Market World Model

我们之前尝试过 DreamerV3。

Dreamer 的标准世界观更适合机器人 / 游戏：

\[
S_t + A_t \rightarrow S_{t+1}
\]

也就是说：

- 当前状态 \(S_t\)
- Agent 动作 \(A_t\)
- 动作会改变世界
- 世界进入 \(S_{t+1}\)

例如：

```text
机器人：
看到桌子
↓
向左移动机械臂
↓
桌面上的物体状态发生变化
```

但交易不是这个世界。

一个普通交易者的：

```text
BUY
SELL
HOLD
```

基本不会决定焦煤市场下一分钟如何运动。

市场更接近：

\[
Market_{t+1}
\approx
f(Market_t, HiddenWorld_t) + Noise
\]

而不是：

\[
Market_{t+1}
=
f(Market_t, TraderAction_t)
\]

这意味着：

> **市场对我们来说首先是一个“不可控但可观察”的动态世界。**

我们的 Action 主要改变：

- 自己的仓位
- 自己的 PnL
- 自己的风险

而不是市场状态本身。

---

# 2. DreamerV3 哪一部分不适合，哪一部分非常有价值

必须把 Dreamer 拆开看。

Dreamer 大致包含：

```text
Observation Encoder
       ↓
RSSM / Latent World State
       ↓
Imagined Dynamics
       ↓
Reward Model
       ↓
Actor / Critic
       ↓
Action
```

对于交易：

## 不适合直接照搬的部分

### 1. Action-conditioned market transition

标准 Dreamer 假定：

\[
S_{t+1} = f(S_t,A_t)
\]

交易里这个因果关系基本不成立。

### 2. Reward-driven representation

如果世界模型和 Actor / Critic 一开始就绑在：

```text
PnL
Reward
Position
```

上，很容易出现：

> 模型只学习对当前 reward function 有利的历史模式，而不是更真实的 Market State。

### 3. Actor 利用 world-model error

在 model-based RL 中，Actor 可能找到世界模型的错误区域并“钻漏洞”。

金融市场本身：

- 高噪声
- 非平稳
- 部分可观察
- regime 会变化

这个问题会更加严重。

因此：

> **DreamerV3 整套端到端 RL 不是我们这个项目的主线。**

---

## Dreamer 中真正值得保留的部分：RSSM

Dreamer 最有价值的思想之一，是：

> 世界的真实状态无法直接观察，因此模型维护一个持续更新的 latent belief state。

形式上：

\[
B_t = f(B_{t-1}, O_t)
\]

其中：

- \(O_t\)：当前市场观察
- \(B_{t-1}\)：之前已经积累的 belief
- \(B_t\)：当前隐藏市场状态

这非常符合交易。

市场真实驱动力可能包括：

- 资金结构
- 供需变化
- 市场参与者状态
- regime
- 风险偏好
- 未观察到的信息

我们只能通过价格、成交量、OI、跨周期结构等不断更新自己的 belief。

所以：

\[
\boxed{
RSSM / Persistent Belief State
}
\]

从一开始就是**最终 Market World Model 的重要核心组件**。

不是普通的“可选增强”。

---

# 3. 为什么又选择 JEPA

如果直接让 RSSM 重建：

```text
未来所有 OHLCV
```

或者直接预测：

```text
下一分钟价格
```

模型会被迫学习大量：

- 高频噪声
- 不可预测细节
- observation-level reconstruction

而我们真正关心的是：

> 对未来行为有意义的抽象状态。

JEPA 的思想非常适合这个问题。

JEPA 不要求：

\[
Predict\ RawFuturePixels
\]

或者：

\[
Predict\ ExactFuturePrice
\]

而是：

\[
Z_{context}
\rightarrow
Z_{future}
\]

也就是：

> 从当前状态预测未来的抽象 representation。

在市场里：

\[
MarketHistory_t
\rightarrow Z_t
\]

然后：

\[
Z_t
\rightarrow
\hat Z_{future}
\]

这更接近：

> 当前市场隐藏状态，对未来市场行为的抽象状态有多少预测能力？

因此最初得到的核心方向是：

\[
\boxed{
JEPA + RSSM
}
\]

二者负责不同问题。

---

# 4. JEPA 与 RSSM 的职责必须区分

## JEPA 负责：

> **什么样的 representation 才是“有未来意义的市场状态表示”？**

即：

\[
History
\rightarrow
Z
\]

通过预测 future latent，迫使 \(Z\) 保留有预测价值的信息，而不是所有 observation 细节。

---

## RSSM 负责：

> **这个隐藏状态如何随着市场一根一根到来持续演化？**

即：

\[
B_t = f(B_{t-1},O_t)
\]

而不是每一分钟都完全重新依赖固定长度窗口。

---

## 最终主干

理想的最终 Market World Model 主干是：

```text
Market Observation_t
        │
        ↓
Observation / JEPA Encoder
        │
        ↓
   latent observation
        │
        ↓
RSSM / Persistent Belief Update
        │
        ↓
HiddenMarketState_t
        │
        ↓
JEPA-style Future State Predictor
        │
        ↓
Future latent trajectory / distribution
```

可以抽象成：

\[
O_t
\rightarrow
B_t
\rightarrow
P(B_{t+1:t+H})
\]

注意：

这里没有 Trader Action 进入 market dynamics。

即不是：

\[
B_{t+1}=f(B_t,A_t)
\]

而主要是：

\[
B_{t+1}=f(B_t,O_{t+1})
\]

以及：

\[
P(B_{future}|B_t)
\]

---

# 5. 为什么 V0 暂时不加入 RSSM

这是整个项目最容易被误解的一点。

我们说：

> **JEPA + RSSM 很适合最终设计。**

但 V0 刻意不加入 RSSM。

原因不是认为 RSSM 不重要。

恰恰相反：

> **因为 RSSM 太重要，所以必须先证明它所维护的 latent state 值得维护。**

如果第一版直接：

```text
JEPA
+
RSSM
+
transition model
+
recurrent memory
```

训练失败，我们无法知道：

```text
JEPA objective 不成立？
还是 RSSM 不行？
还是 recurrent transition 不行？
还是 belief update 不行？
还是固定窗口本来就没有预测信息？
```

变量太多，实验不可证伪。

所以 V0 故意降级成：

```text
固定历史窗口
        ↓
Encoder
        ↓
Z_t
        ↓
Predict Future Latent
```

它只验证最底层命题：

\[
\boxed{
市场历史中到底有没有一个值得学习的 predictive latent state？
}
\]

只有这个命题成立：

> 才值得花力气研究如何用 RSSM 持续维护这个 state。

所以：

\[
\boxed{
V0 不用 RSSM
\neq
最终设计不要 RSSM
}
\]

正确理解是：

\[
\boxed{
V0 是 JEPA+RSSM 最终路线之前的最小必要验证实验
}
\]

---

# 6. 整个项目的主线，不是通用“Phase 0-9 功能堆叠”

项目主线应该围绕一个核心 world-model architecture 逐步验证。

---

# Stage A：Predictive Latent 是否存在（当前 V0）

## 问题

\[
History_t
\rightarrow
Z_t
\]

得到的 \(Z_t\) 是否真的具有稳定未来意义？

## 方法

固定窗口 JEPA。

当前实现进一步使用：

```text
Minute
+
当前可见 partial Daily
+
当前可见 partial Weekly
```

形成：

\[
Z_{market,t}
\]

Future target 仍然是纯未来 market latent。

## 关键验证

### 1. Future Prediction

必须打败：

- Persistence
- Train mean
- Block shuffle

### 2. Latent kNN

\[
Z_i \approx Z_j
\Rightarrow
Future_i \approx Future_j
\]

### 3. Frozen Probe

latent 必须优于公平 raw-feature baseline。

### 4. Strict OOS

必须跨未来年份稳定。

## GO 的真正含义

只允许说：

> **固定历史窗口中存在一个可以被 JEPA 学到、并对未来行为具有稳定意义的 latent representation。**

这一步还不是完整 World Model。

它是在回答：

> **World State 值不值得建。**

---

# Stage B：JEPA + RSSM Persistent HiddenMarketState

只有 Stage A GO 后进入。

这是从实验性 representation 正式进入：

\[
\boxed{
Market World Model
}
\]

的关键阶段。

## 核心变化

V0：

\[
Z_t = Encoder(X_{t-L:t})
\]

每一分钟重新看固定窗口。

Stage B：

\[
B_t = RSSM(B_{t-1},O_t)
\]

即：

```text
旧 Hidden State
      +
新市场 Observation
      ↓
Persistent belief update
      ↓
新 Hidden Market State
```

---

## 为什么这比固定窗口更接近真实市场

市场状态可能持续数小时、数天甚至更久。

例如：

```text
OI持续变化
资金逐渐迁移
趋势结构演化
波动率 regime 切换
多周期状态积累
```

固定：

```text
last 512 minutes
```

存在人为记忆边界。

RSSM 则允许：

> 有用的信息持续留在 belief state 中，直到模型认为它不再重要。

这就是 Persistent Market State。

---

## Stage B 不是 DreamerV3 回归

非常重要。

我们只借：

\[
RSSM
\]

而不是重新加回：

```text
Actor
Critic
Reward
Trader Action → Market Transition
```

Stage B 仍然是：

```text
Observation
↓
Belief State
↓
Predict Future State
```

而不是：

```text
Action
↓
Imagined PnL
↓
Policy Optimization
```

---

## Stage B 必须比较

必须和 V0 fixed-window JEPA 正面对比。

如果：

```text
RSSM-JEPA
```

没有显著优于：

```text
Fixed-window JEPA
```

那么 RSSM 就没有必要保留。

最终设计不能因为“世界模型一般都应该 recurrent”就强行使用 RSSM。

---

# Stage C：Latent Dynamics / Future State Evolution

Stage B 证明 persistent state 有价值以后，下一步不是交易。

而是让模型真正理解：

> 当前 HiddenMarketState 未来可能如何演化。

目标从：

\[
Z_t \rightarrow Z_{future,H}
\]

进一步发展成：

\[
B_t
\rightarrow
P(B_{t+1},B_{t+2},...,B_{t+H})
\]

即：

```text
Current Hidden State
        ↓
Future latent trajectory
        ↓
multiple possible futures
```

这里仍然：

> **没有 Trader Action。**

因为市场是 externally-driven world。

---

# Stage D：Uncertainty / Future Distribution

市场未来不是唯一轨迹。

因此世界模型最终不能只输出：

```text
未来一定进入 Z_37
```

而应该：

\[
P(B_{future}|B_t)
\]

甚至映射到：

\[
P(Return,MFE,MAE,Volatility|B_t)
\]

这一阶段重点研究：

- multi-modal futures
- uncertainty
- calibration
- state transition probability
- regime persistence probability

这一步才开始接近真正意义上的：

> **Market World Model Simulator**

但仍然不是交易策略。

---

# Stage E：Cross-Market / Multimodal World State

在单一市场自身状态已经成立之后，才扩展 HiddenWorld。

因为最初我们就认识到：

\[
Market_{t+1}
\approx
f(Market_t,HiddenWorld_t)+Noise
\]

而：

\[
HiddenWorld_t
\]

并不只有焦煤自己的 K 线。

未来可以加入：

```text
相关期货
黑色产业链
美元
利率
宏观
库存
矿山事故
进出口
政策
新闻
公司公告
```

这里可以借鉴 Cosmos 一类 world-model 思想：

> 多 observation modality 共同估计一个隐藏世界状态。

目标是：

\[
MarketOnlyBelief
\]

升级成：

\[
WorldConditionedMarketBelief
\]

---

# Stage F：Generalization

直到这里才系统验证：

- cross-year
- cross-regime
- cross-contract
- cross-instrument

是：

```text
同一个 shared latent world
```

还是：

```text
每个品种完全独立的 state system
```

这会决定最终架构是：

```text
Shared World Backbone
+
Instrument-specific heads
```

还是完全独立模型。

---

# Stage G：World Model → Decision Layer

只有前面的世界模型被证明成立以后，才第一次允许出现：

```text
BUY
SELL
Position
```

而且仍然必须保持：

\[
WorldModel
\neq
DecisionPolicy
\]

结构：

```text
Market / World Observations
        ↓
JEPA + RSSM World Model
        ↓
Hidden Market State
        ↓
Future Distribution
        ↓
Independent Decision Layer
        ↓
Risk / Position
```

---

# 7. 为什么最终仍不优先回到 Dreamer Actor/Critic

即使世界模型成功，也不意味着：

> 下一步就是 Actor-Critic。

交易决策的问题和市场状态建模的问题必须继续分离。

第一版 Decision 可以非常简单：

```text
future distribution
+
risk constraints
↓
deterministic decision
```

甚至：

```text
如果 expected downside 太大
→ 不做

如果某类 future distribution 明显偏斜
→ candidate
```

只有以后有充分理由，才研究：

- policy learning
- offline RL
- decision optimization

而不是默认回到 Dreamer Actor/Critic。

---

# 8. 原始项目最终想得到的东西

最终理想系统不是：

```text
K线
↓
AI
↓
BUY
```

而是：

```text
                    Real World
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
     Market          Industry         Events
     Data             Data             News
        └───────────────┼───────────────┘
                        ↓
               Observation Encoders
                        ↓
                   JEPA Features
                        ↓
              RSSM Persistent Belief
                        ↓
               HiddenMarketState_t
                        ↓
             Future Latent Dynamics
                        ↓
              Probability Distribution
                        ↓
            ┌──────────────────────┐
            │   World Model Ends   │
            └──────────────────────┘
                        ↓
              Independent Decision
                        ↓
                 Risk / Position
```

这个结构才是最初所谓：

\[
\boxed{
JEPA + RSSM
}
\]

的完整意义。

---

# 9. V0 为什么仍然非常重要

虽然最终架构远比 V0 大，但 V0 是整个项目最关键的 Gate。

因为如果：

\[
History \rightarrow Z_t
\]

本身都不能得到：

```text
可预测
有邻域结构
比 raw features 更有信息
严格 OOS 可泛化
```

那么：

```text
RSSM
Multimodal
News
World Dynamics
Decision
```

都没有基础。

RSSM 不能凭空制造市场中不存在的稳定状态。

更多数据也可能只是增加更多噪声。

因此 V0 的角色是：

> **决定整个 JEPA + RSSM Market World Model 路线是否值得继续。**

---

# 10. 当前 V0 GO 之后的第一件事是什么

不是实时行情。

不是交易。

不是新闻。

不是多品种。

而是：

\[
\boxed{
FixedWindow\ JEPA
\rightarrow
Persistent\ JEPA+RSSM
}
\]

这才是原始路线的下一步。

Stage B 的问题非常具体：

> **既然 V0 已经证明 predictive latent state 存在，那么用持续 belief state 保存和更新它，是否比固定窗口每次重新编码更好？**

这个实验必须直接比较：

```text
V0 fixed-window JEPA
vs
JEPA + RSSM persistent state
```

如果 RSSM 不增加严格 OOS 信息：

> 不保留 RSSM。

如果明显增加：

> 才进入真正的 persistent Market World Model。

---

# 11. 当前项目状态

```text
最初世界模型方向
JEPA + RSSM
      │
      ↓
为了可证伪性拆出最小实验
      │
      ↓
V0 Fixed-window Market-JEPA
      │
      ↓
当前：正在正式训练 / Representation Evaluation
      │
      ├── NO-GO
      │      ↓
      │    停止并分析核心假设
      │
      └── GO
             ↓
       JEPA + RSSM
       Persistent HiddenMarketState
             ↓
       Latent Dynamics
             ↓
       Future Distribution
             ↓
       Multimodal Hidden World
             ↓
       Generalization
             ↓
       Decision Layer
```

---

# 12. 防跑偏判断规则

以后任何新想法先问：

## Q1：V0 GO 了吗？

没有：

> 不讨论 RSSM 实现、新闻、多品种、实时、交易。

---

## Q2：如果 V0 GO，下一主线是否仍然围绕 Persistent HiddenMarketState？

如果直接跳：

```text
V0
↓
实时交易
```

则明显偏离原始设计。

正确主线：

```text
V0
↓
JEPA + RSSM
↓
Persistent HiddenMarketState
```

---

## Q3：是否把 Trader Action 错误地放进 Market Dynamics？

如果出现：

\[
Market_{t+1}=f(Market_t,BUY/SELL)
\]

则偏离了我们最开始对交易世界的定义。

---

## Q4：是否把世界模型和 PnL 优化绑在一起？

如果 Hidden State 的学习重新被：

```text
Reward
PnL
Position
```

主导：

> 项目又回到了我们最初决定离开的 Dreamer-style trading RL。

---

## Q5：新增模块是否提高“世界理解”，还是只提高回测？

项目优先级永远是：

```text
State quality
↓
Future distribution quality
↓
Generalization
↓
最后才是 decision value
```

---

# 13. 给未来 ChatGPT / Codex 的固定 Review 指令

以后把本文和当前代码 / 设计一起提供，并要求：

> **以《Market World Model 项目总纲（原始设计主线版）》为最高层基准，先判断当前工作是否仍沿着“固定窗口 JEPA 验证 predictive latent → JEPA+RSSM persistent belief → latent dynamics / future distribution → multimodal hidden world → independent decision layer”的原始路线。特别检查是否错误地把 Dreamer 的 Actor/Critic/Reward/action-conditioned dynamics重新引入市场模型，或者是否在 V0 GO 之前提前跨阶段。**

输出：

```text
Original Goal Alignment:
Current Stage:
Required Previous Gate:
JEPA Role:
RSSM Role:
Action/Reward Contamination:
Premature Stage Jump:
Verdict:
- ALIGNED
- MINOR DRIFT
- MAJOR DRIFT
- RETURN TO ORIGINAL ROADMAP
```

---

# 14. 最重要的三句话

第一：

\[
\boxed{
V0 不加 RSSM，是为了隔离变量验证 latent state 是否存在；
不是因为 RSSM 不属于最终设计。
}
\]

第二：

\[
\boxed{
最终核心主干从一开始就是 JEPA + RSSM：
JEPA 学“什么状态值得表示”，RSSM 学“这个状态如何持续演化”。
}
\]

第三：

\[
\boxed{
Trader Action 不主导 Market Dynamics。
世界模型负责理解市场，
交易策略负责决定如何利用这个理解。
}
\]
