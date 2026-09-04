# Market-JEPA 理论发现记录：Internal Representation ≠ Predictive State

**文档用途：** 记录 Market-JEPA V0 首轮正式 Validation 后暴露出的关键理论问题，作为后续 JEPA + RSSM Market World Model 设计的重要依据。  
**状态：** 研究结论 / 设计修正依据，不等同于新版本实现方案。  
**背景实验：** Market-JEPA V0.6.1 训练 50 epochs，V0.6.2 Validation-only evaluation。  
**当前正式结论：** V0 按预注册 Gate 判定 `VALIDATION_NO_GO`，2025 Final Test 尚未使用。

---

# 1. 实验事实

首轮 Validation 得到：

```text
Prediction   PASS
Probe        PASS
kNN          FAIL

Overall      VALIDATION_NO_GO
```

核心现象：

- JEPA predictor 在 H64 上显著优于 Persistence。
- Frozen latent Ridge 明显优于当前 raw-feature Ridge baseline。
- 但 latent kNN 显著差于 random historical neighbors。
- 训练没有 full collapse，后期 latent effective rank 健康。
- 代码审核没有发现能解释 kNN FAIL 的实现 bug、泄漏或 bootstrap 方向错误。

因此，不能简单解释为：

> “模型什么都没学到。”

也不能解释为：

> “kNN 代码写错了。”

真正需要解释的是：

\[
\boxed{
Prediction\ PASS
+
Probe\ PASS
+
kNN\ FAIL
}
\]

为何能够同时成立。

---

# 2. 关键理论发现

当前 Market-JEPA 把：

\[
Z_t = Encoder(History_t)
\]

直接称为：

> HiddenMarketState

这是一个过强的定义。

更准确地说：

\[
Z_t
\]

首先只是一个：

> **Internal Representation / Internal Belief Coordinate**

它是网络为了完成 future latent prediction 而学习出的内部编码。

JEPA objective 要求的是：

\[
Predictor(Z_t)
\approx
TargetFutureRepresentation
\]

但是并没有直接要求：

\[
distance(Z_i,Z_j)
\]

必须对应：

\[
distance(
P(Future|H_i),
P(Future|H_j)
)
\]

因此：

\[
\boxed{
Predictive\ Representation
\neq
Predictive\ State\ Geometry
}
\]

这正是当前 V0 最重要的理论发现。

---

# 3. 为什么 Prediction PASS 与 kNN FAIL 可以同时成立

设：

\[
Z = E(H)
\]

Predictor：

\[
\hat Y=P(Z)
\]

JEPA 训练只关心：

\[
L(P(E(H)),Y)
\]

现在任意取一个可逆矩阵：

\[
A\in GL(d)
\]

定义新的 representation：

\[
Z'=AZ
\]

以及新的 predictor：

\[
P'(Z')=P(A^{-1}Z')
\]

则：

\[
P'(Z')
=
P(A^{-1}AZ)
=
P(Z)
\]

因此：

\[
\boxed{
L'=L
}
\]

也就是说：

> 只要 predictor 能吸收这个可逆变换，JEPA prediction objective 完全无法区分 \(Z\) 与 \(AZ\)。

但一般情况下：

\[
cosine(AZ_i,AZ_j)
\neq
cosine(Z_i,Z_j)
\]

所以：

```text
future prediction
完全不变

但是

latent nearest-neighbor geometry
可以发生巨大变化
```

因此 raw latent cosine kNN 并不是 JEPA objective 自然保证的性质。

---

# 4. 为什么 Linear Probe 也可以 PASS

假设某个 future outcome 可以由：

\[
y=w^\top Z
\]

线性提取。

经过：

\[
Z'=AZ
\]

只需要令：

\[
w'=A^{-T}w
\]

则：

\[
w'^\top Z'
=
w^\top Z
\]

所以：

\[
\boxed{
Linear\ Probe
}
\]

同样可以在 latent geometry 被严重扭曲的情况下继续保持良好性能。

因此以下组合在数学上完全可能：

```text
Prediction     PASS
Linear Probe   PASS
Cosine kNN     FAIL
```

当前 V0 的结果正好符合这种结构。

---

# 5. 对当前 V0 结果的重新解释

旧解释：

> Market-JEPA 没有学出 HiddenMarketState。

更准确的新解释：

> **Market-JEPA 学出了包含 predictive information 的 internal representation，但没有证明这个 representation 的自然 cosine geometry 等价于 predictive market-state geometry。**

也就是说：

\[
Z_t
\]

可能是有价值的内部坐标系统。

Predictor 可以利用它。

Linear Probe 也可以利用它。

但：

\[
distance(Z_i,Z_j)
\]

未必有我们原来赋予它的“市场状态相似度”意义。

---

# 6. 原始 kNN Gate 并不是没有价值

虽然 JEPA 本身不保证 latent cosine kNN 语义，但我们最初加入 kNN 并不是错误。

因为项目最终目标不是：

> “获得一个方便下游线性预测的 feature vector。”

而是：

> **获得一个真正有状态语义的 Market World Model。**

我们真正希望满足的是：

\[
\boxed{
H_i\sim H_j
\iff
P(Future|H_i)
\approx
P(Future|H_j)
}
\]

即：

> 两段历史属于“相同市场状态”，应当是因为它们隐含的未来概率分布相似。

因此 kNN Gate 的思想仍然应该保留。

需要修正的是：

> **不能未经证明地把 raw internal latent 的 cosine distance 当成 predictive-state distance。**

---

# 7. Predictive Equivalence：更严格的 HiddenMarketState 定义

以后应该把“相似 Market State”定义为：

\[
H_i\sim H_j
\]

当且仅当：

\[
P(Future|H_i)
\approx
P(Future|H_j)
\]

这是一种：

> **Predictive Equivalence**

理想的 Market State 应该表示 history 的未来预测等价类。

因此真正想得到的不是任意内部 latent：

\[
Z_t
\]

而是某种：

\[
S_t
\]

满足：

\[
\boxed{
S_t
\equiv
\mathcal E[
P(Future|History_t)
]
}
\]

其中 \(\mathcal E\) 是 future conditional distribution 的 representation / embedding。

最终希望：

\[
d(S_i,S_j)
\approx
D(
P(Future|H_i),
P(Future|H_j)
)
\]

---

# 8. Internal Belief 与 Predictive State 必须分离

今后的理论结构应该从：

```text
History
↓
Z_market
↓
HiddenMarketState
```

修改成：

```text
History / Observation
        ↓
Internal Belief Representation B_t
        ↓
Predictive State S_t
        ↓
Future Distribution
```

---

## 8.1 Internal Belief \(B_t\)

\[
B_t
\]

是模型内部保存历史信息的工作空间。

未来它可以由：

- Transformer
- GRU
- RSSM deterministic state
- RSSM stochastic state
- state-space model

实现。

它的职责是：

> 保存对未来有用的历史信息。

它的 raw cosine geometry **不必**具有直接市场语义。

---

## 8.2 Predictive State \(S_t\)

\[
S_t
\]

才应该表示：

\[
P(Future|History_t)
\]

或者足够丰富的 future conditional distribution embedding。

这里才要求：

> State distance 对应 future distribution difference。

因此：

\[
\boxed{
Internal\ Belief
\neq
Predictive\ State
}
\]

这是当前实验最重要的设计修正。

---

# 9. 对 JEPA 的重新定位

JEPA 仍然非常适合这个项目。

但 JEPA 的职责应该明确为：

> **让 internal belief 保留能够预测 future representation 的信息。**

即：

\[
B_t
\rightarrow
\hat Z_{future}
\]

JEPA 负责：

- representation learning
- predictive feature retention
- 避免直接重建大量 observation-level noise
- 用 future prediction pressure 约束历史表示

JEPA 不自动负责：

- latent metric semantics
- predictive-state identifiability
- cosine neighborhood meaning
- future-distribution distance calibration

因此：

\[
\boxed{
JEPA
\text{ 学 predictive information}
}
\]

但：

\[
\boxed{
JEPA
\text{ 不自动定义 predictive-state geometry}
}
\]

---

# 10. 对 RSSM 的重新定位

原始长期路线：

\[
\boxed{
JEPA + RSSM
}
\]

仍然成立。

但现在职责需要更精确。

RSSM 解决的是：

\[
B_t=f(B_{t-1},O_t)
\]

即：

> **persistent belief / memory / filtering**

RSSM 负责：

- 历史状态持续积累
- 新 observation 到来后的 belief update
- 避免固定 512-bar 人工记忆边界
- 维护长期动态上下文

RSSM 不应该被要求：

> raw latent cosine 自然等于 future similarity。

因此最终更合理的结构是：

```text
Observation O_t
        ↓
Observation Encoder
        ↓
RSSM / Recurrent Filter
        ↓
Internal Belief B_t
        │
        ├── JEPA Future Prediction
        │
        ↓
Predictive-State Head
        ↓
S_t
        ↓
Future Distribution
```

---

# 11. JEPA + RSSM + Predictive State

当前更完整的长期理论路线应是：

\[
\boxed{
JEPA
+
RSSM
+
Predictive\ State
}
\]

三者职责不同。

### JEPA

回答：

> 什么历史信息值得保留，因为它能预测未来？

### RSSM

回答：

> 这些历史信息如何随着 observation 持续更新？

### Predictive State

回答：

> 什么才应该被称为“当前市场状态”，以及 state similarity 应该如何定义？

这三者不能再混成一个 \(Z_t\)。

---

# 12. Predictive State Representation（PSR）的意义

PSR 理论提供了一个与当前目标高度一致的观点：

> 动态系统的 state 可以直接用“对未来 observable events 的预测”来表示。

即 state 不一定是一个不可解释 hidden variable，而可以是：

\[
State
=
Predictions\ about\ Future
\]

这和我们真正想要的 Market State 很接近。

因此后续值得系统研究：

- Predictive State Representation
- spectral / nonlinear PSR
- Hilbert-space PSR
- conditional distribution embedding
- future-distribution sufficient statistics

PSR 并不意味着直接替换 JEPA 或 RSSM。

更合理的方向是：

\[
\boxed{
JEPA
\text{ 学 representation}
+
RSSM
\text{ 维护 belief}
+
PSR\text{-style definition}
\text{ 约束 state semantics}
}
\]

---

# 13. 当前 Predictor Output 可能比 Z_market 更接近 Predictive State

当前模型已经存在：

\[
P_{16}(Z_t)
\]

\[
P_{64}(Z_t)
\]

\[
P_{256}(Z_t)
\]

这些输出是直接被训练去接近 future representation 的。

因此它们理论上比：

\[
Z_{market}
\]

更接近：

> **predictive coordinates**

一个重要诊断是：

### A. 已知

\[
kNN(Z_{market})
\]

FAIL。

### B. 下一步可测

\[
kNN(P_{64}(Z_{market}))
\]

### C. 以及

\[
S_t=
[
P_{16}(Z_t),
P_{64}(Z_t),
P_{256}(Z_t)
]
\]

然后测试：

\[
kNN(S_t)
\]

是否对应更好的 future outcome similarity。

---

# 14. 这个诊断为什么重要

如果出现：

```text
Z_market kNN              FAIL
P64(Z) kNN                PASS
Multi-horizon P(Z) kNN    PASS
```

则说明：

> JEPA predictive mechanism 本身可能已经工作。

真正错误的是：

> 我们把 predictor 之前的 internal representation 当成了 predictive state。

这时不需要推翻整个 JEPA 路线。

只需要修改：

```text
Internal Belief ≠ Market State
Predictive Representation = Market State candidate
```

---

如果：

```text
Z_market kNN              FAIL
P64(Z) kNN                FAIL
Multi-horizon P(Z) kNN    FAIL
```

则问题更深：

> 即使 predictor output space 也没有形成 predictive future geometry。

这时才应进入：

- target representation redesign
- predictive-state objective
- distribution embedding
- metric / geometry objective
- collapse / identifiability research

---

# 15. 当前不应该做的事情

这次发现以后，不应该马上：

- 调 K
- 换 cosine / Euclidean 并挑最好
- PCA 后再 kNN
- 删除 latent 维度
- 增加 metric loss 直到 kNN PASS
- 直接加入 SIGReg
- 直接加入 RSSM
- 重新训练 V0
- 打开 2025 Test
- 把 kNN Gate 删除，从而宣布 V0 GO

这些做法都会掩盖：

> **objective 与 state definition 是否一致**

这个真正的问题。

---

# 16. 对 LeJEPA / geometry regularization 的定位

后续可以研究更现代的 JEPA geometry / regularization 工作。

它们可能帮助：

- 防 collapse
- 提高 effective rank
- 改善 conditioning
- 改善 embedding marginal distribution
- 改善 kNN / kernel estimator 的数值性质

但必须区分：

\[
Good\ Marginal\ Geometry
\]

和：

\[
Correct\ Predictive\ Geometry
\]

即使：

\[
Z\sim N(0,I)
\]

也不能自动推出：

\[
distance(Z_i,Z_j)
\approx
D(P_i,P_j)
\]

因此 geometry regularization 不能替代 predictive-state definition。

---

# 17. 对当前 V0 的正式评价

V0 按原冻结判据仍然：

```text
VALIDATION_NO_GO
```

这个结果不能修改。

但是对失败原因的科学解释应该更新。

不应该写：

> “JEPA 没有学到 predictive representation。”

因为 Prediction 与 Probe 已经说明这个说法过强。

更准确的结论：

\[
\boxed{
Market\text{-}JEPA\ V0
学到了可被 predictor / probe 利用的 predictive information，
但没有证明 raw internal latent 的自然 metric geometry
等价于 predictive market-state geometry。
}
\]

因此 V0 暴露的是：

> **Representation-State Definition Mismatch**

而不是简单的：

> “模型没学会。”

---

# 18. 对整个项目路线的影响

原始长期方向：

\[
JEPA + RSSM
\]

不需要推翻。

但应该升级为更精确的：

\[
\boxed{
Observation
\rightarrow
Persistent\ Internal\ Belief
\rightarrow
Predictive\ State
\rightarrow
Future\ Distribution
}
\]

其中：

```text
JEPA
负责 predictive representation pressure

RSSM
负责 persistent belief update

Predictive-State layer
负责 state semantics / future-distribution equivalence
```

最终 Decision / Risk Layer 仍然完全独立。

---

# 19. 当前最有信息价值的下一步

在不重新训练、不开 Test 的条件下：

使用当前 epoch49 模型做 mechanism diagnostic：

```text
1. kNN(Z_market)
   已知 FAIL

2. kNN(P64(Z_market))

3. kNN(concat(P16(Z), P64(Z), P256(Z)))
```

这不是新的 GO / NO-GO。

它只回答：

> **Predictive geometry 到底出现在 encoder latent 之前、predictor output，还是根本没有形成？**

这个答案会直接决定下一版是：

### 路线 A

重新定义 State 的位置。

还是：

### 路线 B

重新设计 predictive-state objective。

---

# 20. 当前最重要的研究结论

第一：

\[
\boxed{
Internal\ Representation
\neq
Predictive\ State
}
\]

第二：

\[
\boxed{
Prediction\ ability
\neq
Correct\ latent\ geometry
}
\]

第三：

\[
\boxed{
Linear\ Probe\ success
\neq
Nearest\ Neighbor\ state\ semantics
}
\]

第四：

\[
\boxed{
如果最终目标是 HiddenMarketState，
state 必须由 future predictive equivalence 定义，
而不能仅由 encoder latent 的 cosine distance 定义。
}
\]

第五：

\[
\boxed{
JEPA + RSSM 的大方向仍然成立，
但需要补上 Predictive-State semantics 这一层。
}
\]

---

# 21. 项目长期主线（修正版）

```text
Market Observation
        ↓
Observation Encoder
        ↓
Persistent Belief / RSSM
        ↓
Internal Belief B_t
        │
        ├──────── JEPA predictive objective
        │
        ↓
Predictive-State Representation S_t
        ↓
Future Distribution
        ↓
Uncertainty / Dynamics
        ↓
Independent Decision Layer
        ↓
Risk / Position
```

在这个结构中：

> **B_t 是模型内部如何记忆世界。**

而：

> **S_t 才是我们真正要称为 Market State 的东西。**

这是 Market-JEPA V0 第一轮正式实验带来的最重要理论修正之一。
