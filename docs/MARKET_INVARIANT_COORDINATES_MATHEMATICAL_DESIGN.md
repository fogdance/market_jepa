# Market Invariant Coordinates (IMC)
## 跨品种 Price / Open Interest / Volume 等价坐标的数学定义与证明

**Version:** 0.1 — Reviewed Mathematical Baseline  
**Date:** 2026-09-05  
**Status:** Reviewed and ready for empirical audit; formulas are baseline-frozen unless later evidence requires revision  

---

## 0. 目标

我们希望为商品期货构造一个跨品种统一坐标系，使模型尽量学习：

\[
\boxed{\text{PricePath}\times\text{OIPath}\times\text{VolumePath}}
\]

的联合动态，而不是依赖某个具体品种的绝对价格、绝对持仓量或绝对成交量。

不同品种可能具有完全不同的数值尺度，例如：

- 焦煤价格约为 1000，而螺纹钢价格约为 3000；
- 焦煤 OI 为几十万手，而螺纹钢 OI 可达到更高数量级；
- 不同品种分钟成交量的绝对手数差异巨大。

这些绝对尺度不是本阶段希望模型优先利用的市场结构。我们希望主动消除它们，同时保留：

1. 价格完整相对路径；
2. 持仓量完整相对路径；
3. 持仓量持续增加、突然增加、持续减少、突然减少；
4. 成交量相对于窗口开始时正常水平的完整路径；
5. 当前相对于最近 20 根 K 线的“放量 N 倍 / 缩量 N 倍”；
6. Price、OI、Volume 三条路径之间的同步、背离、加速与持续关系。

本设计不使用人工的“多头进攻 / 空头进攻”等标签。模型只接收连续无量纲坐标，并自行学习其未来意义。

---

# 1. 数学问题

设一个 causal observation window 为：

\[
t=t_0,t_0+1,\ldots,t_T.
\]

对每个时刻定义：

\[
P_t>0,
\]

\[
OI_t>0,
\]

\[
V_t\ge 0.
\]

其中：

- \(P_t\)：价格；
- \(OI_t\)：Open Interest；
- \(V_t\)：成交量。

对于 OHLC，分别记为：

\[
O_t,H_t,L_t,C_t>0.
\]

不同品种之间允许存在三个彼此独立的正比例尺度：

\[
P'_t=aP_t,
\]

\[
OI'_t=bOI_t,
\]

\[
V'_t=cV_t,
\]

其中：

\[
a,b,c>0.
\]

我们希望构造变换 \(T\)，满足：

\[
\boxed{
T(aP,bOI,cV)=T(P,OI,V)
}
\]

即：**对三个绝对尺度严格不变**。

同时，我们希望这种不变性不是通过粗暴压缩实现，而是在数学上只 quotient 掉无关的乘法尺度，并尽可能保留完整运动路径。

---

# 2. 等价关系与群作用

定义正实数乘法群：

\[
G=(\mathbb R_{>0})^3.
\]

一个元素：

\[
g=(a,b,c)\in G
\]

作用于市场路径：

\[
g\cdot(P,OI,V)
=
(aP,bOI,cV).
\]

我们把以下两组市场路径视为“尺度等价”：

\[
(P',OI',V')\sim(P,OI,V)
\]

当且仅当存在：

\[
a,b,c>0
\]

使得：

\[
P'=aP,
\quad
OI'=bOI,
\quad
V'=cV.
\]

本设计的目标就是构造这个等价关系下的无量纲坐标。

---

# 3. Price 固定原点坐标

## 3.1 Close 路径

选择窗口开始价格：

\[
P_0=C_{t_0}.
\]

定义：

\[
\boxed{
X_t^P
=
\log\frac{C_t}{P_0}
}
\]

因此：

\[
X_{t_0}^P=0.
\]

---

## 3.2 Price 尺度不变性定理

若：

\[
C'_t=aC_t,
\quad a>0,
\]

则：

\[
X_t^{P'}
=
\log\frac{aC_t}{aP_0}
=
\log\frac{C_t}{P_0}
=
X_t^P.
\]

所以：

\[
\boxed{
X^{P'}=X^P
}
\]

严格成立。

因此 Price coordinate 对任意绝对价格比例缩放完全不敏感。

---

# 4. Price 的信息保存性

由定义：

\[
X_t^P
=
\log\frac{C_t}{P_0},
\]

得到：

\[
\boxed{
C_t=P_0e^{X_t^P}
}
\]

因此如果已知 \(P_0\)，原始 Close path 可以精确恢复。

如果故意不保存 \(P_0\)，唯一被删除的是一个全局乘法尺度。

更严格地：

\[
\boxed{
X^P(C)=X^P(\tilde C)
\iff
\exists a>0:\ \tilde C_t=aC_t,\ \forall t
}
\]

所以 \(X^P\) 是 Price path 在正比例缩放作用下的 maximal invariant / injective-modulo-scale 表示。

---

# 5. OHLC 完整路径

只保留 Close 会损失单根 K 线内部形状，因此对所有 OHLC 使用同一个固定价格原点 \(P_0=C_{t_0}\)：

\[
\boxed{
X_t^O=\log\frac{O_t}{P_0}
}
\]

\[
\boxed{
X_t^H=\log\frac{H_t}{P_0}
}
\]

\[
\boxed{
X_t^L=\log\frac{L_t}{P_0}
}
\]

\[
\boxed{
X_t^C=\log\frac{C_t}{P_0}
}
\]

若 OHLC 整体乘以 \(a>0\)，上述四个坐标全部严格不变。

若给定 \(P_0\)，则：

\[
O_t=P_0e^{X_t^O},
\]

\[
H_t=P_0e^{X_t^H},
\]

\[
L_t=P_0e^{X_t^L},
\]

\[
C_t=P_0e^{X_t^C}.
\]

因此完整 OHLC trajectory 除绝对价格尺度外无损。

---

# 6. 单步涨跌、突然上涨与累计走势都被保留

相邻差：

\[
X_t^C-X_{t-1}^C
=
\log\frac{C_t}{C_{t-1}}.
\]

因此：

\[
\boxed{
\Delta X_t^P
=
\log\frac{C_t}{C_{t-1}}
}
\]

即标准 log return 可以从固定原点 Price path 精确恢复。

任意区间 \([t-k,t]\)：

\[
X_t^P-X_{t-k}^P
=
\log\frac{C_t}{C_{t-k}}.
\]

因此：

- 单根突然上涨/下跌；
- 连续上涨/下跌；
- 回撤；
- 加速；
- 任意窗口累计收益；

全部保存在同一条坐标路径中。

---

# 7. Open Interest 固定原点坐标

用户确定的核心定义：**同一 observation period 的第一根 K 线 OI 作为基准。**

令：

\[
OI_0=OI_{t_0}.
\]

定义：

\[
\boxed{
X_t^{OI}
=
\log\frac{OI_t}{OI_0}
}
\]

因此：

\[
X_{t_0}^{OI}=0.
\]

---

# 8. OI 尺度不变性

若：

\[
OI'_t=bOI_t,
\quad b>0,
\]

则：

\[
X_t^{OI'}
=
\log\frac{bOI_t}{bOI_0}
=
X_t^{OI}.
\]

所以：

\[
\boxed{
X^{OI'}=X^{OI}
}
\]

严格成立。

焦煤 30 万手与螺纹 150 万手，只要相对持仓运动路径相同，转换后得到完全相同的 OI trajectory。

---

# 9. OI 路径的信息保存性

由：

\[
X_t^{OI}=\log\frac{OI_t}{OI_0}
\]

可精确反解：

\[
\boxed{
OI_t=OI_0e^{X_t^{OI}}
}
\]

因此如果知道 \(OI_0\)，整个原始 OI path 可精确恢复。

若不保存 \(OI_0\)，唯一删除的是绝对持仓量乘法尺度。

并且：

\[
\boxed{
X^{OI}(A)=X^{OI}(B)
\iff
\exists b>0:\ B_t=bA_t,\ \forall t
}
\]

所以 OI coordinate 同样是 injective modulo scale。

---

# 10. 持续增仓严格保留

定义一阶差分：

\[
\Delta X_t^{OI}
=
X_t^{OI}-X_{t-1}^{OI}.
\]

则：

\[
\boxed{
\Delta X_t^{OI}
=
\log\frac{OI_t}{OI_{t-1}}
}
\]

因此：

\[
OI_t>OI_{t-1}
\iff
\Delta X_t^{OI}>0.
\]

若连续 \(k\) 根：

\[
\Delta X_{t-k+1}^{OI}>0,
\ldots,
\Delta X_t^{OI}>0,
\]

则原始 OI 同样连续递增。

所以“持仓量一直增加”没有因为坐标转换而丢失。

---

# 11. 突然增仓严格保留

单步 OI 相对跳变：

\[
\frac{OI_t}{OI_{t-1}}
\]

在新坐标中精确等价为：

\[
\boxed{
\Delta X_t^{OI}
=
\log\frac{OI_t}{OI_{t-1}}
}
\]

因为 log 在正数域严格单调，所以：

\[
\frac{OI_t}{OI_{t-1}}
>
\frac{OI_s}{OI_{s-1}}
\iff
\Delta X_t^{OI}
>
\Delta X_s^{OI}.
\]

因此单步 OI 增幅的排序被严格保留。

换言之：

\[
\boxed{
\text{Top OI jump ordering is preserved exactly.}
}
\]

“突然增仓”不会被 rolling average 或 z-score 平滑掉。

---

# 12. 任意时间尺度累计 OI 变化严格保留

任意 \(k\)：

\[
X_t^{OI}-X_{t-k}^{OI}
=
\log\frac{OI_t}{OI_{t-k}}.
\]

所以：

\[
\boxed{
D_{t,k}^{OI}
=
\log\frac{OI_t}{OI_{t-k}}
}
\]

无需额外原始信息即可得到。

因此同一条 OI trajectory 可以表达：

- 几分钟累计增仓；
- 数十分钟累计增仓；
- 数小时累计增仓；
- 整个 observation period 累计增仓；
- 增仓后撤退；
- 减仓后重新扩张。

---

# 13. Volume：两个互补坐标

Volume 与 Price/OI 不同。

Price 与 OI 是状态 / 存量型序列；成交量是每根 K 线的 activity flow。

单独使用 rolling ratio：

\[
\frac{V_t}{Median_{20}}
\]

虽然尺度不变，但不能保证完整原始 Volume path modulo scale 可恢复。

因此 Volume 使用两个互补坐标：

1. **Fixed-origin Volume coordinate**：负责保存整个窗口的相对成交量路径；
2. **Rolling Median20 surprise coordinate**：负责精确保留“当前放量 N 倍”。

---

# 14. Fixed-origin Volume baseline

在 observation window 开始之前，使用前 20 根 K 线：

\[
V_{t_0-20},\ldots,V_{t_0-1}.
\]

定义固定 baseline：

\[
\boxed{
M_0^V
=
\operatorname{median}
(V_{t_0-20},\ldots,V_{t_0-1})
}
\]

要求：

\[
M_0^V>0.
\]

然后窗口内全部成交量使用同一个 denominator：

\[
\boxed{
X_t^V
=
\frac{V_t}{M_0^V}
}
\]

这是窗口内部的完整相对成交量 trajectory。

---

# 15. Fixed-origin Volume 的尺度不变性

若：

\[
V'_t=cV_t,
\quad c>0,
\]

则中位数的正齐次性给出：

\[
M_0^{V'}=cM_0^V.
\]

于是：

\[
X_t^{V'}
=
\frac{cV_t}{cM_0^V}
=
X_t^V.
\]

因此：

\[
\boxed{
X^{V'}=X^V
}
\]

严格成立。

---

# 16. Fixed-origin Volume 的信息保存性

给定 baseline \(M_0^V\)：

\[
\boxed{
V_t=M_0^VX_t^V
}
\]

因此完整窗口成交量 path 可以精确恢复。

若不保存 \(M_0^V\)，唯一有意删除的是该品种的绝对成交量尺度。

所以：

\[
\boxed{
X^V
\text{ preserves the entire in-window Volume trajectory modulo one scale.}
}
\]

---

# 17. Rolling Median20 Volume surprise

用户确定的局部成交量基准：**当前 K 线之前 20 根 K 线成交量的中位数。**

定义：

\[
\boxed{
M_t^{V,20}
=
\operatorname{median}
(V_{t-20},\ldots,V_{t-1})
}
\]

在：

\[
M_t^{V,20}>0
\]

时定义：

\[
\boxed{
Q_t^V
=
\frac{V_t}{M_t^{V,20}}
}
\]

这个坐标直接回答：

> 当前成交量是最近 20 根局部正常水平的多少倍？

---

# 18. “放量 N 倍”严格保存

若：

\[
V_t=N\,M_t^{V,20},
\]

则：

\[
\boxed{
Q_t^V=N
}
\]

因此：

- 0.5 倍量 \(\rightarrow 0.5\)；
- 正常量 \(\rightarrow 1\)；
- 2 倍放量 \(\rightarrow 2\)；
- 3 倍放量 \(\rightarrow 3\)；
- 10 倍放量 \(\rightarrow 10\)。

这是精确等价，不是近似分类。

---

# 19. Rolling Volume surprise 的尺度不变性

若：

\[
V'_t=cV_t,
\]

则：

\[
M_t^{V',20}
=
cM_t^{V,20}.
\]

因此：

\[
Q_t^{V'}
=
\frac{cV_t}{cM_t^{V,20}}
=
Q_t^V.
\]

所以：

\[
\boxed{
Q^{V'}=Q^V
}
\]

严格成立。

这意味着不同品种的“3 倍放量”经过转换后严格得到相同数值 3。

---

# 20. 为什么 Fixed-origin Volume 与 Rolling surprise 必须同时存在

假设成交量从正常水平长期提高到 4 倍。

在刚开始放量时：

\[
Q_t^V\gg1.
\]

随着高成交持续超过 20 根，rolling median 自身也会上升，之后可能出现：

\[
Q_t^V\approx1.
\]

这不表示市场重新回到旧的低成交水平，而只表示：

> 当前成交量相对于“最新 20 根局部状态”已经不再继续异常扩张。

此时固定基准坐标仍保持：

\[
X_t^V\approx4.
\]

因此：

- \(Q_t^V\)：局部 surprise / acceleration；
- \(X_t^V\)：相对于窗口开始前正常水平的持续 activity level。

两者互补。

---

# 21. 可选数值压缩，但不作为数学定义本身

对于网络数值稳定性，可以对非负 Volume ratio 使用严格单调可逆变换，例如：

\[
\boxed{
\phi(x)=\log(1+x)
}
\]

因此可输入：

\[
\tilde X_t^V=\log(1+X_t^V),
\]

\[
\tilde Q_t^V=\log(1+Q_t^V).
\]

因为 \(\phi\) 严格单调且可逆：

\[
\boxed{
x=e^{\phi(x)}-1}
\]

所以这种数值压缩不会丢失 ratio 信息或排序信息。

如果希望“正常量 \(Q=1\)”映射到 0，可使用：

\[
\boxed{
\tilde Q_t^V
=
\log(1+Q_t^V)-\log 2
}
\]

其反函数：

\[
\boxed{
Q_t^V=2e^{\tilde Q_t^V}-1
}
\]

同样严格可逆。

---

# 22. 完整 IMC 定义

定义 Market Invariant Coordinates：

\[
\boxed{
T(P,OI,V)
=
\left(
X^{OHLC},
X^{OI},
X^V,
Q^V
\right)
}
\]

其中：

### Price / OHLC

\[
X_t^O=\log\frac{O_t}{P_0},
\]

\[
X_t^H=\log\frac{H_t}{P_0},
\]

\[
X_t^L=\log\frac{L_t}{P_0},
\]

\[
X_t^C=\log\frac{C_t}{P_0}.
\]

### OI

\[
X_t^{OI}=\log\frac{OI_t}{OI_0}.
\]

### Volume fixed-origin path

\[
X_t^V=\frac{V_t}{M_0^V},
\]

其中：

\[
M_0^V
=
Median(V_{t_0-20:t_0-1}).
\]

### Volume rolling surprise

\[
Q_t^V
=
\frac{V_t}
{Median(V_{t-20:t-1})}.
\]

---

# 23. IMC 联合尺度不变性定理

给定：

\[
\tilde O_t=aO_t,
\]

\[
\tilde H_t=aH_t,
\]

\[
\tilde L_t=aL_t,
\]

\[
\tilde C_t=aC_t,
\]

\[
\widetilde{OI}_t=bOI_t,
\]

\[
\tilde V_t=cV_t,
\]

其中：

\[
a,b,c>0.
\]

则：

\[
\boxed{
T(aP,bOI,cV)=T(P,OI,V)
}
\]

严格成立。

### Proof

Price/OHLC：

\[
\log\frac{aP_t}{aP_0}
=
\log\frac{P_t}{P_0}.
\]

OI：

\[
\log\frac{bOI_t}{bOI_0}
=
\log\frac{OI_t}{OI_0}.
\]

Volume fixed baseline：

\[
Median(cV)=cMedian(V),
\]

因此：

\[
\frac{cV_t}{cM_0^V}
=
\frac{V_t}{M_0^V}.
\]

Volume rolling baseline 同理：

\[
\frac{cV_t}
{cMedian(V_{t-20:t-1})}
=
\frac{V_t}
{Median(V_{t-20:t-1})}.
\]

QED.

---

# 24. IMC 的信息边界

IMC 主动删除：

\[
\boxed{
\text{absolute Price scale}
}
\]

\[
\boxed{
\text{absolute OI scale}
}
\]

\[
\boxed{
\text{absolute Volume scale}
}
\]

这些被视为跨品种 nuisance coordinates。

但是 IMC 保留：

### Price

- 完整 OHLC 相对轨迹；
- 任意单步涨跌；
- 任意累计收益；
- 回撤、趋势、加速等从轨迹可推导结构。

### OI

- 完整相对 OI path；
- 单步 OI 变化；
- 突然增仓 / 减仓；
- 持续增仓 / 减仓；
- 任意时间尺度累计 OI 变化。

### Volume

- 相对于窗口开始前正常水平的完整成交量 path；
- 当前相对于前 20 根局部正常水平的精确 N 倍放量；
- 长期高活跃状态；
- 局部突然放量 / 缩量；
- 放量后进入新的高活跃平台。

---

# 25. 联合动态关系的保存

IMC 不输入任何人工市场行为标签。

如果某段原始市场表现为：

\[
Price\uparrow,
\]

\[
OI\uparrow,
\]

\[
Volume\text{ expands},
\]

那么经过 IMC 后表现为：

\[
X^P\text{ upward trajectory},
\]

\[
X^{OI}\text{ upward trajectory},
\]

\[
X^V>1
\]

以及在突然放量阶段：

\[
Q^V\gg1.
\]

如果 Price 与 OI 同步变化、背离、加速或反转，这些同步时序关系仍然存在于 transformed trajectories 中。

因此模型真正需要学习的是：

\[
\boxed{
f(
X^P_{t_0:t},
X^{OI}_{t_0:t},
X^V_{t_0:t},
Q^V_{t_0:t}
)}
\]

而不是某个具体品种的绝对数值。

---

# 26. 对 observation-window 尺度不变量的充分性说明

设某个我们真正关心的市场结构函数：

\[
F(P_{t_0:t_T},OI_{t_0:t_T},V_{t_0:t_T})
\]

只依赖 **当前 observation window 内的相对轨迹**，并且本身满足独立尺度不变性：

\[
F(aP,bOI,cV)=F(P,OI,V).
\]

对 Price 与 OI，fixed-origin coordinates 完整确定其正比例缩放等价类；对当前窗口 Volume，\(X^V=V/M_0^V\) 同样完整确定 in-window Volume path 的正比例缩放等价类。因此，对这类只依赖 observation-window 相对路径的尺度不变量，可以写成某个 IMC 坐标函数：

\[
\boxed{
F(P,OI,V)
=
\tilde F(X^{OHLC},X^{OI},X^V)
}
\]

rolling surprise \(Q^V\) 是额外的、由真实历史局部 baseline 定义的尺度不变量，用于显式暴露“当前相对最近 20 根的放量倍数”。

这里不声称 IMC 保留 **window 开始之前完整 Volume 历史的所有尺度不变量**；pre-window history 在 v0.1 中只通过 \(M_0^V\) 与后续 rolling Median20 参与坐标构造。

因此本设计能够严格支持的结论是：

> 如果某种可迁移的 Price/OI/Volume 规律只依赖当前 observation window 内的相对运动，以及本文显式保留的 rolling Volume surprise，而不依赖绝对单位或被丢弃的更早历史细节，则 IMC 不会因为删除绝对尺度而删除这种规律。

---

# 27. 时间周期与多尺度

“同一时间周期第一根 K 线作为基准”中的周期，应理解为**模型当前 observation sequence**，而不是固定自然日。

例如：

- Minute branch：当前 minute observation window 第一根；
- Daily branch：当前 daily observation sequence 第一根；
- Weekly branch：当前 weekly observation sequence 第一根。

因此同一个数学定义可以跨 minute / daily / weekly 使用。

对于任意两个位置 \(s<t\)，模型均可从固定原点坐标得到：

\[
X_t^P-X_s^P
=
\log\frac{P_t}{P_s},
\]

\[
X_t^{OI}-X_s^{OI}
=
\log\frac{OI_t}{OI_s}.
\]

所以固定原点不会限制模型只能理解“从窗口第一根到当前”的变化；窗口内部任意子区间变化仍可精确恢复。

---

# 28. Causality

所有变换必须是 causal。

Price baseline：

\[
P_0=C_{t_0}
\]

只来自 observation window 起点。

OI baseline：

\[
OI_0=OI_{t_0}
\]

只来自 observation window 起点。

Volume fixed baseline：

\[
M_0^V=Median(V_{t_0-20:t_0-1})
\]

只使用 window start 之前的 20 根。

Rolling surprise：

\[
M_t^{V,20}=Median(V_{t-20:t-1})
\]

严格不包含当前 \(V_t\) 和未来 Volume。

因此本设计不存在 future leakage。

---

# 29. 零值与非法 baseline

这是实现时必须显式处理的数学边界。

## 29.1 Price

要求：

\[
O_t,H_t,L_t,C_t>0.
\]

否则 log coordinate 未定义。

商品期货有效价格正常应满足这一条件。

## 29.2 OI

要求：

\[
OI_t>0.
\]

如存在 \(OI=0\) 或缺失，则该位置不能直接使用 log-ratio，应 mask / exclude / 按数据规则处理，不能偷偷添加任意常数。

## 29.3 Volume

允许：

\[
V_t=0.
\]

因为 ratio numerator 可以为 0。

但固定和 rolling median denominator 必须：

\[
M_0^V>0,
\]

\[
M_t^{V,20}>0.
\]

若 median baseline 为 0，则该 ratio 在数学上未定义。

第一版实现应：

- 显式标记 invalid baseline；
- mask / exclude 对应 coordinate；
- 不使用固定 \(\epsilon\) 强行替代 denominator。

原因是常数 \(\epsilon\) 会破坏严格尺度不变性：

\[
\frac{cV}{cM+\epsilon}
\ne
\frac{V}{M+\epsilon}.
\]

---

# 30. 连续主力换月边界

OI 与 Volume 的比例坐标只有在前后数据属于具有可比口径的连续序列时才有市场含义。

如果连续主力换月产生：

\[
OI_{old}\rightarrow OI_{new}
\]

的大幅机械跳变，则：

\[
\log\frac{OI_{new}}{OI_{old}}
\]

不能解释为真实资金突然减仓/增仓。

因此若存在 contract roll metadata，必须满足至少一种处理：

1. 在 roll boundary 切断 observation segment；
2. roll 后重新建立 Price / OI / Volume baseline；
3. mask 跨 roll 的单步 change；
4. 禁止需要连续市场语义的 path 跨越未处理的 roll boundary。

这是数据语义问题，不是 IMC 数学公式本身的问题。

---

# 31. 为什么不使用 per-commodity z-score 作为核心坐标

普通标准化：

\[
z_t=\frac{x_t-\mu}{\sigma}
\]

主要解决 numerical scale，但没有直接对应：

- “上涨 1%”；
- “OI 累计增加 10%”；
- “成交量是最近 20 根中位数的 3 倍”。

而且 per-commodity \(\mu,\sigma\) 可能将整个历史分布压缩到一个统计尺度，并引入对 fit period 的依赖。

IMC 的目标不同：

\[
\boxed{
Raw Market Coordinates
\rightarrow
Dimensionless Relative Market Coordinates
}
\]

它首先解决的是**语义等价性和跨品种尺度不变性**，而不是普通数值标准化。

后续如果神经网络仍需 shared numerical scaler，可以在 IMC 之后再仅用训练品种 fit 一个 shared scaler；这属于第二层工程处理，不改变 IMC 数学定义。

---

# 32. 第一版冻结公式

## Price / OHLC

\[
\boxed{
X_t^{OHLC}
=
\left[
\log\frac{O_t}{P_0},
\log\frac{H_t}{P_0},
\log\frac{L_t}{P_0},
\log\frac{C_t}{P_0}
\right]
}
\]

其中：

\[
P_0=C_{t_0}.
\]

## Open Interest

\[
\boxed{
X_t^{OI}
=
\log\frac{OI_t}{OI_0}
}
\]

其中：

\[
OI_0=OI_{t_0}.
\]

## Volume — fixed-origin path

\[
\boxed{
X_t^V
=
\frac{V_t}{M_0^V}
}
\]

其中：

\[
\boxed{
M_0^V
=
Median(V_{t_0-20:t_0-1})
}
\]

## Volume — rolling 20-bar surprise

\[
\boxed{
Q_t^V
=
\frac{V_t}
{Median(V_{t-20:t-1})}
}
\]

以上四组坐标构成 IMC v0.1 的核心。

---

# 33. 需要保留的数学冗余与可导出量

以下量不增加理论信息，但可以作为模型显式输入以降低学习难度：

### Price one-step return

\[
\Delta X_t^P
=
X_t^P-X_{t-1}^P.
\]

### OI one-step change

\[
\Delta X_t^{OI}
=
X_t^{OI}-X_{t-1}^{OI}.
\]

### 任意预定义窗口累计变化

\[
D_{t,k}^{P}=X_t^P-X_{t-k}^P,
\]

\[
D_{t,k}^{OI}=X_t^{OI}-X_{t-k}^{OI}.
\]

这些属于 derived coordinates，不是新的市场信息。

第一版模型是否显式加入，应通过独立 ablation 决定，而不是在数学定义阶段直接假设越多越好。

---

# 34. Empirical Audit 必须验证什么

数学证明只说明 IMC 对理想的正比例尺度变换成立。

真实多品种数据还需要独立验证：

## A. Synthetic scale invariance

对同一真实品种人为构造：

\[
P' = 3P,
\quad
OI'=10OI,
\quad
V'=100V.
\]

要求：

\[
\max|T(P',OI',V')-T(P,OI,V)|
\]

在浮点误差范围内为 0。

## B. Relative path reconstruction

验证：

\[
\frac{P_t}{P_0}=e^{X_t^P},
\]

\[
\frac{OI_t}{OI_0}=e^{X_t^{OI}},
\]

\[
\frac{V_t}{M_0^V}=X_t^V.
\]

## C. Volume N-times preservation

验证：

\[
Q_t^V
=
V_t/Median_{20}.
\]

特别检查 top volume-surprise events。

## D. OI jump ordering

原始：

\[
\log(OI_t/OI_{t-1})
\]

与：

\[
\Delta X_t^{OI}
\]

应逐点完全一致。

## E. Persistent OI change

验证任意 \(k\)：

\[
X_t^{OI}-X_{t-k}^{OI}
=
\log(OI_t/OI_{t-k}).
\]

## F. Cross-commodity distribution comparison

比较 raw coordinates 与 IMC coordinates 的跨品种分布差异。

目标不是让所有品种分布完全相同，而是：

\[
\boxed{
\text{减少由绝对单位/数量级造成的差异，保留真实行为差异。}
}
\]

---

# 35. 本文档不声称什么

IMC v0.1 的数学证明只建立：

1. 对独立正比例尺度变换的严格不变性；
2. Price/OI/fixed-volume path 在删除一个绝对尺度后的可恢复性；
3. rolling Volume surprise 对“放量 N 倍”的精确保留；
4. 连续 Price/OI/Volume 时序关系在 relative coordinate 中仍然存在。

它**不证明**：

- 不同品种确实共享相同预测规律；
- Price×OI×Volume 一定能够预测未来；
- 所有跨品种 nuisance factors 都只是乘法尺度；
- 合约换月、最小变动价位、交易单位、涨跌停、保证金制度等全部被消除；
- IMC 已经是唯一或最优的市场坐标系。

这些必须由后续多品种 empirical audit 和模型实验检验。

---

# 36. Review 结论

在正式留底前，对当前公式做了以下数学 review。

## 36.1 通过项

### Price

- fixed-origin log coordinate 尺度不变：**PASS**；
- OHLC modulo scale 可恢复：**PASS**；
- 单步与任意累计变化可从 path 精确得到：**PASS**。

### OI

- 第一根 OI 为 baseline 的 log-ratio 尺度不变：**PASS**；
- modulo absolute OI scale 可恢复：**PASS**；
- 持续增仓 / 突然增仓保留：**PASS**；
- OI jump 排序严格保留：**PASS**。

### Volume

- fixed pre-window Median20 coordinate 尺度不变：**PASS**；
- in-window Volume path modulo fixed scale 可恢复：**PASS**；
- rolling previous-20 Median surprise 尺度不变：**PASS**；
- “放量 N 倍”数值精确保留：**PASS**。

### Joint

- 对独立 \((a,b,c)\) 正比例缩放联合不变：**PASS**；
- 同步/背离等相对时序结构未被人工离散化：**PASS**。

## 36.2 必须保留的限制

1. **Rolling Volume ratio 单独不是 lossless representation。**  
   因此必须同时保留 fixed-origin Volume path。

2. **Volume baseline=0 时 ratio 未定义。**  
   第一版实现必须 mask，而不是随意加固定 epsilon。

3. **OI=0 时 log-ratio 未定义。**  
   必须明确数据处理规则。

4. **连续主力 roll 会制造伪 OI / Volume jump。**  
   必须在数据层显式识别或切断。

5. **IMC 只证明 scale invariance，不证明跨品种 predictive invariance。**  
   后者必须由 FG/SA/JM/SH/SP → RB 等 held-out commodity 实验验证。

---

# 36.3 数值 sanity check

在正式 handoff 前，另做了一次独立合成数值检查：

- 对随机正值 OHLC path 乘以 \(a=3.7\)；
- 对随机正值 OI path 乘以 \(b=11.2\)；
- 对随机 Volume path 及其 pre-window history 同时乘以 \(c=53\)。

重新计算 IMC 后，原坐标与缩放后坐标的最大绝对差为：

\[
4.44\times10^{-16}
\]

属于 IEEE-754 双精度浮点舍入量级。

同时检查：

\[
e^{X_t^C}=C_t/C_0,
\]

\[
e^{X_t^{OI}}=OI_t/OI_0,
\]

\[
X_t^V=V_t/M_0^V
\]

在该检查中重建误差为 0（双精度表示下）。

该 sanity check 不是数学证明的替代品，只用于确认公式实现方向不存在明显代数错误。

---

# 37. 当前冻结结论

IMC v0.1 可以概括为：

\[
\boxed{
\text{Price}
\rightarrow
\log(OHLC/P_0)
}
\]

\[
\boxed{
\text{OI}
\rightarrow
\log(OI/OI_0)
}
\]

\[
\boxed{
\text{Volume persistent path}
\rightarrow
V/M_0^V
}
\]

\[
\boxed{
\text{Volume local surprise}
\rightarrow
V/Median_{20}(V_{past})
}
\]

其中：

- \(P_0\)：当前 observation period 第一根 K 线的价格基准；
- \(OI_0\)：当前 observation period 第一根 K 线 OI；
- \(M_0^V\)：observation period 开始之前 20 根 K 线成交量中位数；
- rolling Median20：当前 K 线之前 20 根 K 线成交量中位数。

最终数学目标是：

\[
\boxed{
\text{Remove commodity scale, preserve market motion.}
}
\]

即：

> 删除“这个品种绝对有多大”，保留“这个市场是怎么运动的”。

---

## Next Step

下一步不是立即训练模型，而是使用 FG / SA / JM / SH / SP / RB 真实数据进行：

\[
\boxed{
\text{Cross-Commodity IMC Empirical Equivalence Audit}
}
\]

验证理论尺度不变性在真实数据处理实现中是否成立，并检查关键事件（放量、突然增仓、持续增仓）是否被完整保留。
