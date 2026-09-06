# CRITICAL V1.1 ADDENDUM
# Causal Partial Daily / Weekly Bars

补充一个必须遵守的 V1.1 数据规则。

预生成 Daily / Weekly bars 只能作为 CLOSED BAR CACHE。

任何历史训练 anchor 或实时 anchor，
当前正在形成的 Daily / Weekly bar
都必须根据该 anchor 时刻之前已经发生的 minute data 动态构造。

严禁直接读取该交易日 / 该交易周最终 OHLCV/OI。

============================================================
1. Historical replay == Live semantics
============================================================

对于任意 minute anchor t：

模型输入必须等价于：

    “如果真实交易系统运行到 t，
     当时能够看到什么？”

因此：

    historical replay
    live inference

必须共享同一套 as-of aggregation semantics。

禁止：

    使用 hindsight final Daily
    使用 hindsight final Weekly

============================================================
2. Current Daily
============================================================

Daily sequence：

    completed Daily bars
    +
    ONE current partial Daily bar

当前 partial Daily 必须由：

    current real contract minute bars
    with timestamp <= anchor

生成。

定义：

    open =
        first valid minute open
        of current trading_date

    high =
        max minute high up to anchor

    low =
        min minute low up to anchor

    close =
        latest minute close at anchor

    volume =
        accumulated minute volume up to anchor

    open_interest =
        latest observable OI at anchor

如果 minute volume 字段不是 per-bar volume，
先 audit 数据源语义，
不要盲目 sum。

============================================================
3. Current Weekly
============================================================

Current Weekly sequence：

    completed current-contract Weekly bars
    +
    ONE current partial Weekly bar

partial Weekly 只能使用：

    current-contract observations
    already available up to anchor

定义：

    open =
        first valid open of current lifecycle week

    high =
        max high through anchor

    low =
        min low through anchor

    close =
        latest close at anchor

    volume =
        accumulated volume through anchor

    open_interest =
        latest observable OI at anchor

禁止读取当前周最终 Friday/week-end values。

============================================================
4. Real contract lifecycle boundary
============================================================

Current Daily / Weekly 仍然遵守 V1.1：

    current real contract only

如果 main lifecycle 在 calendar week 中途开始：

    Current Weekly lifecycle begins at main_start

不要把 main_start 之前其他 contract / pre-lifecycle 数据
偷偷合入 current-contract lifecycle Weekly bar。

============================================================
5. trading_date semantics
============================================================

Daily grouping 必须优先使用：

    trading_date

不是：

    datetime.date()

夜盘属于数据定义的下一交易日时，
必须进入下一 trading_date 的 partial Daily。

Weekly grouping也基于真实 trading_date calendar。

============================================================
6. Partial-bar flags
============================================================

Daily / Weekly token context 增加：

    bar_is_partial

completed:

    0

currently forming:

    1

建议同时提供 causal progress context：

    elapsed_minutes_in_trading_day

以及 Weekly：

    elapsed_trading_days_in_week

不要使用任何需要知道未来实际成交结果的信息。

============================================================
7. Daily IMC origin correction
============================================================

禁止使用 main-start day 的 FINAL Daily OI
作为生命周期 origin。

Daily lifecycle origin 必须由
main_start 时已经可以观察到的信息定义。

冻结：

    P0_daily =
        first valid minute Open
        at contract main-start lifecycle

    OI0_daily =
        first valid observable minute OI
        at contract main-start lifecycle

后续所有 Daily partial / completed states：

    Price_IMC =
        log(P_daily_asof / P0_daily)

    OI_IMC =
        log(OI_daily_asof / OI0_daily)

因此 origin 在整个真实 contract lifecycle 内固定，
同时不存在 intra-day future leakage。

============================================================
8. Current Weekly IMC origin
============================================================

Current Weekly 使用同样原则。

生命周期 origin：

    P0_weekly =
        first observable price at current contract main-start

    OI0_weekly =
        first observable OI at current contract main-start

不是 main-start week 的最终 Weekly close/OI。

============================================================
9. Partial Volume IMC
============================================================

Current Daily：

    V_daily_asof =
        volume accumulated only through anchor

Current Weekly：

    V_weekly_asof =
        volume accumulated only through anchor

绝不能使用完整 day/week final Volume。

Daily Q20 denominator：

    previous 20 CLOSED daily volumes

Weekly Q20 denominator：

    previous 20 CLOSED weekly volumes

如果不足：

    validity mask
    no future backfill

============================================================
10. Pre-generated Daily/Weekly cache
============================================================

允许提前生成：

    closed Daily bars
    closed Weekly bars

用于性能优化。

但 dataset 在 anchor t 构造输入时：

    if bar end > anchor:
        DO NOT USE FINAL CACHED BAR

必须用 causal partial aggregator 替代。

============================================================
11. Close parity tests
============================================================

增加重要测试：

在一个交易日最后有效 minute：

    dynamically aggregated Daily

应该与：

    pre-generated closed Daily

在定义一致的字段上完全一致 / tolerance一致。

同理：

在一周最后有效 minute：

    dynamically aggregated Weekly

应该与：

    pre-generated closed Weekly

一致。

如果不一致：

    investigate data semantics

不要 silent switch。

============================================================
12. Anti-leakage tests
============================================================

增加：

    test_daily_partial_does_not_see_future_minutes
    test_weekly_partial_does_not_see_future_minutes

方法：

给定 anchor t：

    build Daily/Weekly

然后任意修改：

    minute data after t

重新 build。

要求：

    all V1.1 online Daily/Weekly tensors
    remain bitwise/effectively identical

同时：

    modify minute data <= t

必须能够改变对应 partial bar。

============================================================
13. Example audit
============================================================

随机抽历史 anchor：

例如：

    10:00
    11:00
    14:00

输出：

    Daily as-of OHLCV/OI
    Weekly as-of OHLCV/OI

确认随 anchor 单调/合理更新：

    high cannot decrease
    low cannot increase
    cumulative volume cannot decrease
    close/OI update to current anchor

并确认没有使用日终/周终数据。

============================================================
14. Formal invariant
============================================================

对于 online feature builder F：

    F(history <= t)

必须满足：

    changing any raw observation with timestamp > t

不能改变：

    F_t

即：

    F(X_{<=t}, X_{>t})
        =
    F(X_{<=t}, X'_{>t})

这是 V1.1 Daily / Weekly 数据层的硬性 causality gate。

没有通过：

    V1_1_DATA_INTEGRATION_FAIL
