# Market-JEPA V0.6.1 性能优化记录

**文档版本：0.1.0**  
**日期：2026-09-03**  
**状态：候选实现已完成，等待独占 GPU 复测后合入正式工作区**  
**基准实现：`070cfa04ad31a6feaf131da8d577c8c83ff60cb6`**

## 1. 目的与约束

本轮工作只寻找不改变 Market-JEPA V0.6.1 数学协议的实现级加速。冻结项仍包括模型、数据、样本顺序、micro batch=64、gradient accumulation=2、每 epoch 全量 validation、optimizer、scheduler、EMA 更新公式、checkpoint selection 和 deterministic exact-resume contract。

候选修改在临时 checkout `/tmp/market-jepa-opt.OLB1ao` 中开发。正式训练工作区没有被修改，已经启动的正式训练也没有使用本文候选优化。

验收原则：

1. 模型输入和 loss 数学定义不变；
2. 连续训练和 checkpoint resume 的轨迹保持一致；
3. CUDA AMP 下多步更新的指标与最终 `state_dict` 精确一致；
4. 只根据独占 GPU 基准决定最终性能结论，共享 GPU 数字只作方向性参考。

## 2. 已观察到的基准与瓶颈

正式模型基准报告 `artifacts/performance/final.json` 记录：

| 项目 | 观察值 |
| --- | ---: |
| train samples | 413,764 |
| validation samples | 164,682 |
| train microbatches / epoch | 6,464 |
| validation batches / epoch | 2,574 |
| train wall time / batch | 177.74 ms |
| validation wall time / batch | 77.98 ms |
| 预计 train / epoch | 19.15 min |
| 预计 validation / epoch | 3.35 min |
| 预计 50 epochs | 18.75 h |
| CUDA peak allocated | 1.86 GiB |
| CUDA peak reserved | 2.88 GiB |

分阶段计时显示单个训练 batch 的主要耗时为：

| 阶段 | 时间 |
| --- | ---: |
| Minute Market Transformer | 29.93 ms |
| Daily GRU | 22.72 ms |
| Weekly GRU | 6.68 ms |
| 三个 EMA future target 合计 | 约 14.75 ms |
| backward | 103.51 ms |
| optimizer + EMA | 2.45 ms |

60 秒 GPU 采样中，大部分时刻 SM utilization 为 74%--98%；正式训练进程约占用 175% CPU，四个 DataLoader worker 各约 16%。短 profiler 还确认 PyTorch 已经使用 FlashAttention。CUDA self-time 的主要类别为 fill 13.5%、copy 11.1%、FlashAttention backward 9.24%、cuDNN RNN backward 9.12%、matrix multiply 9.07% 和 add 8.57%。

因此确认的结论是：

- DataLoader 曾有可消除的 Python 构造和传输开销；
- 但预取稳定后训练主体主要是 GPU forward/backward，单靠 validation cache 无法把总时间从约 18.75 小时降到 12 小时；
- 当前显存有较大余量，但冻结的 batch/accumulation 协议禁止仅为速度更改 batch。

## 3. 保留的实现优化

### 3.1 关闭的 VICReg 项不再进入 backward graph

V0 default 的 `lambda_var=0`、`lambda_cov=0`。原实现仍从带梯度的 latent 构造 variance/covariance loss，虽然乘零后不改变参数更新，但 backward 仍遍历对应 graph。

候选实现改为：

- prediction loss 仍按原定义参与梯度；
- 当权重为零时，variance/covariance 只从 `latent.detach()` 计算，继续完整记录诊断指标；
- 只有权重大于零时才把相应正则加入 total loss 和 backward graph。

新增测试同时覆盖“关闭时没有正则梯度”和“开启时仍有正则梯度”。

### 3.2 训练专用 lean Dataset 与 collate

原训练 batch 同时构造和搬运了只供 export/evaluation 使用的字段，包括 source provenance、timestamp、trading day、persistence windows 和 future outcomes。

候选实现增加 `include_evaluation_fields=False` 的训练模式及 `collate_training_batch`：

- 训练和 validation 只生成 model inputs 与 future targets；
- 完整 Dataset 默认行为不变，export/evaluation 仍获得全部字段；
- Daily/Weekly feature snapshot 可跳过 source-index array 构造；
- full/lean 两种 Dataset 的所有模型 tensor 已做逐字节一致测试。

真实数据、`num_workers=0` 的 CPU 构造基准为：

| schema | 时间 / batch | tensor bytes / batch |
| --- | ---: | ---: |
| full | 116.8 ms | 4.92 MB |
| lean | 2.82 ms | 3.71 MB |

即纯 CPU batch 构造快 41.45 倍、batch tensor 体积减少 24.6%。在具有 worker prefetch 的完整训练中，端到端收益明显小于 41.45 倍，因为 GPU compute 仍是主耗时。

### 3.3 sequence lengths 和导出 metadata 保留在 CPU

原路径把 `daily_lengths`/`weekly_lengths` 搬到 CUDA，随后 `pack_padded_sequence` 又调用 `.cpu()`。候选实现集中管理 device transfer：

- sequence lengths 保持在 CPU；
- export-only provenance、timestamp 和 outcomes 也保持在 CPU；
- 其余 tensor 继续用 non-blocking transfer 搬到目标 device。

CUDA 测试明确检查 lengths 在 CPU、模型 tensor 在 CUDA。

### 3.4 缓存 optimizer 参数列表和 EMA 参数映射

候选实现只在初始化时建立一次：

- optimizer/gradient clipping 的 trainable parameter list；
- online Minute Market Encoder 到 EMA target encoder 的有序参数和 buffer 对。

EMA 仍逐 tensor 执行原有 `mul_().add_()`，buffer 仍逐一 copy；没有改用不同的 foreach 算术，也没有省略 buffer 同步。初始化时先校验 online/target schema，避免静默错配。

## 4. 精确等价验证

已完成以下等价检查：

1. CPU 三步 baseline/candidate：全部 loss/diagnostics 相同，最终 state SHA-256 相同；
2. CUDA AMP 三步 baseline/candidate：全部指标逐项相同，最终 state SHA-256 均为  
   `9ffe0c28c8f8ef467240fee5826a40ddaa00b5a84a0e3199109df13adf5ed86b`；
3. lean/full Dataset：所有模型输入和 future target tensor byte-identical；
4. CPU smoke：2 epochs 完整通过；
5. CUDA deterministic smoke：2 epochs 完整通过；
6. 当前全量测试：35 passed。

这些结果证明候选修改没有改变已覆盖的 CPU/CUDA 三步参数轨迹和 smoke 路径。50-epoch baseline/candidate 的完整轨迹尚未重复运行，因此不能把三步验证表述为完整 50-epoch 证明。

## 5. 已尝试但拒绝的方案

### 5.1 Daily/Weekly GRU prefix cache

设想是在 eval/export 时缓存 completed history 的两层 GRU hidden state，只对当前 partial token 再走一步。该方法语义上合理，但实测没有满足严格数值等价：

- AMP 最大绝对误差：`2.44e-4`；
- FP32 最大绝对误差：`5.87e-5`。

由于 validation H64 loss 决定 best checkpoint，即使小数值差异也可能改变 checkpoint selection。本轮不采用 prefix cache。

### 5.2 增大 validation batch

候选吞吐结果：

| validation batch | samples/s | 相对 batch=64 | peak reserved |
| --- | ---: | ---: | ---: |
| 64 | 548 | 1.00x | 0.41 GiB |
| 128 | 653 | 1.19x | 未单列 |
| 256 | 710 | 1.29x | 1.74 GiB |
| 512 | 653 | 1.19x | 未单列 |
| 1024 | 696 | 1.27x | 6.19 GiB |

batch=256 最快，但同一模型、同一批样本的结果出现：

- `z_market` 最大绝对差 `1.8288e-3`；
- H64 prediction 最大绝对差 `4.8828e-4`；
- H64 epoch metric 差 `2.657e-7`；
- H64 target 本身差为 0。

该方案会改变决定 checkpoint 的 validation metric，因此不属于严格等价的 V0.6.1 实现优化。可作为预先登记的 V0.6.2 方案，不在本轮采用。

### 5.3 `torch.compile`

共享 GPU 短基准中，默认 compile mode 的训练吞吐约提高 1.45--1.46 倍，且 bound forward 的 `state_dict` schema 不变。但它未通过 exact-resume contract：

- 连续运行 2 epochs；
- 与“冷启动 1 epoch、保存、在新进程恢复第 2 epoch”比较；
- epoch 0 一致，恢复后的 epoch 1 参数轨迹不同。

尝试 `align_random_eager=True` 和 `fallback_random=True` 后仍不同。`reduce-overhead` 还会在 gradient accumulation=2 下触发 CUDAGraph gradient-buffer overwrite RuntimeError。

因此本轮完全移除 compile 路径。除非未来能保存/恢复编译 dropout 的完整随机状态，或 V0.6.2 明确放宽 exact-resume contract，否则不能采用。

### 5.4 训练期共享 GRU prefix

训练态包含 dropout、autograd 和每样本 graph。共享 prefix 会改变 dropout mask、浮点计算顺序和梯度图，无法继续声称 V0.6.1 严格等价，因此未实现。

## 6. 共享 GPU 下的候选结果

正式训练占用同一块 GPU 时进行的短 A/B 结果为：

| 指标 | baseline | candidate | 变化 |
| --- | ---: | ---: | ---: |
| train wall / batch | 313.38 ms | 312.45 ms | 约 0.3% 加速 |
| validation wall / batch | 126.82 ms | 119.73 ms | 约 5.9% 加速 |
| loader | 30.81 ms | 4.35 ms | 显著下降 |
| H2D | 16.58 ms | 9.05 ms | 显著下降 |
| backward | 183.16 ms | 183.12 ms | 基本不变 |

这些数字证明 lean Dataset 和 CPU length 路径确实消除了 loader/H2D 工作，但由于两个进程共享 GPU，不能据此计算最终 50-epoch ETA。尤其 train 总时间几乎不变，说明剩余主成本仍是模型 forward/backward。

## 7. 合入与后续复测门禁

正式训练结束前，不得把候选源码复制回正式工作区，因为 Trainer 每 epoch 校验 implementation hash，修改跟踪文件会使正在运行的 V0 checkpoint hard fail。

候选代码合入前必须完成：

1. `git diff --check`；
2. 全量 pytest；
3. CPU smoke；
4. CUDA deterministic smoke；
5. 独占 GPU、正式模型、相同数据和相同 batch 数的 baseline/candidate A/B；
6. 报告 `max_memory_allocated`、`max_memory_reserved`、train/validation 分段时间和预计 50-epoch 时间；
7. 自审 diff，确认没有残留 `torch.compile` 或改变 validation batch；
8. 正式训练结束并确认 checkpoint/产物完整后，再把候选 commit 合入正式工作区，并从 epoch 0 启动优化实现的正式实验。

当前可确认的是实现等价性和 loader/H2D 减负；尚不能确认最终总训练时间是否低于 16.8 小时。
