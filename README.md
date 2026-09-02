# Market-JEPA V0

独立的多时间尺度 JEPA 研究实验，用于检验当前 causal market latent 是否包含可泛化的未来市场分布信息。冻结的研究协议见 [设计文档](docs/market_jepa_v0_design.md)，当前版本为 `0.6.1`，状态为 `Frozen — Approved for V0 implementation`。

V0 不包含交易动作、持仓、PnL、强化学习或外部行情依赖。唯一行情源是仓库根目录的 `8Y_DCE_JM2601_1m.csv`。

## 已实现范围

- 从 1m 数据推断 trading day，并在每个 anchor 构建严格 causal 的 partial Daily/Weekly snapshot；
- 独立的 Minute Market Transformer 和 Minute Context GRU；EMA target 只接收与 online market encoder 同构的 market-only 输入；
- H16/H64/H256 独立 predictor、persistence/train-mean/block-shuffle controls；
- train-only normalization、固定 chronological split、future-tail purge；
- preflight 异常报告、checkpoint、latent export、kNN/frozen probe/raw baseline 和 block bootstrap；
- deterministic CPU smoke 与正式训练前的 CUDA smoke 门禁。

## 环境与数据

项目由 `uv.lock` 锁定依赖。创建环境及安装：

```bash
uv sync --dev
```

安装完成后先确认 PyTorch 能看到目标 GPU：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

CSV 必须保留字段：

```text
Date,Open,High,Low,Close,Volume,OpenInterest
```

导出 metadata 使用品种 `symbol=JM` 和来源序列 `series_id=8Y_DCE_JM2601`；不把该八年文件表述为真实单合约 JM2601。

实现不会自动删除、修复或 winsorize 异常行情；preflight 只生成报告。

## 验证顺序

先运行单元测试：

```bash
uv run pytest -q
```

CPU 端到端 smoke 使用缩小的模型、2 epochs 和每个 split 最多 16 个样本，只验证数据到评估的管线：

```bash
uv run python train_market_jepa.py --smoke
```

正式训练前必须在目标 RTX 4060 Ti 上运行 CUDA deterministic smoke：

```bash
uv run python train_market_jepa.py --cuda-smoke
```

正式模型的性能基准会比较 `num_workers=0/2/4`、校验候选批次逐字节一致，
并把选中的纯运行时 DataLoader 参数写入
`artifacts/performance/selected_runtime.json`：

```bash
uv run python benchmark_market_jepa.py \
  --output artifacts/performance/optimized.json
```

正式训练会自动读取该运行时文件；它不改变冻结的数据、模型、batch 或优化器协议。

只有 CUDA smoke 完整执行 forward、AMP backward、optimizer、scheduler、EMA 且无 deterministic 错误后，才允许启动冻结的 50-epoch V0：

```bash
uv run python train_market_jepa.py
```

恢复训练必须使用同一数据文件和运行时参数；程序会校验 source SHA-256、配置、
normalizer、Python 源码 manifest、依赖锁文件及全部 RNG state：

```bash
uv run python train_market_jepa.py --resume artifacts/checkpoints/market_jepa_v0_default/last.pt
```

## 导出与评估

导出 train/validation/test 的 frozen latent：

```bash
uv run python export_market_latents.py \
  --checkpoint artifacts/checkpoints/market_jepa_v0_default/best.pt \
  --split all
```

仅导出某个 split 时可用 `--split train|validation|test`。默认在 CPU 上导出；显式指定 `--device cuda` 可使用 GPU。

执行预注册评估并写入一次性 Test manifest：

```bash
uv run python eval_market_jepa.py \
  --checkpoint artifacts/checkpoints/market_jepa_v0_default/best.pt \
  --latents-dir artifacts/latents/market_jepa_v0_default
```

若必须重跑，需给出标签及新输出路径，避免覆盖原始 Test 记录：

```bash
uv run python eval_market_jepa.py \
  --checkpoint artifacts/checkpoints/market_jepa_v0_default/best.pt \
  --latents-dir artifacts/latents/market_jepa_v0_default \
  --output artifacts/evaluation/market_jepa_v0_default/evaluation_rerun.json \
  --rerun-label reason
```

## 产物

所有运行产物都在 `artifacts/` 下，并被 Git 忽略：

- `<experiment_id>/preflight/`：JSON 汇总和 Top-100 异常 CSV；
- `checkpoints/<experiment_id>/`：`best.pt`、`last.pt`、训练历史；
- `latents/<experiment_id>/`：各 split 的 NPZ；
- `evaluation/<experiment_id>/`：评估 JSON 和带 hash 的 manifest。

`best.pt` 只按完整 Validation 的 H64 learned-to-target cosine error 选择；Test 不参与训练、调参或 checkpoint selection。

三种预注册 ablation 分别使用以下配置；除 experiment ID 和 ablation 外，冻结参数完全相同：

- `configs/market_jepa_v0_minute.yaml`
- `configs/market_jepa_v0_minute_daily.yaml`
- `configs/market_jepa_v0.yaml`（minute + Daily + Weekly）

## 已知限制

- CSV 横跨 2018–2025，但没有权威合约类型、显式 trading day 或交易所日历；推断结果需在生产研究前逐日核对。
- 全量 CPU smoke 不能替代目标 GPU 的 deterministic smoke。
- smoke profile 的统计结果不用于正式 GO/NO-GO。
