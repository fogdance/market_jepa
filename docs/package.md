# Market-JEPA V0 训练结束 Review 打包要求

正式 50 epoch 训练结束后，不修改模型、不重新训练、不提前进行 V1/RSSM/实时交易工作。

请收集本次正式 V0 实验的训练证据，并生成：

`market_jepa_v0_training_review.tar.bz2`

## 1. 必须包含的文件

### A. 训练历史

必须包含：

`artifacts/checkpoints/market_jepa_v0_default/history.json`

这是最重要的文件，必须包含完整 epoch 0–49 的 Train / Validation history。

---

### B. 完整训练日志

必须包含：

`artifacts/logs/market_jepa_v0_default.log`

如果训练曾 resume，不要只提供最后一段日志。

需要提供从正式 epoch 0 开始到 epoch 49 结束的完整日志。

---

### C. 正式实验配置

必须包含：

`configs/market_jepa_v0.yaml`

必须是实际用于正式 V0 default experiment 的配置文件。

---

### D. Preflight 数据审计

查找本次正式实验实际使用的 preflight 输出，将整个 preflight 目录一起放入包中。

例如：

`artifacts/market_jepa_v0_default/preflight/`

或者项目实际生成的对应目录。

至少必须包含：

* preflight JSON
* anomaly CSV
* normalization / distribution audit
* source SHA-256
* 数据行数与日期范围
* trading day / week 统计

不要重新生成覆盖原始正式训练使用的 preflight 文件。

---

### E. Best checkpoint 元数据

**不需要默认把 best.pt 本体打包。**

生成：

`review_metadata/best_checkpoint_sha256.txt`

内容来自：

```bash
sha256sum artifacts/checkpoints/market_jepa_v0_default/best.pt
```

同时生成：

`review_metadata/best_checkpoint_info.txt`

至少写入：

```text
best checkpoint path
best epoch
best Validation H64 cosine error
checkpoint file size
checkpoint SHA-256
```

如果 best epoch 已经记录在 history/checkpoint 中，从正式记录读取，不要手工推断。

---

### F. Last checkpoint 元数据

生成：

`review_metadata/last_checkpoint_sha256.txt`

来自：

```bash
sha256sum artifacts/checkpoints/market_jepa_v0_default/last.pt
```

不需要默认打包 `last.pt` 本体。

---

### G. Git 信息

生成：

`review_metadata/git_commit.txt`

内容：

```bash
git rev-parse HEAD
```

生成：

`review_metadata/git_status.txt`

内容：

```bash
git status --short
```

正式实验结束时应该能判断是否存在未提交的源码改动。

再生成：

`review_metadata/git_log.txt`

内容：

```bash
git log --oneline -10
```

---

### H. Implementation hash

如果项目已有 implementation manifest/hash，必须复制正式训练对应的：

* implementation manifest
* implementation SHA-256

到：

`review_metadata/`

如果 checkpoint 内记录 implementation hash，也将其导出成文本：

`review_metadata/implementation_sha256.txt`

---

### I. Source data hash

从正式实验 checkpoint / preflight 中提取并保存：

`review_metadata/source_data_sha256.txt`

必须对应正式训练实际使用的 minute CSV。

不要重新根据另一个 CSV 猜测。

---

### J. Runtime / 性能配置

如果正式训练使用：

`selected_runtime.json`

或类似 runtime benchmark 结果，请一起包含。

例如：

`artifacts/performance/selected_runtime.json`

并把正式训练实际使用的：

```text
num_workers
persistent_workers
prefetch_factor
pin_memory
device
AMP
```

整理到：

`review_metadata/runtime_info.txt`

---

### K. 环境信息

生成：

`review_metadata/environment.txt`

内容至少包括：

```bash
python --version
uv --version
nvidia-smi
```

以及通过当前项目环境获取：

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

---

### L. 测试结果

正式训练结束后，在**不修改任何源码**的情况下执行：

```bash
uv run pytest -q
```

保存完整输出：

`review_metadata/pytest.txt`

如果测试失败，不要修复后覆盖这个实验状态；先保留失败信息。

---

## 2. 建议包含

如果存在以下文件，也一起加入：

* benchmark JSON
* performance baseline
* manifest
* experiment metadata
* checkpoint metadata JSON
* training timing statistics
* epoch timing
* GPU peak memory
* preflight manifest
* resume manifest

但不要加入无关大文件。

---

## 3. 第一批不要打包

除非后续明确要求，否则第一批 Review 包不要包含：

* `best.pt`
* `last.pt`
* 大型 `.npz` latent
* 原始行情 CSV
* Test latent
* 大型 profiler trace
* 整个 `.venv`
* 整个 `artifacts/` 目录
* cache 文件

模型文件只提供 SHA-256 和 metadata。

---

## 4. 生成统一 Review 目录

建立：

```text
market_jepa_v0_training_review/
├── config/
│   └── market_jepa_v0.yaml
│
├── training/
│   ├── history.json
│   └── market_jepa_v0_default.log
│
├── preflight/
│   └── <正式实验原始 preflight 文件>
│
├── performance/
│   └── <selected runtime / benchmark，如存在>
│
└── review_metadata/
    ├── git_commit.txt
    ├── git_status.txt
    ├── git_log.txt
    ├── implementation_sha256.txt
    ├── source_data_sha256.txt
    ├── best_checkpoint_sha256.txt
    ├── last_checkpoint_sha256.txt
    ├── best_checkpoint_info.txt
    ├── runtime_info.txt
    ├── environment.txt
    └── pytest.txt
```

如实际文件名稍有不同，可以复制到以上标准名字，但不能修改文件内容。

---

## 5. 额外生成 REVIEW_MANIFEST.txt

生成：

`market_jepa_v0_training_review/REVIEW_MANIFEST.txt`

列出：

```text
Experiment ID
Design version
Git commit
Implementation SHA-256
Source data SHA-256
Config SHA-256
Best epoch
Best Validation H64 error
Best checkpoint SHA-256
Last checkpoint SHA-256
Training start time
Training finish time
Whether resume occurred
Number of completed epochs
Pytest result
GPU
PyTorch version
CUDA version
```

并列出包内所有文件及各自 SHA-256。

---

## 6. 最终压缩

最终生成：

```bash
tar -cjf market_jepa_v0_training_review.tar.bz2 \
    market_jepa_v0_training_review
```

然后：

```bash
sha256sum market_jepa_v0_training_review.tar.bz2 \
    > market_jepa_v0_training_review.tar.bz2.sha256
```

最终需要给用户两个文件：

```text
market_jepa_v0_training_review.tar.bz2
market_jepa_v0_training_review.tar.bz2.sha256
```

---

## 7. 重要限制

本步骤只是为了进行：

**Market-JEPA V0 Training Stage Review**

不要在打包过程中：

* 修改模型
* 修改配置
* 重跑正式训练
* 重选 best checkpoint
* 根据 Test 结果修改任何内容
* 接入实时行情
* 开始 RSSM
* 开始交易策略
* 生成 BUY/SELL
* 开始 V1

完成打包后停止。

