# TAR 复现指南

分阶段 checklist，用于复现 **TAR: Temporal Action Regularization for Smooth Action Chunking in Vision-Language-Action Models** 论文结果。

---

## 上服务器前 / 上服务器后对照

### 上服务器前（本机可完成）

| 项目 | 状态 | 说明 |
|------|------|------|
| 代码实现 | ✅ 已完成 | TAR loss、FSDP、smoothness 评估 |
| 训练 / 评估脚本 | ✅ 已完成 | `scripts/train_tar.sh`、`eval_tar.sh` 等 |
| 实验文档 | ✅ 已完成 | 4 份报告 + 论文 + 本指南 |
| Git 推送 | ✅ 已完成 | `hnqhnq/robotic`  monorepo |
| 本机 GPU 逻辑验证 | ✅ 可选 | `bash scripts/verify_tar_gpu.sh` |
| LIBERO 数据 | ⚠️ 部分 | 本机 `src/data` 约 3.5GB；需链接或上传到服务器 |
| bitvla-bf16 权重 | ❌ 待下载 | 当前为 LFS 占位符，需完整下载（数 GB） |
| 完整 Python 环境 | ⚠️ 服务器装 | LIBERO sim、`accelerate` 等依赖 |
| 端到端 train / eval | ❌ 需服务器 | 训练需 4×24GB GPU |

**本机可选准备：**

```bash
cd bitvla-tar
bash scripts/link_local_assets.sh   # 链接 monorepo 已有 data / checkpoints
bash scripts/download_models.sh     # 下载完整 bitvla-bf16（需网络 + 磁盘）
```

---

### 上服务器后（按顺序执行）

#### Day 1 — 环境与资产（Phase 0–1）

```bash
git clone https://github.com/hnqhnq/robotic.git && cd robotic/bitvla-tar

# 环境（参考 BitVLA README）
pip install -e transformers/
pip install -e openvla-oft/bitvla/
# 安装 LIBERO、robosuite 等仿真依赖

bash scripts/download_data.sh       # 或 rsync / scp 本机已有数据
bash scripts/download_models.sh     # bitvla-bf16 + 可选 baseline

bash scripts/verify_tar_gpu.sh      # 确认 GPU / CUDA / TAR 代码路径
```

#### Day 2 — Smoke test（Phase 3，约 30 分钟）

```bash
MAX_STEPS=100 bash scripts/train_tar.sh 0.05
# 检查 logs：tar_loss 应从 ~0.018 向 ~0.001 下降
```

#### Day 3+ — 正式实验（Phase 4–5）

**最低成本路径 L1（推荐）：**

1. Eval 官方 baseline（不训练）→ 确认 sim 管道正常
2. 只训 **λ=0.10**（论文主结果 93.6%）→ 约 40h 训练 + 11h eval

**完整复现 L3：**

```bash
bash scripts/train_tar.sh 0.05          # v3 主实验，~40h
bash scripts/train_v4_ablation.sh       # v4 消融 λ=0.01 + 0.10，~80h

# 每次 eval 前修复 checkpoint
bash scripts/fix_fsdp_checkpoint.sh runs/.../10000_chkpt
bash scripts/eval_tar.sh runs/.../lora_adapter v3-tar

# v4 串联 eval（~23h）
bash scripts/eval_v4_both.sh <ckpt_0.01> <ckpt_0.10>
```

**GPU 预算参考：**

| 路径 | 训练 | 评估 | 合计 |
|------|------|------|------|
| L1（baseline + λ=0.10） | ~40h | ~22h | ~55–65 GPU·h |
| L2（+ λ=0.05） | +40h | +11h | ~105 GPU·h |
| L3（Table IV 四点） | ~120h | ~44h | ~200+ GPU·h |

---

## Phase 0 — 资产准备

| 资产 | 来源 | 本地路径 |
|------|------|----------|
| 预训练 BitVLA | [lxsy/bitvla-bf16](https://huggingface.co/lxsy/bitvla-bf16) | `checkpoints/bitvla-bf16/` |
| LIBERO RLDS | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) | `data/modified_libero_rlds/` |
| Baseline（可选） | [ft-bitvla-libero-long](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16) | `checkpoints/ft-bitvla-libero-long/` |

```bash
bash scripts/download_data.sh
bash scripts/download_models.sh
```

---

## Phase 1 — 环境与 baseline 评估

1. 安装依赖（CUDA、PyTorch、LIBERO、robosuite — 对齐 BitVLA README）。
2. 用官方 long checkpoint 在 LIBERO-10 上 eval（50 trials × 10 tasks = 500 episodes）。
3. 确认仿真管道正常后再开始训练（1 GPU 约 11–14 小时）。

预期 proxy baseline：成功率约 80–90%（官方 checkpoint 可能与论文 86.4% 略有出入）。

---

## Phase 2 — TAR 代码实现

**状态：已实现。** 环境就绪后可跳过本章，直接进入 Phase 3 smoke test。

### 2.1 训练 loss

文件：`openvla-oft/vla-scripts/finetune_bitnet.py`

在 `run_forward_pass()` 中，`predicted_actions` 计算完成后施加 TAR：

```python
v_start = predicted_actions[:, 1, :] - predicted_actions[:, 0, :]
v_end   = predicted_actions[:, -1, :] - predicted_actions[:, -2, :]
tar_loss = (v_start.abs() + v_end.abs()).mean()
loss = loss + tar_lambda * tar_loss
```

CLI 参数：`--tar_lambda`（默认 `0.0`）。`tar_loss` 会写入训练 metrics。

### 2.2 FSDP（4× 4090D 24GB）

论文 v3 使用 FSDP + `gradient_checkpointing=True` 替代 DDP。`scripts/train_tar.sh` 默认已开启：

```bash
--gradient_checkpointing True --use_fsdp True
```

### 2.3 评估 smoothness 指标

文件：`openvla-oft/experiments/robot/libero/smoothness_metrics.py` + `run_libero_eval_bitnet.py`

评估时记录预测 action chunk，并聚合为：

- **S1** — chunk 内逐步平均变化量
- **S2** — chunk 内二阶变化量
- **InterBoundary (IB)** — chunk 边界跳跃
- **MaxJump** — chunk 内峰值步速
- **GripperSwitches** — 每 chunk 夹爪切换次数

输出：`runs/eval_logs/smoothness_<run_id>.json`（或通过 `--smoothness_json_path` 指定）。

### 2.4 FSDP checkpoint 修复（eval 前）

若 checkpoint 保存在 `lora_adapter/` 下，执行：

```bash
bash scripts/fix_fsdp_checkpoint.sh /path/to/run--10000_chkpt
```

确保 eval 目录包含：

- `dataset_statistics.json`（含 `libero_10_no_noops` key）
- 完整 tokenizer / processor（词表约 128264）
- 指向父目录 `proprio_projector*.pt`、`action_head*.pt` 的符号链接

详见 `doc/20260418-实验报告.md` 第四节。

---

## Phase 3 — Smoke test（约 30 分钟）

```bash
MAX_STEPS=100 bash scripts/train_tar.sh 0.05
```

检查日志：`tar_loss` 应从 ~0.018 向 ~0.001 下降。

---

## Phase 4 — 完整训练

| 参数 | 值 |
|------|-----|
| 基座模型 | `bitvla-bf16` |
| 数据集 | `libero_10_no_noops` |
| GPU | 4 × 4090D（FSDP） |
| max_steps | 10,001 |
| effective batch | 64（bs=2 × accum=8 × 4） |
| learning_rate | 4e-4（ViT 8e-5） |
| image_aug | True |

### v3（λ=0.05）— 主实验

```bash
bash scripts/train_tar.sh 0.05          # ~40–46 h
# checkpoint run tag: v3-tar--image_aug+tar-0.05
```

报告：`doc/20260418-实验报告.md`

### v4（λ=0.01, 0.10）— 论文消融

```bash
bash scripts/train_v4_ablation.sh       # ~80 h（A 然后 B）
# 或分开跑：
bash scripts/train_tar.sh 0.01           # v4-tar001
bash scripts/train_tar.sh 0.10           # v4-tar01
```

报告：`doc/20260504-实验报告.md`

---

## Phase 5 — 评估

```bash
bash scripts/eval_tar.sh runs/<checkpoint>/lora_adapter [label]
```

500 episodes，1 GPU 约 11 小时 / 模型。

**v4 串联 eval**（对齐原云工作流）：

```bash
bash scripts/eval_v4_both.sh \
  runs/.../tar-0.01--v4-tar001--10000_chkpt/lora_adapter \
  runs/.../tar-0.1--v4-tar01--10000_chkpt/lora_adapter
```

### 目标数值

| λ | 成功率 | 来源 |
|---|--------|------|
| 0.05 | 90.8% | v3 |
| 0.10 | **93.6%** | v4（论文主结果） |

| 指标（λ=0.05） | Baseline | TAR |
|----------------|----------|-----|
| 成功率 | 86.4% | 90.8% |
| S1 | 0.0154 | 0.0104 |
| MaxJump | 0.1613 | 0.0794 |
| IB | 0.0477 | 0.0451 |

因 seed / FSDP vs DDP 差异，精确复现可能有 ±1–3 个百分点波动。

---

## Phase 6 — 论文完整四点消融

| λ | 训练命令 | 论文中已有？ |
|---|----------|--------------|
| 0.00 | baseline eval（官方 long ckpt） | ✅ |
| 0.01 | `train_tar.sh 0.01` | v4 |
| 0.05 | `train_tar.sh 0.05` | v3 |
| 0.10 | `train_tar.sh 0.10` | v4 |

全部 train + eval 合计约 200+ GPU·h。

---

## 相关文档

- `doc/README.md` — 全部实验文档索引
- `doc/TAR-NingqiuHe.pdf` — 论文
- `doc/20260401-实验报告.md` — v1/v2 失败分析与 v3 设计
- `doc/20260418-实验报告.md` — v3 主实验（λ=0.05）
- `doc/20260504-实验报告.md` — v4 消融（λ=0.01/0.10，论文 Table IV）
