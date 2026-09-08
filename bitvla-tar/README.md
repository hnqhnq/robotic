# BitVLA-TAR

**TAR（Temporal Action Regularization）** — 面向 VLA 模型动作 chunk 平滑性的时序动作正则化。

基于 [BitVLA](https://github.com/ustcwhy/BitVLA)。本目录是 [robotic](https://github.com/hnqhnq/robotic) monorepo 内的独立复现工作区，后续可拆分为独立仓库。

## 项目结构

```text
bitvla-tar/
├── README.md              # 本文件
├── doc/                   # 论文、实验报告、复现指南
├── scripts/               # 下载、训练、评估辅助脚本
├── openvla-oft/           # VLA 训练与 LIBERO 评估（BitVLA fork + TAR 补丁）
├── transformers/          # 修改版 transformers（BitVLA 依赖）
├── data/                  # LIBERO RLDS 数据（本地下载，gitignore）
├── checkpoints/           # 预训练与微调权重（gitignore）
└── runs/                  # 训练 / 评估输出（gitignore）
```

## 快速开始

### 1. 环境

参考 BitVLA 上游安装说明，再安装本地包：

```bash
cd bitvla-tar
pip install -e transformers/
pip install -e openvla-oft/bitvla/
# LIBERO 仿真环境：见 doc/REPRODUCE.md
```

### 2. 下载数据与模型

```bash
bash scripts/download_data.sh
bash scripts/download_models.sh
```

若 robotic monorepo 的 `../src/data` 或 `../src/checkpoints` 已有资产：

```bash
bash scripts/link_local_assets.sh   # 可选：本地开发符号链接
```

### 3. 复现 TAR 实验

完整 checklist 见 **[doc/REPRODUCE.md](doc/REPRODUCE.md)**（含上服务器前 / 后对照表）。

**实验阶段**（论文 = v3 + v4）：

| 阶段 | λ | 脚本 | 报告 |
|------|---|------|------|
| v3 主实验 | 0.05 | `train_tar.sh 0.05` | `doc/20260418-实验报告.md` |
| v4 消融 | 0.01, 0.10 | `train_v4_ablation.sh` | `doc/20260504-实验报告.md` |

**训练**（4× GPU）：

```bash
# v3 — 首次成功的 TAR 训练（90.8%）
bash scripts/train_tar.sh 0.05

# v4 — 论文消融（89.0% + 93.6%），合计 ~80h
bash scripts/train_v4_ablation.sh

# 或手动指定任意 λ：
bash scripts/train_tar.sh 0.01    # → run tag v4-tar001
bash scripts/train_tar.sh 0.10    # → run tag v4-tar01
```

**评估**（1× GPU）：

```bash
bash scripts/eval_tar.sh /path/to/checkpoint [label]

# v4 串联 eval（~23h）：
bash scripts/eval_v4_both.sh /path/to/ckpt_0.01 /path/to/ckpt_0.10
```

兼容旧名：`train_v3_tar.sh` / `eval_v3_tar.sh` 为薄封装。

## 论文结果（参考）

| λ_TAR | 成功率 (%) | S1 ↓ | MaxJump ↓ | InterBoundary ↓ |
|-------|------------|------|-----------|-----------------|
| 0.00  | 86.4       | 0.0154 | 0.1613  | 0.0477          |
| 0.01  | 89.0       | 0.0111 | 0.0844  | 0.0421          |
| 0.05  | 90.8       | 0.0104 | 0.0794  | 0.0451          |
| 0.10  | 93.6       | 0.0099 | 0.0761  | 0.0460          |

详情：`doc/TAR-NingqiuHe.pdf`、`doc/20260418-实验报告.md`、`doc/20260504-实验报告.md`（完整 4 点消融）。全部文档索引见 [doc/README.md](doc/README.md)。

## 进度

- [x] 项目骨架与 BitVLA 代码拷贝
- [x] TAR loss（`openvla-oft/vla-scripts/finetune_bitnet.py`）
- [x] FSDP + gradient checkpointing（4×24GB 训练）
- [x] LIBERO 评估 smoothness 指标
- [ ] 完整训练与评估复现（需 GPU 服务器）

## 引用

若使用本工作，请引用 TAR 论文（见 `doc/TAR-NingqiuHe.pdf`）及 BitVLA 论文：

```bibtex
@misc{bitvla2025,
  title={BitVLA: 1-bit Vision-Language-Action Models},
  author={...},
  howpublished={\url{https://github.com/ustcwhy/BitVLA}}
}
```

## 许可证

代码继承 BitVLA 上游许可证（`LICENSE`）。TAR 相关新增部分属于本复现项目。
