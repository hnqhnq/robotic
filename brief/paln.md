# BitVLA 复现与基线实验计划（含模型与数据下载地址）

---

# 1. 复现目标

## Level 1：先完成评测复现

* 跑通 BitVLA 官方 checkpoint 的 **LIBERO eval**
* 在你自己的机器上拿到 baseline 分数

## Level 2：完成训练闭环

* 跑通 **短程训练（debug）**
* 验证：数据读取 → 训练 → 保存 checkpoint → 评测

## Level 3：完成单卡 baseline

* 用单卡完成 BitVLA baseline 训练
* 为后续 TAR 方法提供对照

---

# 2. 前置准备

## 2.1 代码

* BitVLA 代码仓：你已提供 `BitVLA.zip`
* 建议解压后固定目录，并记录 commit

建议目录结构：

```text
project/
  BitVLA/
  data/
    modified_libero_rlds/
  checkpoints/
    bitvla_pretrained/
    bitvla_finetuned/
  runs/
    eval/
    train/
    videos/
  notes/
```

---

## 2.2 模型下载地址（最重要）

## A. 推荐预训练模型（优先下载）

### BitVLA - VL & VLA pre-trained

* Hugging Face：`https://huggingface.co/lxsy/bitvla-bf16`

这是 README 明确标注的 **Recommended** 版本，适合你直接作为 baseline 起点。

---

## B. 旧版 VL 预训练模型（可选）

### BitVLA - VL pre-trained

* Hugging Face：`https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16`

注意：

* 这个旧模型**不能直接无脑用于当前代码**
* 需要先执行格式转换：

```bash
python convert_ckpt.py /path/to/bitvla-bitsiglipL-224px-bf16
```

所以你当前阶段**不建议优先折腾这个**。

---

## C. LIBERO 各任务微调模型

### 1. LIBERO-Spatial

* `https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_spatial-bf16`

### 2. LIBERO-Object

* `https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16`

### 3. LIBERO-Goal

* README 表格里这一行实际给的是 `libero_long` 链接，疑似写错了
* 表中显示地址为：
  `https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16`

### 4. LIBERO-Long

* `https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16`

---

## D. 官方 fine-tuned 模型集合页

* `https://huggingface.co/collections/hongyuw/bitvla-68468fb1e3aae15dd8a4e36e`

这个集合页适合统一查看所有已发布的 BitVLA 微调模型。

---

## E. 可选 BF16 SigLIP 版本

* `https://huggingface.co/hongyuw/bitvla-siglipL-224px-bf16`

这个更偏视觉语言评测，不是你当前 LIBERO baseline 的最优先项。

---

## 2.3 数据下载地址

## LIBERO 微调数据

### modified_libero_rlds

* Hugging Face 数据集页：
  `https://huggingface.co/datasets/openvla/modified_libero_rlds`

README 给出的拉取方式：

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

如果你本机没有配置 Hugging Face SSH，也可以改用 HTTPS 克隆或在网页中下载。

---

## Open X-Embodiment 相关集合（仅了解）

BitVLA README 还提到其机器人预训练基于 Open X-Embodiment 的整理子集：

* 集合页：
  `https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot`

但这部分是**预训练层级**，你当前做 baseline 复现时**不用先准备**。

---

# 3. 环境安装

## 3.1 创建环境

```bash
conda create -n bitvla python=3.10 -y
conda activate bitvla
```

## 3.2 安装 PyTorch

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

## 3.3 安装项目依赖

```bash
cd BitVLA
pip install -e openvla-oft/
pip install -e transformers

cd openvla-oft/
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/
pip install -r experiments/robot/libero/libero_requirements.txt

cd ..
pip install -e bitvla/
```

---

# 4. 第一阶段：环境验证

## 4.1 import 测试

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print('transformers ok')"
python -c "import libero; print('libero ok')"
python -c "import bitvla; print('bitvla ok')"
```

## 4.2 GPU 测试

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

---

# 5. 第二阶段：先做评测复现（最优先）

## 5.1 推荐先用的模型

先用下面两个：

1. `lxsy/bitvla-bf16`
2. `hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16`

其中你当前最该优先跑的是：

* **LIBERO-Long fine-tuned checkpoint**

---

## 5.2 先做 smoke test

先少量 rollout，确认：

* 模型能加载
* 环境能运行
* 能输出结果
* 不报错

示例：

```bash
python experiments/robot/libero/run_libero_eval_bitnet.py \
    --pretrained_checkpoint /path/to/ft-bitvla-bitsiglipL-224px-libero_long-bf16 \
    --task_suite_name libero_10 \
    --info_in_path "smoke_test_long" \
    --model_family "bitnet"
```

---

## 5.3 再跑 full eval

smoke test 通过后，跑完整评测。

---

# 6. 第三阶段：训练复现

## 6.1 单卡 debug 先行

先做短程训练：

```text
max_steps = 200
batch_size = 1 或 2
grad_accumulation_steps = 8 或更高
```

目标：

* loss 下降
* checkpoint 能保存
* 保存后的模型能评测

---

## 6.2 中等短跑

```text
max_steps = 1000
```

目标：

* 训练稳定
* checkpoint 可用
* 初步得到单卡训练经验

---

## 6.3 单卡等效 batch 建议

原脚本等效 batch：

```text
4 × 2 × 8 = 64
```

单卡建议改成：

```text
1 × 2 × 32 = 64
```

即：

* `nproc-per-node = 1`
* `batch_size = 2`
* `grad_accumulation_steps = 32`

---

# 7. 必须记录的数据与指标

## 7.1 基础信息

* 代码 commit
* 模型路径
* 数据路径
* GPU / CUDA
* batch size
* grad accumulation
* seed

---

## 7.2 评测指标

* Success Rate
* LIBERO-Long Success
* 各 task success rate
* Latency（avg / p95）
* Throughput
* Peak VRAM

---

## 7.3 动作质量指标（为 TAR 做准备）

### 一阶平滑度

[
S_1 = \frac{1}{K-1}\sum |a_{t+1} - a_t|
]

### 二阶平滑度

[
S_2 = \frac{1}{K-2}\sum |a_{t+2} - 2a_{t+1} + a_t|
]

还建议记录：

* chunk 内最大 action jump
* gripper 切换稳定性
* 后半段 episode 的动作抖动

---

## 7.4 训练日志

* total loss
* learning rate
* step time
* GPU memory
* checkpoint 保存时间
* 每个 checkpoint 对应 eval 结果

---

# 8. 建议建立的记录文件

```text
notes/
  env_manifest.md
  assets.md
  repro_log.md
  baseline_results.md
```

---

## env_manifest.md

记录：

* Python
* torch
* CUDA
* GPU
* 系统版本
* 代码 commit

## assets.md

记录：

* 模型下载地址
* 数据下载地址
* 本地保存路径

## repro_log.md

按时间记录你做过什么

## baseline_results.md

记录每次实验结果

---

# 9. 推荐执行顺序

1. 解压代码并记录 commit
2. 安装环境
3. 完成 import 测试
4. 下载推荐模型
5. 下载 `modified_libero_rlds`
6. 先跑 LIBERO eval smoke test
7. 再跑 full eval baseline
8. 再做单卡训练 debug
9. 再做中等短跑
10. 最后才开始加 TAR

---

# 10. 当前最优先任务

> **只做一件事：跑通 LIBERO-Long baseline eval**

不要现在先去做：

* 旧模型转换
* 复杂训练
* 创新改代码
* Open X 预训练复现

---

# 11. 当前建议你优先下载的清单

## 模型

* `https://huggingface.co/lxsy/bitvla-bf16`
* `https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16`

## 数据

* `https://huggingface.co/datasets/openvla/modified_libero_rlds`

---

# 12. 一句话总结

> **先用官方 long checkpoint 跑通 eval，建立 baseline；再跑单卡 debug 训练；最后再做 TAR。**
