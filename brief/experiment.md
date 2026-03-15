# 一、BitVLA 研究整体路线图

| 阶段      | 目标         | 任务                           | 输出             |
| ------- | ---------- | ---------------------------- | -------------- |
| **阶段1** | 完全理解BitVLA | 阅读论文+跑通demo                  | 模型结构图          |
| **阶段2** | 找创新入口      | 分析代码结构                       | 创新设计文档         |
| **阶段3** | 创新1实现      | Adaptive Action Tokenization | 第一个实验结果        |
| **阶段4** | 创新2实现      | Temporal History Fusion      | long-horizon实验 |
| **阶段5** | 创新3实现      | Hybrid Action Head           | 控制平滑实验         |
| **阶段6** | 实验总结       | ablation + benchmark         | 完整论文           |

---

# 二、论文三点创新结构

| 创新点                                  | 研究问题                   | 方法      | 代码改动              | 实验           |
| ------------------------------------ | ---------------------- | ------- | ----------------- | ------------ |
| **创新1** Adaptive Action Tokenization | 256 action bins 是否冗余   | 自适应分桶   | action tokenizer  | token数量vs成功率 |
| **创新2** Temporal History Fusion      | 低比特模型如何处理 long-horizon | 历史帧融合   | dataset + encoder | LIBERO-Long  |
| **创新3** Hybrid Residual Action Head  | chunk action 是否不够平滑    | 离散+连续残差 | action head       | smoothness   |

---

# 三、创新路线逻辑

三点创新其实是一条主线：

```
BitVLA
   │
   ├── Action Tokenization
   │
   ├── Temporal Memory
   │
   └── Hybrid Action Head
```

论文故事：

**低比特VLA效率很好，但动作表示、时序建模和控制输出仍有优化空间。**

---

# 四、代码修改地图

**最重要的实践指南**。

| 创新点                   | 修改文件                                | 修改内容                 | 难度  |
| --------------------- | ----------------------------------- | -------------------- | --- |
| Adaptive Tokenization | `bitvla/bitnet_action_tokenizer.py` | 自适应bin               | ⭐   |
|                       | `bitvla_transform.py`               | action encode/decode | ⭐   |
| Temporal Fusion       | `dataset loader`                    | 加历史帧                 | ⭐⭐  |
|                       | `bitvla_for_action_prediction.py`   | temporal module      | ⭐⭐⭐ |
| Hybrid Action Head    | `action_heads.py`                   | residual head        | ⭐⭐⭐ |
|                       | `openvla_utils.py`                  | action head调用        | ⭐⭐  |

---

# 五、实验设计表

**Experiment部分的框架**。

| 实验类型     | 目的   | 对比                 |
| -------- | ---- | ------------------ |
| Baseline | 基准   | BitVLA             |
| Token实验  | 动作压缩 | 256 vs 128 vs 64   |
| Memory实验 | 时序能力 | history=0,1,2      |
| Head实验   | 控制平滑 | discrete vs hybrid |
| Ablation | 模块贡献 | 去掉模块               |

---

# 六、论文方法图

**Method图**：

```
Input Images
     │
Vision Encoder
     │
Connector
     │
BitNet LLM
     │
Temporal Fusion
     │
Adaptive Action Tokenizer
     │
Hybrid Action Head
     │
Robot Control
```

---

# 七、预计实验指标

关注 **4个指标**：

| 指标                | 说明             |
| ----------------- | -------------- |
| Success Rate      | 任务成功率          |
| Inference Latency | 推理速度           |
| Action Smoothness | 动作平滑           |
| Token Length      | action token数量 |

---

# 八、论文贡献

**Contribution**

1️⃣ 提出 **Adaptive Action Tokenization**
降低 action sequence length

2️⃣ 提出 **Temporal History Fusion**
提升 long-horizon manipulation

3️⃣ 提出 **Hybrid Residual Action Head**
提高控制稳定性

---

# 九、最优执行顺序


```
1 Adaptive Tokenization
2 Temporal Fusion
3 Hybrid Action Head
```

原因：

| 创新           | 难度  |
| ------------ | --- |
| Tokenization | ⭐   |
| Temporal     | ⭐⭐  |
| Hybrid Head  | ⭐⭐⭐ |

---

# 十、最终论文故事

论文的核心逻辑是：

**Efficient Long-Horizon Vision-Language-Action Models**

具体：

```
BitVLA
  ↓
优化动作表示
  ↓
增强时序能力
  ↓
改进动作控制
```

---

# 十一、给你一个真实建议（非常重要）

现在很多VLA论文失败的原因是：

**创新点太散。**

而你现在这三点有一个统一主线：

```
Efficient Action Representation
```

# 研究路线图

Step1  跑通 BitVLA baseline
Step2  复现 LIBERO 评测
Step3  找到 long-horizon 失败案例
Step4  加 temporal memory
Step5  做 ablation
Step6  写论文

# 评价指标

| 指标                | 说明      |
| ----------------- | ------- |
| Success Rate      | 成功率     |
| Latency           | 推理速度    |
| Action Smoothness | 控制平滑    |
| Token Length      | token数量 |

# 步骤

论文训练流程其实是三阶段：

Stage1  VLM training
Stage2  Quantize-then-Distill
Stage3  Robotics Pre-training
Stage4  OFT Fine-tuning

其中：

| 阶段                    | 是否开源  | 说明         |
| --------------------- | ----- | ---------- |
| VLM training          | ❌ 未开源 | 多模态预训练     |
| Quantize-Distill      | ❌ 未开源 | 低比特蒸馏      |
| Robotics Pre-training | ❌ 未开源 | 大规模机器人预训练  |
| OFT Fine-tuning       | ✅ 已开源 | LIBERO任务微调 |

所以 README 的意思其实是：

前三个阶段没开源。

但最后一个阶段 OFT fine-tuning 是开源的。

**第一步（必须）**

先 只跑评测：

官方 checkpoint
      ↓
LIBERO evaluation

目的：

检查环境

确认仿真

确认模型加载

确认脚本

这个阶段 不需要训练。

**第二步（baseline复现）**

等评测跑通后，再：

bitvla-bf16
      ↓
OFT fine-tuning
      ↓
evaluation

这样你才有自己的 baseline。

# 模型和数据

```bash
# 1) LIBERO 环境
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# 2) 训练自己的 baseline：基础模型（二选一）
git clone https://huggingface.co/lxsy/bitvla-bf16
# 或
git clone https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16

# 3) 只做评测复现：四个官方 LIBERO 微调模型
git clone https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_spatial-bf16
git clone https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16
git clone https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_goal-bf16
git clone https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16

# 4) LIBERO 微调数据
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```