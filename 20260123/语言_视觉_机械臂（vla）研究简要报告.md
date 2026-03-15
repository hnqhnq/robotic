# 语言 + 视觉 + 机械臂（Vision-Language-Action, VLA）研究简要报告

语言驱动行为的研究可以追溯到早期符号主义人工智能系统 SHRDLU（Winograd, 1970），该系统通过手工规则将自然语言指令映射到符号规划与动作执行。尽管 SHRDLU 并不具备现实机器人操作能力，但其“语言—规划—行为”的基本思想为后续研究奠定了概念基础。现代 Vision-Language-Action（VLA）研究则在大规模数据与学习模型的支持下，对这一问题进行了系统化与工程化扩展。

## 一、研究背景与问题定义

**Vision-Language-Action (VLA)** 关注如何将：
    - **语言（Language）**：人类指令、任务描述、规划约束
    - **视觉（Vision）**：RGB / RGB-D / 3D / 场景表征
    - **动作（Action）**：机械臂控制、轨迹规划、操作策略

统一到一个可学习、可泛化的控制框架中，实现“**听得懂、看得见、做得对**”的机器人操作。

## 二、典型研究模式（Patterns）

**模式 P1：语言 → 任务规划 → 低层控制**
    - LLM / VLM 负责**指令理解与分解**
    - 传统或学习型控制器执行
    - 代表：Language Policies, LLM Task Planner

**模式 P2：视觉-语言联合表征 → 端到端策略**
    - 图像 + 语言 → 动作（Action Tokens / 连续控制）
    - 多用于 imitation learning / offline RL
    - 代表：PerAct, VIMA, RoboCAT

**模式 P3：多模态大模型（VLM / VLA Foundation Model）**
    - 类似 GPT-4V / InternVLA
    - 强调**跨任务、跨场景泛化**
    - 代表：InternVLA, CogACT, RT-X / Open-X

**模式 P4：生成式策略（Diffusion / World Model）**
    - 使用 Diffusion / World Model 生成动作序列
    - 代表：3D Diffusion Policy, WMPO

## 三、研究里程碑（代表性工作一览）

### 表 1.1：Vision–Language–Action（VLA）关键研究里程碑

语言驱动机器人操作的研究可以追溯到早期以符号规划为核心的模块化方法。
在这一研究范式中，自然语言主要用于描述任务目标或高层约束，随后通过语言 grounding 将其映射到符号化表示或任务规划空间，再由传统规划器或控制器执行具体动作。本文将这一类方法归纳为 P1（Language → Planning → Control）模式。

表 1.1 总结了 P1 路线中的代表性里程碑工作。这些研究在不同阶段系统性地探讨了语言理解、空间语义 grounding 以及任务与运动规划（Task and Motion Planning, TAMP）之间的结合问题。其中，早期工作（如 Tellex 等与 Paul 等）奠定了语言与符号规划之间的理论与算法基础；而随着大语言模型的发展，近期方法（如 SayCan、Code as Policies）通过引入神经语言模型，进一步增强了高层推理与任务分解能力，实现了语言生成规划或可执行程序的能力提升。

尽管 P1 方法通常依赖模块化系统设计，且在端到端泛化能力方面存在一定局限，但其在可解释性、可控性以及长时序任务分解等方面仍具有重要价值。特别是在大模型时代，P1 路线重新焕发生机，并成为语言模型与机器人系统结合的重要研究分支。因此，本文将 P1 相关工作单独列为一条子线（见表 1.1），以区别于后续以端到端学习和基础模型为主的 Vision–Language–Action 主时间线（见表 1.2）。

| 时间   | 类型     | 模式 | 工作               | 论文         | 核心贡献                |
| ---- | ------ | -- | ---------------- | ---------- | ------------------- |
| 2011 | Method | P1 | Tellex et al.    | IJRR 2011  | 语言 grounding 到符号规划  |
| 2016 | Method | P1 | Paul et al.      | RSS 2016   | 高效语言-空间概念 grounding |
| 2022 | Method | P1 | SayCan           | arXiv 2022 | LLM + 可执行性分离        |
| 2023 | Method | P1 | Code as Policies | arXiv 2023 | 语言生成程序控制机器人         |

### 表 1.2：Vision–Language–Action（VLA）关键研究里程碑
在 2024 年，生成式策略（P4）通过代表性工作（如 3D Diffusion Policy）首次系统性地验证了 Diffusion 模型在机器人操作策略学习中的可行性。然而，在 2025 年，该方向尚处于方法探索与路线分化阶段。现有工作在 Diffusion 建模方式、World Model 构建、潜变量动作表示以及长时序规划等方面呈现出多样化探索，但尚未形成被广泛接受的统一范式或标准基线。因此，本文将这些工作列为探索性代表（见表 1.3），而未将其纳入关键研究里程碑（表 1.2）。

| 时间   | 类型         | 对应模式 | 项目 / 论文标题                  | GitHub                                                                                                             | Stars | 作者                    | 论文标题                                                                                                             | 论文地址                                                                 | 数据来源                 | 预训练模型                    | 描述                                            |
| ---- | ---------- | ---- | -------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----: | --------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------- | ------------------------ | --------------------------------------------- |
| 2021 | Benchmark  | P2   | CALVIN                     | [https://github.com/mees/calvin](https://github.com/mees/calvin)                                                   |   816 | Mees et al.           | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks           | [https://arxiv.org/abs/2112.03227](https://arxiv.org/abs/2112.03227) | CALVIN Dataset       | 无                        | 语言条件长时序操作基准与评测框架                              |
| 2022 | Method     | P2   | PerAct²（双臂扩展）              | [https://github.com/markusgrotz/peract_bimanual](https://github.com/markusgrotz/peract_bimanual)                   |   113 | Grotz et al.          | Perceiver-Actor for Robotics（PerAct 系）                                                                           | [https://arxiv.org/abs/2209.05451](https://arxiv.org/abs/2209.05451) | RLBench（含双臂扩展）       | 无                        | 显式 3D 表征 + Transformer 的端到端策略（双臂设置）           |
| 2022 | Method     | P2   | VIMA                       | [https://github.com/vimalabs/VIMA](https://github.com/vimalabs/VIMA)                                               |   844 | Jiang et al.          | VIMA: General Robot Manipulation with Multimodal Prompts                                                         | [https://arxiv.org/abs/2210.03094](https://arxiv.org/abs/2210.03094) | VIMA-Bench           | 无                        | Prompt 驱动的视觉-语言-动作学习范式                        |
| 2022 | Method     | P3   | RT-1（Robotics Transformer） | [https://github.com/google-research/robotics_transformer](https://github.com/google-research/robotics_transformer) |  1700 | Brohan et al.（Google） | RT-1: Robotics Transformer                                                                                       | [https://arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817) | Google Robot Dataset | 无（论文中为视觉编码器+Transformer） | 大规模模仿学习的 Robotics Transformer 路线起点            |
| 2023 | Method     | P3   | RT-2                       | 无                                                                                                                  |     无 | Brohan et al.（Google） | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control                                    | [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) | Web + Robot Data     | PaLM-E（论文体系）             | 将 Web 级 VLM 知识迁移到机器人动作控制                      |
| 2023 | Method     | P3   | RoboCat（第三方实现）             | [https://github.com/kyegomez/RoboCAT](https://github.com/kyegomez/RoboCAT)                                         |    87 | Reed et al.（DeepMind） | RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation                                              | [https://arxiv.org/abs/2306.11706](https://arxiv.org/abs/2306.11706) | 多机器人任务数据             | 无                        | “自我改进/泛化”Foundation Agent 思路（注意：该 repo 为社区实现） |
| 2023 | Dataset    | —    | Open-X Embodiment          | [https://github.com/google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)       |  1600 | DeepMind              | Open X-Embodiment: Robotic Learning Datasets and RT-X Models                                                     | [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864) | Open-X Dataset       | 多模型                      | 统一多源机器人数据格式与模型/数据发布（数据里程碑）                    |
| 2024 | Foundation | P3   | InternVLA-M1               | [https://github.com/InternRobotics/InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1)                   |   353 | InternRobotics        | InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy                    | 无                                                                    | 多源机器人数据              | 无                        | 通用 VLA 框架（语言头+动作头、空间引导）                       |
| 2024 | Method     | P3   | CogACT                     | [https://github.com/microsoft/CogACT](https://github.com/microsoft/CogACT)                                         |   401 | Microsoft             | CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation | 无                                                                    | 无                    | 无                        | Foundation VLA 架构，强调“认知—动作”协同与专用动作模块          |
| 2024 | Method     | P4   | 3D Diffusion Policy        | [https://github.com/YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy)                 |  1200 | Ze et al.             | 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations                      | [https://arxiv.org/pdf/2403.03954](https://arxiv.org/pdf/2403.03954) | 多任务操作数据              | 无                        | Diffusion/生成式策略代表作（P4）                        |


### 表 1.3：Vision–Language–Action（VLA）关键研究里程碑

定位说明：
本表用于补充 P4（生成式策略：Diffusion / World Model）在 2025 年的代表性探索性工作。
这些工作展示了重要研究趋势，但尚未形成统一范式或社区共识，因此未纳入表 1.2 的“关键里程碑”。


| 时间   | 类型     | 模式 | 工作                                             | GitHub                                                         | 论文                    | 核心思想                                                 | 当前阶段判断            |
| ---- | ------ | -- | ---------------------------------------------- | -------------------------------------------------------------- | --------------------- | ---------------------------------------------------- | ----------------- |
| 2025 | Method | P4 | Diffusion-VLA                                  | 无统一官方仓库                                                        | arXiv 2024 / v3: 2025 | 将 VLM 表征与 Diffusion Action Policy 结合，尝试统一感知、语言与生成式控制 | 探索期，尚未成为 baseline |
| 2025 | Method | P4 | World Model Policy Optimization（WMPO 系扩展）      | [https://github.com/WM-PO/WMPO](https://github.com/WM-PO/WMPO) | arXiv 2024+           | 学习环境动力学并在模型中进行动作规划与优化                                | 成本高、复现难，未形成共识     |
| 2025 | Method | P4 | Diffusion-based Long-Horizon Manipulation（多工作） | 多个零散实现                                                         | 多篇 arXiv              | 使用 Diffusion 生成长时序动作轨迹，关注 long-horizon 任务            | 路线分散，无统一代表作       |
| 2025 | Method | P4 | Latent Action Diffusion Policies               | 无统一仓库                                                          | arXiv 2024–2025       | 在潜变量空间中进行生成式动作建模，降低控制维度                              | 理论潜力大，工程尚早        |

汇总说明：
| 表         | 作用          | 学术定位            |
| --------- | ----------- | --------------- |
| **表 1.1** | P1 路线历史里程碑  | 范式起源 / 模块化方法    |
| **表 1.2** | VLA 主时间线里程碑 | 共识级方法 / 数据 / 基准 |
| **表 1.3** | P4 探索性工作    | 前沿趋势 / 非共识      |

## 四、数据集与基准套件

### 数据与基准
| 类别（资源形态）             | 类型（研究属性）        | 名称                      | 地址                                                                                                           | 说明                                        |
| -------------------- | --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| 仿真 / Benchmark Suite | 操作基准（仿真任务集）     | Robosuite               | [https://github.com/ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite)               | 常用机械臂操作仿真与任务集合（常用于策略学习与评测）                |
| 仿真 / Benchmark       | 操作基准（语言/视觉条件操作） | RLBench                 | [https://github.com/stepjam/RLBench](https://github.com/stepjam/RLBench)                                     | 基于仿真的机械臂操作基准，广泛用于语言/视觉条件的 manipulation 评测 |
| 数据集（多源汇总）            | 语言-动作           | Open-X Embodiment       | [https://github.com/google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) | 大规模跨机器人、跨任务的语言-动作数据汇总（也包含相关模型/配套资源）       |
| 数据集                  | 语言-策略           | LanguagePoliciesDataset | [https://github.com/sstepput/LanguagePoliciesDataset](https://github.com/sstepput/LanguagePoliciesDataset)   | 面向语言→控制策略学习的数据（偏层次化/模块化策略）                |
| 数据集 + Benchmark      | 多任务 / 长时序       | CALVIN                  | [https://github.com/mees/calvin](https://github.com/mees/calvin)                                             | 长时序语言条件操作数据与评测基准（既是数据集也是 benchmark）       |

---

## 五、Awesome / 知识整理类仓库

### 表 3.1：Awesome 类项目

| 名称 | GitHub | 覆盖内容 |
|---|---|---|
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | RL + VLA 研究汇总 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 3D 机器人感知 |
| Large-VLM-based-VLA | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 大模型 VLA |

### 表 3.2：Awesome 类项目详细内容

| 仓库 | GitHub 地址 | 模块类型 | 内容模块名称 | 模块功能 | 算力等级 | 最低可行 GPU 配置 |
|----|------------|---------|-------------|----------|----------|------------------|
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 模型架构 | Monolithic Models → Single-System | 单体端到端VLA：视觉+语言直接到动作 | 高 | RTX 4090（冻结VLM） / A100（端到端） |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 模型架构 | Monolithic Models → Dual-System | 双系统VLA：理解与控制解耦 | 中 | RTX 3060 / 4060 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 模型架构 | Hierarchical Models → Planner Only | 语言/视觉规划器，不直接控制 | 低 | RTX 2060 / 3060 / 4060 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 模型架构 | Hierarchical Models → Planner + Policy | 分层VLA：规划+执行 | 中 | RTX 3060 / 4060 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 训练方法 | RL-based Methods | 使用强化学习优化VLA | 中 | RTX 3060 / 4060（离线） |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 训练方法 | Training-Free Methods | 提示/搜索/规划免训练 | 低 | RTX 2060 / 3060 / 4060 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 数据来源 | Learning from Human Videos | 人类操作视频学习 | 中~高 | RTX 4090 / A100 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 世界模型 | World Model-based VLA | 世界模型+规划 | 高 | A100 |
| Large-VLM-based-VLA-for-Robotic-Manipulation | https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation | 数据集 / 评测 | Datasets & Benchmarks → Simulation | 仿真数据集与基准 | 低 | RTX 2060 / 3060 / 4060 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 训练范式 | Offline RL-VLA | 离线数据强化学习 | 中 | RTX 3060 / 4060 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 训练范式 | Online RL-VLA | 在线交互式RL | 中~高 | RTX 4090 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 训练范式 | Test-time RL-VLA | 推理期自适应 | 低~中 | RTX 3060 / 4060 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 动作优化 | RL-VLA Action Optimization | 动作搜索与价值引导 | 低 | RTX 2060 / 3060 / 4060 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 模型资源 | Base VLA Models | 主流VLA模型索引 | 中 | RTX 3060 / 4060 |
| Awesome-RL-VLA | https://github.com/Denghaoyuan123/Awesome-RL-VLA | 数据集 / 评测 | Datasets & Benchmarks | RL-VLA数据与评测 | 低 | RTX 2060 / 3060 / 4060 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 表征 | Representations | 点云/隐式场/场景图 | 中 | RTX 3060 / 4060 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 策略学习 | Policy Learning | 基于3D的控制策略 | 中 | RTX 3060 / 4060 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 预训练 | Pretraining | 3D/多模态预训练 | 高 | RTX 4090 / A100 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 多模态模型 | VLM and LLM | LLM/VLM用于3D机器人 | 低~中 | RTX 3060 / 4060 |
| Awesome-Robotics-3D | https://github.com/zubair-irshad/Awesome-Robotics-3D | 数据集 / 仿真 | Simulations, Datasets & Benchmarks | 3D仿真与评测 | 低 | RTX 2060 / 3060 / 4060 |

## 六、当前研究趋势总结

1. **从模块化走向统一大模型（Foundation VLA）**
2. **数据规模与数据多样性成为核心瓶颈**
3. **Diffusion / World Model 成为新热点**
4. **仿真 + 真实迁移（Sim2Real）仍是关键问题**

## 七、细分方向完整补全

> 说明：以下表格为当前（2025.1.31）收集的全部仓库。

### 表 A：VLA / VLM 核心模型与方法（研究主线）

| 子方向 | 项目 | GitHub | 时间 | 是否论文 | 论文地址 | 简要说明 |
|---|---|---|---|---|---|---|
| Foundation VLA | InternVLA-M1 | https://github.com/InternRobotics/InternVLA-M1 | 2024 | 是 | 无 | 通用视觉-语言-动作基础模型 |
| Foundation VLA | CogACT | https://github.com/microsoft/CogACT | 2024 | 是 | 无 | 认知驱动的动作生成框架 |
| Diffusion Policy | 3D-Diffusion-Policy | https://github.com/YanjieZe/3D-Diffusion-Policy | 2024 | 是 | https://arxiv.org/pdf/2403.03954 | 基于3D表征的扩散动作生成 |
| Diffusion / WM | WMPO | https://github.com/WM-PO/WMPO | 2024 | 是 | 无 | World Model Predictive Optimization |
| Diffusion VLA | DiscreteDiffusionVLA | https://github.com/Liang-ZX/DiscreteDiffusionVLA | 2024 | 是 | 无 | 离散动作扩散建模 |
| MoE | HiMoE-VLA | https://github.com/ZhiyingDu/HiMoE-VLA | 2024 | 是 | 无 | 层次化 MoE 的 VLA |
| Memory | MemoryVLA | https://github.com/shihao1895/MemoryVLA | 2024 | 是 | 无 | 引入长期记忆的 VLA |
| Cache | vla-cache | https://github.com/siyuhsu/vla-cache | 2024 | 否 | 无 | 推理缓存加速 |
| Reflection | Reflect-VLM | https://github.com/yunhaif/reflect-vlm | 2024 | 是 | 无 | VLM 反思与自我纠错 |
| World + Lang | SoFar | https://github.com/qizekun/SoFar | 2023 | 是 | 无 | 世界建模 + 语言规划 |
| Multi-agent | MoLe-VLA | https://github.com/RoyZry98/MoLe-VLA-Pytorch | 2024 | 是 | 无 | 多层语言-动作推理 |

---

### 表 B：语言 → 策略 / 规划 / 控制（NLP 强相关）

| 类型 | 项目 | GitHub | 是否论文 | 说明 |
|---|---|---|---|---|
| Language Policy | LanguagePolicies | https://github.com/ir-lab/LanguagePolicies | 是 | 语言到策略映射 |
| Dataset | LanguagePoliciesDataset | https://github.com/sstepput/LanguagePoliciesDataset | 是 | 语言-策略数据 |
| Task Planner | llm-task-planner | https://github.com/kalaiselvan-t/llm-task-planner | 否 | LLM 任务分解 |
| Language Control | nl-act | https://github.com/krohling/nl-act | 否 | 语言动作映射 |
| Language Conditioned | Orion | https://github.com/Soheil-jafari/Orion-Language-Conditioned-Robotic-Manipulation | 是 | 语言条件操作 |
| Hierarchical | HULC | https://github.com/lukashermann/hulc | 是 | 语言条件层次控制 |

### 表 C：数据集 / Benchmark / 仿真套件

| 类型 | 名称 | GitHub | 说明 |
|---|---|---|---|
| Simulator | Robosuite | https://github.com/ARISE-Initiative/robosuite | 经典机械臂仿真 |
| Dataset | Open-X Embodiment | https://github.com/google-deepmind/open_x_embodiment | 大规模语言-动作 |
| Benchmark | CALVIN | https://github.com/mees/calvin | 长时序语言操作 |
| Benchmark | ARNOLD | https://github.com/arnold-benchmark/arnold | 具身智能评测 |
| Cloth | gen-cloth | https://github.com/landert-elon/gen-cloth | 布料操作 |
| Ravens | LoHo-Ravens | https://github.com/Shengqiang-Zhang/LoHo-Ravens | Ravens 扩展 |

### 表 D：VLM + 3D / 场景理解

| 项目 | GitHub | 是否论文 | 说明 |
|---|---|---|---|
| robot-3dlotus | https://github.com/vlc-robot/robot-3dlotus | 是 | 3D 场景语言操作 |
| KECRA | https://github.com/RWTH-E3D/KECRA | 是 | 知识增强机器人 |
| vlm-affordance | https://github.com/Ashutosh-cpu-glitch/vlm-affordance-demo | 否 | 可供性学习 |
| BitVLA | https://github.com/ustcwhy/BitVLA | 是 | 高效 VLA 表征 |
| RoboGround | https://github.com/ZzZZCHS/RoboGround | 是 | 具身 grounding |

### 表 E：工程系统 / Demo（非核心研究，但可落地）

| 项目 | GitHub | 说明 |
|---|---|---|
| nlp-pnp-robotic-arm | https://github.com/sahilrajpurkar03/nlp-pnp-robotic-arm | NLP 控制机械臂 |
| AI-Powered-Robotic-Arm-Control | https://github.com/HuangJunkai2023/AI-Powered-Robotic-Arm-Control | 中文语音+控制 |
| Dual-Arm-Embodied-AI | https://github.com/successfulbarrier/Dual-Arm-Embodied-AI | 双臂系统 |
| text-and-voice-controlled-robot | https://github.com/NivasPiduru/text-and-voice-controlled-robot | 文本/语音控制 |
| Vision-language-Construction-Robot | https://github.com/CarlYang-coder/Vision-language-Model-Based-Construction-Robot-System-as-Demonstrators-for-Imitation-Learning | 工程示例 |
