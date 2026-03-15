# 文献

## Sample-efficient and occlusion-robust reinforcement learning for robotic manipulation via multimodal fusion dualization and representation normalization

```bash
（1）文献标题：Sample-efficient and occlusion-robust reinforcement learning for robotic manipulation via multimodal fusion dualization and representation normalization
（2）文献类型：方法类论文（Method Paper）
（3）研究方向/归类：
- 机器人操控（Robotic Manipulation）
- 视觉强化学习（Visual Reinforcement Learning）
- 多模态表示学习（Vision + Proprioception）
- 偏 AI / 机器人交叉方向
（4）发表信息：
- 期刊：Neural Networks
- 年份：2025
- 评价体系：CCF B
- 实验平台：仿真（MuJoCo / DeepMind Control Suite）
（5）研究背景与问题：在基于视觉的机械臂强化学习中，目标物体一旦被部分遮挡，纯视觉或简单多模态融合方法容易导致表示不稳定、样本效率低，训练过程难以收敛。现有方法通常让 actor 和 critic 共享同一套多模态融合表示，在遮挡和奖励较稀疏的场景下，这种设计可能会放大 critic 学习的不稳定性，并进一步干扰策略学习。
（6）核心思想/方法概述：
在不改变强化学习算法与奖励设计的前提下，本文从多模态表示结构入手，提出两点关键改进：
a）Fusion Dualization：为 actor 和 critic 分别设计独立的多模态融合模块，而非共享同一融合表示，使二者能够根据各自的优化目标学习更合适的状态表示；
b）Representation Normalization：在表示学习过程中引入 LayerNorm 与 SimplexNorm，以提升梯度传播与特征分布的稳定性，从而改善训练稳定性与样本效率。
（7）实验设置
- 环境：DeepMind Control Suite（MuJoCo 仿真）
- 任务类型：Pushing、Lifting、Stacking（均包含视觉遮挡）
- 观测空间：
    - RGB 图像（84×84，3 帧堆叠）
    - 本体感知（关节角度与关节速度）
- 动作空间：连续关节控制指令（底层力矩由传统控制器执行）
- 奖励函数：固定设计（非本文研究重点）
（8）对比方法/Baseline：
- 视觉强化学习方法：DrQ、DrQ-v2
- 表示学习增强方法：CURL、TACO、DrM
- 同时进行消融实验，分析 fusion dualization 与 normalization 的独立贡献
（9）实验结果与结论：
在遮挡操控任务中，相比现有视觉强化学习和表示学习方法，本文方法在收敛速度、样本效率和训练稳定性方面均表现更优。实验结果表明，actor–critic 融合分离以及表示归一化在遮挡和奖励较弱的条件下能够显著提升学习效果。
（10）作者贡献总结：
本文并未直接针对稀疏奖励或探索机制提出新方法，而是通过改进多模态表示结构与训练稳定性，在既定 reward 条件下缓解了稀疏奖励带来的学习困难，属于一种结构层面的、相对克制但有效的改进。
（11）个人理解/可延伸思考：
该工作更偏向“表示与结构优化”，而非“reward 或探索机制”。其思路可与关键点表示、触觉感知或 curriculum 学习等方法结合，作为后续研究的潜在扩展方向。
```

## Multimodal Learning of Keypoint Predictive Models for Visual Object Manipulation

```bash
（1）文献标题：
Multimodal Learning of Keypoint Predictive Models for Visual Object Manipulation
（2）文献类型：
方法类论文（Method Paper，偏模型与控制）
（3）研究方向/归类：
- 机器人操控（Robotic Manipulation）
- 视觉操控（Visual Object Manipulation）
- 关键点表示学习（Keypoint-based Representation）
- 模型预测控制（Model Predictive Control, MPC）
- 扩展身体图式 / 扩展运动学链（Extended Body Schema / Extended Kinematic Chain）
- 偏机器人 / 视觉 / 控制交叉方向（弱 RL）
（4）发表信息：
- 期刊：IEEE Transactions on Robotics (T-RO)
- 年份：2023
- 评价体系：CCF B
- 实验平台：仿真 + 真实机器人（KUKA iiwa）
（5）研究背景与问题：
在视觉 MPC 框架下，操控依赖低维视觉表示、预测模型与动作优化的协同工作。然而在 object manipulation 场景中仍存在两个关键挑战：
（a）自监督训练得到的关键点不一定稳定落在被操控物体上，且难以跨时间一致地追踪同一物体部位；
（b）现有关键点预测模型多采用无结构的神经网络动力学建模，在分布外或长时域预测时容易出现误差快速累积，从而影响基于模型的视觉控制效果。
（6）核心思想/方法概述：
本文对“keypoint-based visual MPC”这一经典框架在两个关键环节进行了结构化改进：一是提升 on-object 关键点的稳定性，二是以扩展运动学链替代无结构神经网络动力学，从而实现稳定、可微的视觉预测与控制。
整体方法采用两阶段设计：
a）多模态关键点学习：在关键点检测的训练阶段引入本体感知（关节信息），通过末端位姿在图像中的投影对关键点学习施加弱几何约束，引导关键点稳定落在被操控物体上；
b）扩展运动学建模：将视觉关键点视为虚拟关节，回归其相对末端的几何参数，从而构建包含被抓取物体的扩展运动学链；
c）视觉域 MPC 控制：基于该扩展运动学模型，在关键点空间中进行模型预测控制，实现对目标视觉状态（goal image / goal keypoints）的精确操控。
（7）实验设置：
（a）实验平台：
- 机器人：7-DoF KUKA iiwa
- 设置：仿真与真机实验相结合，用于验证方法的稳定性与可落地性
（b）数据采集方式：
采集以“末端执行器与手中物体”为主要运动来源的多模态序列数据（RGB、Depth、关节状态），以减少背景干扰并提升关键点学习的稳定性。
（c）Phase 1：关键点学习与评估：
利用本体感知信息辅助训练关键点检测器，评估重点在于关键点是否稳定落在被操控物体上。论文通过统计关键点到物体中心线的距离（均值与方差）进行定量评估，结果表明多模态训练显著提升了关键点在物体上的一致性与稳定性。值得注意的是，本体感知仅在训练阶段使用，推理阶段可仅依赖视觉输入，且训练时可混合使用视觉-only 与多模态数据。
（d）Phase 2：扩展运动学建模：
将视觉关键点建模为虚拟关节，学习物体相对末端的几何参数，并验证模型在不同机械臂姿态以及重新抓取条件下的稳定性与泛化能力。
（e）下游控制与对比实验：
在基于关键点的视觉 MPC placing 任务中验证整体方法效果，并与基于神经网络的黑盒视觉动力学模型进行对比。
（8）对比方法/Baseline：
Baseline 统一在关键点空间中学习神经网络动力学模型 s_{t+1}=g(s_t,u_t)，其中 s_t=[θ_t,z_t]，并通过以下变量控制进行对比：
- 2D 关键点（无 depth、无 proprioception）
- 3D 关键点（加入 depth）
- 2D + 多模态关键点检测（训练阶段使用 proprioception）
- 3D + 多模态关键点检测
上述 baseline 用于对比无结构学习动力学与扩展运动学模型在长时预测与控制中的表现差异。
（9）实验结果与结论：
实验结果表明，相比基于神经网络学习的视觉动力学模型，本文提出的基于关键点的扩展运动学模型在长时预测中更加稳定，并在 placing 等下游视觉操控任务中取得显著更小的最终误差。论文在多个 placing 任务中使用像素空间 RMSE 进行评估（Table I），并进一步展示尽管神经网络动力学模型在训练误差上已收敛，其长时域预测误差仍会快速累积（Fig.12），而扩展运动学模型误差保持稳定，从而带来数量级上的性能提升。
（10）作者贡献总结：
本文提出了一种将关键点表示与传统运动学建模相结合的视觉操控方法，通过多模态训练提升关键点在手持物体场景下的稳定性，并利用可微的扩展运动学链实现可靠的视觉预测与模型预测控制。论文进一步在仿真与真实机器人平台上验证了该方法在多种设置下的有效性，展示了结构化模型在视觉操控任务中的显著优势。
（11）个人理解 / 可延伸思考：
该工作属于“表示与结构驱动”的操控范式，强调通过关键点与运动学结构提升稳定性与可解释性，而非依赖 reward 或探索机制进行策略学习。其方法可与强化学习方向（如 KETO、视觉 RL）形成互补，例如作为高质量状态表示或模型模块，应用于更复杂或接触丰富的操控任务中。
```


## KETO: Learning Keypoint Representations for Tool Manipulation

```bash
【文献总结】
（1）文献标题
Learning Keypoint Representations for Tool Manipulation（KETO）
- 备注：Keypoint representations for Expressive TOol manipulation
（2）文献类型
- 方法类论文（Method Paper）
- 关注表示学习与执行解耦
- 非端到端策略学习方法
（3）研究方向/归类
- 机器人操控（Robotic Manipulation）
- 工具使用与泛化操控（Tool Use & Generalization）
- 三维视觉表示学习（Point Cloud Representation）
- 自监督学习（Self-supervised Learning）
- Learning + Optimization 混合范式
（4）发表信息
- 会议：ICRA 2020
- 评价体系：CCF B
- 实验平台：PyBullet 仿真
- 传感器：RGB-D → 点云
- 硬件设置：单臂机械臂 + 标准夹爪（无特殊硬件）
（5）研究背景与问题
在机器人工具操控任务中，端到端方法通常需要大量数据、泛化能力有限且缺乏可解释性；而基于人工几何规则的方法又难以扩展到复杂或未知工具形状。
本文关注的核心问题是：是否存在一种任务相关、可学习且可泛化的中间表示，能够在不依赖人工标注的情况下，有效连接视觉感知与动作生成，从而提升工具操控的稳定性与泛化能力。
（6）核心思想/方法概述
本文提出一种“任务语义关键点表示（Task-oriented Keypoints）”来描述工具使用方式，并将学习与控制显式解耦：
- 网络从工具点云中预测少量关键点（抓取点、功能点、效果点）；
- 动作生成不由网络直接输出，而是通过基于关键点的显式优化（QP）完成；
- 关键点通过真实交互的成功/失败信号进行自监督学习，无需人工标注。
该方法强调表示优先于策略，并利用优化器保证执行稳定性。
（7）实验设置
- 任务类型：Hammering、Pushing、Reaching（三类典型工具使用任务）
- 输入：RGB-D 点云
- 输出：关键点表示 + 优化生成的抓取与操作动作
- 训练方式：纯自监督交互（binary success signal）
- 控制模块：固定，不参与学习（用于隔离表示学习效果）
（8）对比方法 / Baseline
- 端到端方法：直接从点云预测动作
- Template-based 方法：基于形状匹配拷贝关键点
- Heuristic 方法：人工几何规则设计关键点
- 并通过跨形状分布测试评估泛化能力
（9）实验结果与结论
实验结果表明：
- 在三类任务中，KETO 的成功率显著高于端到端和基于模板的方法；
- 在形状分布外测试中，关键点表示展现出更强的泛化能力；
- 不同任务下，同一工具可自动学习到不同的使用关键点，体现出任务语义一致性。
结果验证了关键点表示 + 优化执行在工具操控中的有效性。
（10）作者贡献总结
本文的主要贡献不在于硬件改进或控制策略设计，而在于：
- 提出一种任务语义明确、低维且可解释的工具表示；
- 证明该表示可以通过自监督交互有效学习；
- 展示了学习与优化解耦在提升泛化性和稳定性方面的优势。
- 整体属于表示与方法结构层面的创新。
（11）个人理解/可延伸思考
该工作体现了典型的“Learning for Representation, Optimization for Execution”思想，为后续结合强化学习、触觉感知或更复杂物理交互的研究提供了良好的结构化基础。其关键点表示也可作为更高层策略或任务规划的状态抽象。
```

## OpenVLA: An Open-Source Vision-Language-Action Model

## RT-2: Vision–Language–Action Models Transfer Web Knowledge to Robotic Control

CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation
arXiv 2024 — 推出了新的 VLA 架构设计，在多机器人和泛化性能上都有较大提升。

Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos
arXiv 2025 — 利用大规模现实生活手部视频作为 VLA 预训练数据，推动泛化与 zero-shot 能力。

ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
arXiv 2025 — 引入力觉融合模块增强 VLA 在接触丰富任务上的表现，为深度学习 + 物理反馈集成方向提供思路。

RationalVLA: A Rational Vision-Language-Action Model with Dual System
arXiv 2025 — 搭建一个带指令鲁棒性评估的 VLA 模型，能够检测并拒绝不可执行命令。

🧪 未来趋势 & Benchmarks（对论文写作很重要）

VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics
arXiv 2024 — 为 VLA / LCM（Language-Conditioned Manipulation）任务构建 benchmark，有助于评测模型。

VLATest: Testing and Evaluating Vision-Language-Action Models
arXiv 2024 — 提供 VLA 测评框架，可用于对比各种 VLA 模型。

Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation
arXiv 2312.10807 — 语言条件操控方向综述，对文献梳理极有帮助。

Vision Language Action Models in Robotic Manipulation: A Systematic Review
arXiv 2507.10672 — 深度梳理 VLA 领域近年成果、架构、数据集与趋势（含 102 个模型）。