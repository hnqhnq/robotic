#### **原文的问题：**

BitVLA 的核心优势很明确：它把 VLA 主干压到了极低比特，因此模型非常轻、推理非常快。  
但它在动作端仍然沿用了 OpenVLA-OFT 风格的 **parallel chunk decoding + L1 regression**，也就是一次直接回归一段未来动作。

这类做法的优点是高效，但也有一个明显问题：

- **动作生成是“一步猜出来”的，不够细致。**  
  当任务进入接触、翻转、精细放置、遮挡干扰或长时序误差累积这些更困难的阶段时，直接回归的动作头更容易出现“整体方向对了，但关键细节不够准”的问题。

换句话说，**BitVLA 已经很好地解决了“轻量”和“速度”问题，但动作输出端还不够强**。  
如果继续一味加大主干、提高比特宽度，虽然可能涨点，但会直接破坏它最有价值的轻量优势。  
因此，一个更自然的问题是：

> **能不能不动 BitVLA 的低比特主干，只在动作关键路径上补一点精度，把动作做得更准？**

#### **我们的做法：**

我们保留 BitVLA 的 **1-bit / low-bit backbone** 不变，不去破坏它已经验证过的轻量优势；  
只在动作输出路径上做增强，让模型形成“**先快，再准**”的两阶段动作生成机制。

具体来说：

- **第一步：保留原始 BitVLA 的 coarse action prediction。**  
  模型先像 BitVLA 一样直接预测一个粗粒度动作序列（coarse action chunk），保证推理速度快、部署成本低。

- **第二步：加入轻量 LoRA Adapter。**  
  在 **ViT 后 4 层** 和 **LLM 后 4 层** 插入少量 **LoRA Adapter**，给动作相关的信息流留出一个小型高精度残差通道。  
  这样做的目的不是重训整个模型，而是只在最影响动作决策的路径上增加一点可学习容量，同时尽量不破坏低比特 backbone 的内存和速度优势。

- **第三步：增加 residual diffusion head / flow-matching head。**  
  在 coarse action chunk 的基础上，再接一个轻量的 **residual diffusion head** 或 **residual flow-matching head**，不从零生成整段动作，而是只学习一个小的动作残差，对关键时刻的动作进行细化修正。  
  也就是说：
  
  > 原始 BitVLA 负责“先给出一个快而粗的动作”，  
  > residual generative head 负责“把这个动作修得更准”。

这样，整个方法的核心思想就变成：

> **不是在整个模型里平均恢复精度，而是只沿着动作关键路径恢复精度。**

#### **预期目标：**

在尽量保持 BitVLA 轻量优势的前提下，我们希望：

- 在 **LIBERO-Long** 这类长时序任务上，提高动作稳定性和最终成功率；
- 在 **real-world OOD 跨物体泛化任务** 中，提高模型对新物体的动作适应能力；
- 在 **视觉干扰场景** 中，提高动作输出的鲁棒性；
- 在只引入很小额外参数和有限额外推理开销的前提下，取得比原始 BitVLA 更好的动作质量。

更概括地说，我们希望证明：

> **BitVLA 负责“快”，新增的 residual generative head 负责“准”。**  
> **精度不需要在整个 low-bit VLA 中均匀恢复，而应该只在动作最关键的地方恢复。**

#### **参考论文链接：**

- [BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation](https://arxiv.org/abs/2506.07530)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Fine-Tuning Vision-Language-Action Models as Generalist Robot Policies (OpenVLA-OFT)](https://arxiv.org/abs/2502.19645)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)