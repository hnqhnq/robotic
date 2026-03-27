# BitVLA Action-Path Precision Rescue + AttnRes-Inspired Depth Routing — `plan.md`

> 核心目标：**不改 BitVLA 的 1-bit / low-bit backbone，不破坏它的轻量与速度优势；只沿着动作关键路径逐步加小模块，让动作更准、更稳、更抗干扰。**
>
> 新增核心想法：**吸收 AttnRes 的“按内容选择深度信息”这个好思想，但不直接把它当成整网 residual replacement 搬进 BitVLA。** 先做一个**局部的、动作条件驱动的 depth routing / depth readout 模块**，再把它接到 coarse head、residual head、flow refiner 和 gate 上。

---

## 0. 先把项目边界固定死

### 0.1 你这篇 extension 要回答的问题
BitVLA 已经证明：低比特 VLA 可以做到很轻、很快，同时在 LIBERO 上已经很强。它在动作端仍然沿用了 **parallel chunk decoding + L1 regression** 的 fast path，本质上是一次直接回归一段未来动作。这个设计高效，但在接触、翻转、精细放置、遮挡恢复、长时序误差累积这些更难的阶段，更容易出现：

- 大方向对，但关键细节不够准；
- 末端接触前动作不够细；
- gripper 开关时序不够稳；
- 后半段 drift 后修正不够好；
- 视觉干扰下 coarse chunk 不够可靠。

你要做的不是“把 BitVLA 重新做大”，而是回答：

> **能不能在不动 BitVLA 低比特 backbone 的前提下，只在动作关键路径上恢复一小部分高精度能力？**

### 0.2 更新后的总 hypothesis
这次把 AttnRes 的启发也纳入进来，主假设变成两层：

#### 第一层：Action-path precision rescue
> **BitVLA 负责快；新增的小型高精度动作通道负责准。**

#### 第二层：Depth-selective action routing
> **动作端不一定应该只读“最顶层 hidden state”；在 low-bit VLA 中，不同深度的表示对不同状态可能有不同价值。困难状态下，动作头应该有能力按内容选择性读取更合适的深度信息，而不是永远只信最后一层。**

### 0.3 对 AttnRes 的正确吸收方式
这里要非常清楚地区分两件事：

#### 可以保留的好思想
1. **uniform residual accumulation 不是唯一合理的 depth aggregation 方式**；按内容选择深度信息可能更好。
2. **Block-level depth selection** 比 full all-layer selection 更实用。
3. 小模块保持高精度是合理的，因为参数量很小，开销可控。
4. **identity-preserving initialization** 很重要，插入新模块时不能一上来把原模型行为打乱。

#### 不能直接照搬的地方
1. 原版 AttnRes 的主要验证场景是 **从头预训练的大型 PreNorm LLM**，不是“给一个已经训练好的 BitVLA 做后插模块式 fine-tuning”。
2. 原版 AttnRes 是 **backbone 级 residual replacement**；你的主线必须是 **action-path-localized**，否则会直接偏离“只修动作关键路径”的核心故事。
3. 如果你直接在整个 backbone 上改 residual，风险有三层：
   - 工程改动大；
   - latency/memory 可能上升；
   - reviewer 很容易问：你这到底是动作扩展，还是 backbone 重构？

### 0.4 本项目的主线与支线

#### 主线（必须先做）
- **Action-AttnRes-Lite / Action-conditioned Depth Routing**
- **Adapter-only precision rescue**
- **Deterministic residual head**
- **Residual latent flow-matching refiner**
- **Selective gating**

#### 支线（结果好再做）
- Residual latent diffusion 作为强对照
- Optional KD
- Dual-branch ViT+LLM depth routing
- Optional high-risk tail Block AttnRes graft

### 0.5 绝对不改的东西
1. **不改 BitVLA backbone 的 low-bit 主体。**
2. **不做 BitVLA 全流程 pretraining 复现。**
3. **不把 full-backbone Block AttnRes 作为第一主线。**
4. **所有新模块都必须支持“关掉后严格退化回原始 BitVLA fast path”。**
5. **所有实验都必须同时汇报成功率、延迟、吞吐、显存、额外参数量。**
6. **第一轮只打 LIBERO-Long。**
7. **所有新设计都先在单卡 5090 可验证的规模上成立，再考虑做大。**

---

## 1. 第一步：锁定 baseline、代码版本和实验口径

### 目标
让后面所有 ablation 都建立在一个“可重复、可比较、不会飘”的基础上。

### 具体要做
1. 记录下面这些版本信息到 `docs/experiment_manifest.md`：
   - BitVLA repo commit
   - OpenVLA-OFT repo commit
   - Python / PyTorch / CUDA / xFormers / Flash-Attention / bitsandbytes 版本
   - checkpoint 名称
   - LIBERO 数据路径
   - 训练随机种子列表
2. 明确 baseline 只用下面这几个：
   - `BitVLA official checkpoint eval`
   - `BitVLA official finetune script on LIBERO-Long`
   - 之后所有方法都从这条分支扩展
3. 提前定好主表格字段：
   - `Long success`
   - `Average success`
   - `Latency (avg / p95)`
   - `Throughput`
   - `Peak VRAM`
   - `Trainable params`
   - `Checkpoint size increase`
   - `Refinement ratio`
   - `Depth-routing overhead`
4. 在 `docs/compare_protocol.md` 里明确：
   - eval episodes 数量
   - 采样方式
   - action chunk 配置
   - 推理精度（bf16/fp16 等）
   - 是否 warmup

### 通过标准
- 你能随时回答：**“我现在这个结果到底在跟谁比？”**
- 你不会在中途悄悄改评测口径。

---

## 2. 第二步：搭环境，只走官方支持的最短路径

### 目标
完全照着官方 repo 跑通，不在环境问题上浪费研究时间。

### 具体要做
1. 按 BitVLA README 的 OFT 路线装环境。
2. 安装顺序建议固定：
   - conda 环境
   - PyTorch / torchvision / torchaudio
   - `openvla-oft/`
   - `transformers`
   - `LIBERO`
   - `bitvla/`
3. 单独写一个 `docs/setup_log.md`，记录：
   - 报错位置
   - 修复方式
   - 最终可复现安装命令
4. 提前验证：
   - `import torch`
   - `import libero`
   - `import transformers`
   - `import bitvla`

### 最小目录建议
```text
project_root/
  repos/
    BitVLA/
    LIBERO/
    openvla-oft/
  data/
    modified_libero_rlds/
  checkpoints/
    bitvla/
    baselines/
  outputs/
    baseline/
    depthmix/
    action_attnres/
    adapter/
    det_residual/
    flow/
    diffusion/
    gated/
    risky_backbone_branch/
  docs/
    experiment_manifest.md
    compare_protocol.md
    setup_log.md
    smoke_test.md
  scripts/
  models/
```

### 通过标准
- 官方 sample / eval script 能跑起来。
- 你能一条命令进入正确环境并启动 eval。

---

## 3. 第三步：下载 checkpoint 和数据，明确“从哪儿开始”

### 目标
不要从头训练任何重模型，直接站在作者释放的 checkpoint 上开始。

### 具体要做
1. 下载：
   - 官方推荐的 **new pre-trained BitVLA model**
   - 官方 released 的 LIBERO fine-tuned checkpoint（尤其是 `libero_long`）
   - `openvla/modified_libero_rlds`
2. 建一个 `docs/assets.md`，写清楚：
   - checkpoint 文件名
   - 本地路径
   - 对应任务套件
   - 是否需要转换格式
3. 如果旧模型需要转换，先跑 `convert_ckpt.py`；否则一律优先用作者推荐的新模型。
4. 第一阶段不要折腾真实离线 1.58-bit 部署打包，先用作者给的 master weights + online quantization 做算法验证。

### 通过标准
- 所有 checkpoint 都能被脚本正确加载。
- 所有路径都在配置文件里，不要散落在命令行历史里。

---

## 4. 第四步：先跑官方 inference smoke test

### 目标
确认输入管线、checkpoint、模型加载和 action 输出没有错位。

### 具体要做
1. 用官方 eval / inference 脚本跑一条样本：
   - 读入 observation
   - 生成一个 action chunk
   - 打印 shape、数值范围、gripper 维度
2. 记录到 `docs/smoke_test.md`：
   - 输入 image shape
   - proprio shape
   - 输出 action chunk shape
   - 每一维动作的 min / max / mean
   - 单次推理时间
3. 可视化 10 个 action chunk：
   - 是否全是 0
   - 是否爆掉
   - gripper 是否永远不变
4. 固定一个小的 `smoke_seed_list.txt`，以后每次大改模块都先复查。

### 通过标准
- action chunk 维度正确。
- 数值范围正常。
- 没有 NaN / Inf。
- 同一输入多次前向，输出一致。

---

## 5. 第五步：复现 baseline eval，只打 LIBERO-Long

### 目标
先拿到一个“你自己机器上”的 baseline 结果，后面所有模块都和它比。

### 具体要做
1. 先只跑 `libero_long` 的 official eval。
2. 固定：
   - seed
   - eval episodes
   - checkpoint
   - 设备
   - dtype
3. 记录：
   - success rate
   - 单 episode 平均步数
   - failure video 列表
   - latency / throughput / VRAM
4. 保存：
   - `outputs/baseline/baseline_eval_long.json`
   - `outputs/baseline/baseline_eval_long.md`
   - `outputs/baseline/videos/`

### 重点
- 第一轮不要追 paper 数字一模一样。
- 你真正需要的是：**自己机器上的“相对可比 baseline”。**

### 通过标准
- 你手上有一个固定 baseline 数值，后面每个模块都能和它比较。
- 你能导出成功 / 失败 rollout 视频。

---

## 6. 第六步：复现一条 baseline 训练线，至少做短跑 + 一次完整跑

### 目标
证明你不只是会 eval，还能在这个代码库里训练东西。

### 具体要做
1. 先跑官方 `ft_bitvla_libero_long.sh` 的 **debug 短跑**：
   - 500~1000 step
   - 只看 loss 是否下降
   - 看 checkpoint 是否能存
   - 看 eval 是否能跑
2. 再跑一次 **完整 baseline finetune**：
   - 不求 paper 数字复刻到百分位
   - 求训练曲线合理、结果可复现
3. 把 debug 和 full run 的：
   - 配置
   - 日志
   - ckpt
   - eval 结果
   都存下来
4. 形成一个 `baseline_retrain_reference/` 目录，后面所有模块共用。

### 通过标准
- 你已经验证：自己可以在这个 repo 上完成训练 → 保存 ckpt → eval 的闭环。

---

## 7. 第七步：先加“研究仪表盘”，再加模型模块

### 目标
先把分析工具搭好，否则后面涨点了也不知道为什么涨。

### 你必须先补的 logging

#### 7.1 动作级 logging
- `a_coarse`
- `a_final`
- `a_gt`
- `|a_coarse - a_gt|`
- `|a_final - a_gt|`
- `delta_pred`
- `delta_gt`

#### 7.2 维度级 logging
- translation 维误差
- rotation 维误差
- gripper 维误差

#### 7.3 时间级 logging
- 每个 episode 每一步误差
- subgoal boundary 前后的误差
- gripper open/close 切换前后的误差
- late-stage precision 区间误差

#### 7.4 效率 logging
- avg latency
- p95 latency
- throughput
- max allocated VRAM
- trainable params
- extra checkpoint size

#### 7.5 Depth-routing 相关 logging（新增）
- `alpha_depth`（每个深度源的权重）
- `depth_entropy`
- `top_only_action` vs `depth_mixed_action` 的差异
- `depth_shift_score`（相较于 top-layer-only，是否明显向更浅层偏移）
- hard-state 中 depth 权重的平均分布

#### 7.6 可视化
- 成功 / 失败视频
- coarse vs refined 动作曲线
- residual magnitude 曲线
- gate activation 曲线
- depth attention 随 episode step 的变化图

### 推荐直接建的脚本
```text
scripts/
  eval_libero_long.py
  eval_efficiency.py
  dump_rollouts.py
  analyze_action_error.py
  analyze_gate_curve.py
  analyze_depth_curve.py
  analyze_depth_entropy.py
  build_hard_subset.py
```

### 通过标准
- 你已经不是只看 final success。
- 你能回答“改进主要发生在什么阶段、什么动作维度、什么状态”。

---

## 8. 第八步：先定义 hard-state 子集

### 目标
你的方法如果主要在困难状态涨点，这不是坏事，而是论文故事的核心。所以你必须提前把困难状态定义出来。

### 第一版 hard-state 定义（不依赖额外标注）
建议先用 rule-based heuristic：
1. **gripper transition**
   - gripper 维度在 chunk 内发生明显开/关切换
2. **high-curvature action**
   - chunk 内动作方向快速变化
3. **late-stage precision**
   - episode 后半段动作
4. **recovery-like states**
   - 某一步后误差突然增大，接下来 5~10 步仍在修正
5. **large residual states**
   - `|a_gt - a_coarse|` 特别大的样本
6. **depth-ambiguous states（新增）**
   - depth entropy 很高
   - 或 top-only / depth-mixed action disagreement 很大

### 第二版 hard-state 定义（如果能从 env 拿更多信息）
1. 接触前 10 步
2. grasp 后 10 步
3. place 前 10 步
4. occlusion 发生时段
5. object pose 被扰动时段

### 通过标准
- 你能单独报告：
  - 全量结果
  - hard-state 结果
  - easy-state 结果

---

## 9. 第九步：先搭一个“模块化可插拔外壳”

### 目标
让所有新模块都变成开关，而不是把官方代码改得面目全非。

### 必须做的工程重构
1. 在 config 里加入这些 flag：
```yaml
use_static_depth_mix: false
use_action_attnres: false
use_dual_depth_attn: false
use_adapter: false
use_det_residual_head: false
use_flow_refiner: false
use_diffusion_refiner: false
use_gate: false
use_depth_uncertainty_gate: false
use_action_kd: false
use_tail_block_attnres: false   # 高风险支线，默认永远 false
```
2. 所有扩展都必须满足：
   - `all flags = false` 时，输出与 baseline 一致
3. 把新增模块统一放到一个目录：
```text
models/act_refine/
  depth_tap.py
  static_depth_mix.py
  action_attnres.py
  fusion.py
  adapters.py
  det_residual_head.py
  latent_encoder.py
  flow_refiner.py
  diffusion_refiner.py
  gate_head.py
  losses.py
```
4. 给 backbone 留统一接口：
```python
outputs = backbone(obs, lang, return_hidden=False, return_multidepth=False)
```
最少返回：
- top hidden states
- instruction summary / cls-like summary
- action head 输入 hidden
- optional tapped multi-depth summaries

### 通过标准
- 你能一条命令切换 ablation。
- 不会因为 copy 10 份训练脚本把自己搞乱。

---

## 10. 第十步：先打通 multi-depth tap + online summary 接口

### 目标
在**不改 backbone 主体逻辑**的情况下，把 AttnRes 的“多深度可读”思想变成一个工程上可用的接口。

### 为什么这一步必须先做
你后面的 Action-AttnRes-Lite、gate、residual refiner 都需要多深度输入。如果你一上来让模型 `output_hidden_states=True` 返回所有层：
- 显存可能上升；
- 速度可能下降；
- 某些 fused kernel 路径可能失效。

因此，**优先做 targeted hook + online summary**。

### 10.1 先只 tap 哪些层
#### MVP：LLM-only
- 只 tap **最后 4 层** LLM hidden states；
- 如果最后 4 层太相似，再改成 **最后 8 层聚成 4 个 block**。

#### 第二版：Dual-branch
- 在 LLM 之外，再 tap **ViT 最后 4 个 block**；
- 但这一版必须等 LLM-only 先稳定。

### 10.2 如何 tap
优先顺序：
1. **forward hooks** 只挂在目标层/目标 block；
2. 只缓存你真正需要的 summary，不缓存所有 token 的完整 hidden；
3. 如果某个实现里 `output_hidden_states=True` 不影响速度且简单，再用它做 debug 版本，但正式实验最好切回 hook 版。

### 10.3 online summary 怎么做
你需要两种 summary 模式：

#### 模式 A：全 chunk 共享 summary（先做）
```python
s_i = mean_pool(h_i[:, action_token_positions, :])   # [B, D]
```
优点：
- 简单
- 显存低
- 足够先验证“多深度是否有用”

#### 模式 B：每个 action slot 独立 summary（第二阶段）
```python
s_i = h_i[:, action_token_positions, :]              # [B, K, D]
```
优点：
- 更细，可能更适合 chunk 内不同 future step 的不同需求
- 更适合精细动作与 gripper 时序

### 10.4 降维建议
如果直接存 `[B, K, D]` 太大，可以先加一个很小的投影：
```python
s_i_small = W_tap * s_i    # D -> 256
```
推荐默认：
```yaml
tap_dim: 256
tap_mode: chunk_shared   # first
llm_tap_layers: last_4
vit_tap_blocks: none     # first
```

### 10.5 detach 策略
#### Phase 1：depth module only，backbone frozen
- tapped summaries 可以 `detach()`，节省显存；
- 因为这时你只训练 depth selector / readout 小模块。

#### Phase 2：加 adapter 后
- 对带 adapter 的那些 tapped 层，**不要 detach**；
- 否则 adapter 学不到通过 multi-depth path 回传的梯度。

### 10.6 必做 sanity check
1. hook 前后，baseline 输出一致
2. 开启 taps 但不启用任何新模块时，指标不变
3. `tap_cache` 的 shape、dtype、device 全正确
4. latency overhead 单独测一遍

### 通过标准
- 你已经能稳定拿到 selected depths 的小 summary，而不是暴力返回全层 hidden。

---

## 11. 第十一步：先做一个最便宜的对照 —— Static Depth Mixing

### 目标
先回答一个更基础的问题：

> **到底是“任何多深度读出”都有帮助，还是只有 AttnRes 式 input-conditioned depth selection 才有帮助？**

### 具体做法
固定多深度 sources 后，先做三个极简对照：

1. **Top-only**
   - 只用最后一层（原始 baseline）
2. **Uniform Avg**
   - 对 tapped layers 做平均
3. **Learned Static Mix**
   - 学一个输入无关的标量权重 `w_i`
   - `alpha = softmax(w)`
   - `h_depth = sum_i alpha_i s_i`

### 融合方式
```python
h_fused = h_top + gamma_depth * Proj(h_depth)
```
其中：
- `gamma_depth` 初始化为 `0`
- 保证模块插入时退化为 baseline

### 你要回答的核心问题
- 只要多读几层，是不是就能涨？
- 如果 static mix 就已经很强，说明“多深度信息本身”很有价值；
- 如果必须要 action-conditioned 才明显涨，说明 AttnRes 的“按内容选深度”确实有必要。

### 通过标准
- 这一步至少给你一个方向：
  - `static mix` 几乎没用 → 重点做 content-conditioned depth routing
  - `static mix` 已经有效 → 后面 Action-AttnRes-Lite 只需要做更精细的版本

---

## 12. 第十二步：做 Module S —— Action-AttnRes-Lite（主线新增模块）

### 目标
吸收 AttnRes 的最好思想：**按内容选择深度信息**，但只作用在**动作读出路径**，不去替换整个 backbone residual。

### 12.1 命名纪律
代码里可以叫：
- `action_attnres`
- `depth_routing`

论文里不要说“我们直接把 AttnRes 搬进 BitVLA”。
更稳妥的说法是：

> **受 AttnRes 启发，我们设计了一个 action-conditioned depth routing module，使动作路径能够按内容选择性聚合多深度表示。**

这样既诚实，也更容易形成你自己的方法身份。

### 12.2 为什么它比直接 backbone Block AttnRes 更适合你
1. **更符合项目主 hypothesis**：只修动作路径。
2. **更适合 post-hoc fine-tuning**：不需要重写每层 residual。
3. **更容易保住 BitVLA 的速度优势**：只在 readout 端做小模块。
4. **更容易在 5090 上快验证**。

### 12.3 MVP 设计：LLM-only Action-AttnRes-Lite
先只做 LLM 深度路由：

#### 输入源
- `s_1 ... s_N`：来自 selected LLM depths 的 summaries
- `h_top`：最后一层 action readout hidden
- `inst_summary`
- `proprio_proj`
- 如果已经有 coarse chunk，也可以拼 `a_coarse`

#### Query 设计
第一版：
```python
q = MLP([LN(h_top), inst_summary, proprio_proj])
```
第二版（给 residual refiner 用）：
```python
q_ref = MLP([LN(h_top), inst_summary, proprio_proj, a_coarse])
```

#### Key / Value
```python
k_i = W_k s_i
v_i = W_v s_i
alpha_i = softmax(q @ k_i / sqrt(d) / tau_depth)
h_depth = sum_i alpha_i * v_i
```

#### 融合
```python
h_fused = h_top + gamma_depth * Proj(h_depth)
```
推荐默认：
```yaml
depth_hidden_dim: 256
depth_tau: 1.0
depth_fusion: residual_add
gamma_depth_init: 0.0
```

### 12.4 Identity-preserving 初始化
这一步很关键。你的初始化不能把 baseline 打乱。

推荐同时做两件事：
1. `gamma_depth = 0`
2. 对 `alpha` 的 bias 让它初始偏向 top layer/block

这样：
- 一开始几乎就是 baseline；
- 训练过程中再慢慢学会在 hard states 借其他深度。

### 12.5 第二版：per-action-slot depth routing
如果 chunk-shared 版本有效，再做更细版本：
```python
q_k = q[:, k, :]      # 每个 chunk slot 一个 query
alpha_{k,i} = softmax(q_k @ k_i)
```
这样可以回答一个更细的问题：

> **chunk 内不同 future action slot 是否会偏好不同深度？**

这对 gripper 时序和 late-stage precision 可能很有帮助。

### 12.6 第三版：Dual-branch depth routing（ViT + LLM）
当 LLM-only 路线稳定后，再做：
- 一套 `alpha_llm` over LLM block summaries
- 一套 `alpha_vit` over ViT block summaries
- 最后融合：
```python
h_depth = Fuse(h_depth_llm, h_depth_vit)
```

#### 为什么值得做
- occlusion / texture / local geometry 可能更依赖中后层视觉表示
- instruction / task phase 可能更依赖语言侧深层表示

### 12.7 可加但不要一开始就加的正则
#### 低熵正则（鼓励选择性）
```text
L_ent = lambda_ent * H(alpha)
```
只在 alpha 太平时再上，不要第一版就加。

#### top-layer anchor 正则
```text
L_anchor = lambda_anchor * ||h_fused - h_top||
```
当模块太容易飘时再加。

### 12.8 Action-AttnRes-Lite 的最关键 ablation
1. top-only
2. uniform avg
3. learned static mix
4. Action-AttnRes-Lite (LLM-only)
5. Action-AttnRes-Lite (per-slot)
6. Action-AttnRes-Lite (dual-branch)

### 12.9 你必须观察的现象
1. hard-state 中是否更偏向非顶层表示
2. gripper transition 前后 depth 权重是否变化
3. occlusion / recovery-like states 中 depth entropy 是否升高
4. top-only vs depth-mixed action disagreement 是否能预测失败

### 12.10 通过标准
- `BitVLA + Action-AttnRes-Lite` 至少在 Long 或 hard-state 指标上有正信号；
- latency 增幅很小；
- 你已经可以证明：**不是所有深度都一样，动作路径确实受益于按内容选深度。**

---

## 13. 第十三步：做 Module A —— Adapter-only precision rescue

### 目标
先验证：**只在动作关键路径附近加一点可学习高精度容量，是否已经能提升。**

### 你的具体设计
1. **冻结整个 BitVLA backbone 主体**
2. 只在以下位置加 LoRA / Adapter：
   - ViT 最后 4 层
   - LLM 最后 4 层
3. 第一版默认超参：
```yaml
adapter_type: lora
adapter_rank: 8
adapter_alpha: 16
adapter_dropout: 0.05
vit_layers: last_4
llm_layers: last_4
```
4. module name 不统一时：
   - 优先打 attention output / MLP up-down projection
   - 实在不行就自写 low-rank delta wrapper 包住 `Linear`

### 和 Action-AttnRes-Lite 的关系
你不能直接跳到“Adapter + Action-AttnRes-Lite + Residual + Flow”。
建议顺序固定：
1. `BitVLA`
2. `BitVLA + Action-AttnRes-Lite`
3. `BitVLA + Adapter`
4. `BitVLA + Action-AttnRes-Lite + Adapter`

这样你才能回答：
- 是 depth routing 在起作用？
- 还是 adapter 容量在起作用？
- 两者是否互补？

### 必做 sanity checks
1. 统计 trainable params 占比
2. 128 个样本 overfit 测试
3. 打开/关闭 adapter，验证 forward path 不坏
4. eval 一条 rollout，看动作没有整体飘掉
5. 和 Action-AttnRes-Lite 一起开时，确认 hook / grad 路径正常

### 通过标准
- `BitVLA + Adapter` 能稳定训练；
- Long 不一定立刻大涨，但不能大退步；
- 参数量和延迟增幅必须很小。

---

## 14. 第十四步：做 Module B —— deterministic residual head

### 目标
先证明“动作侧多一点容量 + 更好的 depth readout”有效，再引入 generative head。

### 设计原则
保留原始 coarse action head：
```python
a_coarse = H_coarse(h_readout)
```
其中 `h_readout` 可以是：
- `h_top`
- `h_fused`（如果开了 Action-AttnRes-Lite）

再加一个小 residual head：
```python
delta = H_res(h_readout, a_coarse, proprio, inst_summary)
a_final = a_coarse + delta
```

### 第一版默认结构
1. 输入：
   - `h_readout`
   - `proprio_proj`
   - `a_coarse`
   - `inst_summary`
   - optional: `depth_entropy`
2. 网络：
   - 2~3 层 MLP
   - hidden dim 512 或 1024
3. 输出：
   - `delta_a`，shape 与 `a_coarse` 一致

### loss
```text
L_total = L_huber(a_final, a_gt)
        + lambda_res * L2(delta_a)
        + lambda_aux * L_huber(a_coarse, a_gt)   # optional
```
推荐起点：
```yaml
proposal_loss: huber
lambda_res_reg: 1e-3
lambda_aux_coarse: 0.3
```

### 你要验证的核心问题
- 加少量动作侧容量后，哪些维度最受益：translation / rotation / gripper？
- Action-AttnRes-Lite 是否能让 deterministic residual 更稳？
- 提升是否集中在 hard states？

### 通过标准
- 如果 deterministic residual 都不 work，后面的 generative refiner 很可能只是更复杂地做无效工作；
- 因此这一步必须认真做干净。

---

## 15. 第十五步：明确 generative 主线，只选一个作为“主稿主角”

### 目标
不要一上来同时做 diffusion 和 flow-matching 两条主线，不然工程爆炸、故事分散。

### 推荐决策
- **主线推荐：Residual Latent Flow Matching**
  - 更适合少步推理
  - 更容易做 1-step / 2-step / 4-step 的速度-精度 tradeoff
  - 很适合你“快 + 准”的故事
- **备选/对照：Residual Latent Diffusion**
  - 更贴近 Diffusion Policy 文献
  - 作为强对照或备份

### 执行原则
1. 先做 deterministic residual
2. 再做一个 generative 主线
3. 最后再做另一个 generative 作为对照

---

## 16. 第十六步：先实现 shared latent residual interface

### 目标
无论你选 flow 还是 diffusion，都先把动作残差统一编码到 latent 空间里，降低采样成本。

### 具体做法
1. 定义 residual target：
```python
delta_gt = a_gt - a_coarse.detach()
```
2. 定义 latent encoder / decoder：
```python
z0 = E(delta_gt)      # latent dim = 64 first
delta_hat = D(z_hat)
a_refined = a_coarse + delta_hat
```
3. 第一版默认：
```yaml
latent_dim: 64
latent_encoder: mlp
latent_decoder: mlp
action_norm: dataset_stats
```
4. 所有 generative head 都只在 `z` 上工作，而不是直接在 `K x D_a` 上做 full generative。

### 必做检查
- latent encoder / decoder 单独训练能否重建 `delta_gt`
- 如果 autoencoding 本身都不行，不要急着上 flow / diffusion

### 通过标准
- latent reconstruction loss 合理下降；
- 用 `D(E(delta_gt))` 得到的 residual 已经能逼近 `delta_gt`。

---

## 17. 第十七步：做主线 Module C —— residual latent flow-matching refiner

### 目标
让模型学会：**不是从零生成整段动作，而是只修 coarse chunk 的残差分布。**

### 第一版最简实现
1. 条件输入：
   - `h_readout`（最好用 `h_fused`）
   - `proprio_proj`
   - `inst_summary`
   - `a_coarse`
   - optional `depth_entropy`
2. 训练：
   - `z0 = E(delta_gt)`
   - 采样 `z1 ~ N(0, I)`
   - `t ~ U(0, 1)`
   - `z_t = (1 - t) * z0 + t * z1`
   - 目标 velocity：`v* = z1 - z0`
   - 网络预测 `v_theta(z_t, t, cond)`
3. 推理：
   - 从 `z1 ~ N(0, I)` 开始
   - Euler / Heun 积分 1、2、4 步
   - 得到 `z_hat0`
   - `a_ref = a_coarse + D(z_hat0)`

### 第一版默认超参
```yaml
flow_hidden_dim: 512
flow_num_layers: 4
latent_dim: 64
flow_solver_steps_eval: [1, 2, 4]
lambda_refine: 1.0
lambda_residual_reg: 1e-3
```

### loss
```text
L_flow = ||v_pred - v_target||^2
L_ref  = Huber(a_ref, a_gt)
L_all  = L_flow + lambda_ref * L_ref + lambda_reg * ||delta_hat||^2
```

### 训练顺序
1. 先 always-on，不加 gate
2. 先 1-step 看是否有收益
3. 再试 2-step / 4-step
4. 保留最小有效步数，后面拿去做 gating

### 必做 sanity checks
1. 在 very small split 上过拟合
2. 观察 residual magnitude 是否合理
3. 看 refined action 是否总是在 over-correct
4. 看 gripper 维是否异常放大
5. 看不同 depth routing 版本对 flow 的影响

### 通过标准
- `Action-AttnRes-Lite + Adapter + deterministic residual + flow(always-on)` 在 Long 上优于前一版；
- 即使全量 success 涨不多，hard-state error 也应下降。

---

## 18. 第十八步：做 Module D —— residual latent diffusion refiner（强对照或备份）

### 目标
给论文一个清晰对照：提升到底来自“residual generative refinement”本身，还是来自某个特定 flow 公式。

### 什么时候做
- 如果 flow-matching 已有效：把 diffusion 当对照
- 如果 flow-matching 不稳定：把 diffusion 当备胎主线

### 最小实现
1. 仍然使用相同 latent interface：
   - `z0 = E(delta_gt)`
2. 训练：
   - 标准 DDPM / VP-style latent diffusion
   - train timesteps 先 20
3. 推理：
   - 只做 DDIM 2-step / 4-step
   - 不要一上来 20-step

### 第一版默认超参
```yaml
diff_train_steps: 20
diff_eval_steps: [2, 4]
latent_dim: 64
beta_schedule: cosine
lambda_refine: 0.5
```

### 通过标准
- 结果至少能和 flow-matching 形成清晰比较：
  - 哪个更快
  - 哪个更稳
  - 哪个更适合 gated selective refinement

---

## 19. 第十九步：做 Module E —— gate / selective refinement

### 目标
把“always-on generative refinement”的增益保住，同时把平均开销拉回去。

这是你“快 + 准”故事里最关键的一步。

### 19.1 gate 的输入
建议和 refiner 共享 condition：
- `h_readout`
- `proprio`
- `a_coarse`
- optional `delta_det`
- `depth_entropy`
- `top_only` vs `depth_mixed` action disagreement
- hard-state heuristic flags

### 19.2 gate label 第一版怎么造
先不要纠结完美标签，先做能跑的版本。推荐顺序：
1. **基于 coarse error 的 oracle label（训练时）**
   - 如果 `|a_coarse - a_gt| > tau_err`，标 1
2. **基于 residual norm**
   - 如果 `|delta_gt| > tau_delta`，标 1
3. **基于 gripper transition**
   - gripper 切换强制标 1
4. **基于 depth ambiguity（新增）**
   - 如果 `depth_entropy` 高于阈值，标 1
   - 或 top-only / depth-mixed disagreement 大于阈值，标 1
5. **基于 teacher disagreement**（可选）
   - 后面加 KD 再用

### 19.3 推理逻辑
```python
if gate_prob < tau:
    action = a_coarse_or_det
else:
    action = a_refined
```
这里有两种模式：
- `coarse -> refined`
- `det_residual -> flow_refined`

建议先做第二种：
> coarse 先过 deterministic residual，真正 expensive 的 flow 只给 hard states。

### 19.4 你要调的不是一个 tau，而是一条 Pareto 曲线
至少报告：
- `tau = low`
- `tau = medium`
- `tau = high`

分别对应：
- 更多 refine、更高成功率、更高开销
- 中间点
- 更少 refine、更低开销

### 19.5 关键指标
- refinement trigger ratio
- success rate
- avg latency
- p95 latency
- hard-state success
- depth-uncertainty correlation

### 通过标准
- gated 版本尽量接近 always-on 成功率；
- 平均 latency 明显低于 always-on；
- 你能证明：**额外计算主要被投到 hard states，而不是平均撒在所有状态上。**

---

## 20. 第二十步：可选 Module F —— action-aware KD（只在前面涨点不够时加）

### 目标
如果 `depth routing + adapter + residual + flow` 还不够强，再引入 teacher signal，但它不是第一阶段必做。

### 建议 teacher 顺序
1. **最快验证版**：BF16 BitVLA / released checkpoint
2. **更强版**：OpenVLA-OFT
3. **后续强版**：多 teacher 或 trajectory selection

### 最好先离线缓存的 teacher 信息
1. teacher coarse action chunk
2. teacher final action chunk（如果有）
3. teacher top hidden summary
4. teacher selected token hidden（只选动作相关位置）

### KD loss 建议
```text
L_act_kd = Huber(a_student, a_teacher)
L_hid_kd = MSE(h_student_sel, h_teacher_sel)
```
不要第一轮就上 attention map KD。

### 一个更聪明的 KD 方向（后续可做）
如果你真的想把 AttnRes 灵感再往前推一步，可以尝试：
- 用 teacher 的多个深度 summary 训练一个“更合理的 depth selector”
- 但这一定是后续工作，不是第一轮 MVP

### 执行原则
- 只有在前面模块都 work 但涨幅还不够时才加 KD
- 不要让 KD 把主线从“动作路径 refinement”变成“teacher distillation 工程”

---

## 21. 第二十一步：loss、训练策略、稳定性细节要一步步加，不要全开

### 目标
把训练稳定下来，别让模型在最后一步被小细节搞崩。

### 推荐的 loss 加入顺序
1. baseline 原始 loss
2. `+ static depth mix`
3. `+ Action-AttnRes-Lite`
4. `+ adapter`
5. `+ final action Huber`
6. `+ residual regularization`
7. `+ flow / diffusion loss`
8. `+ gate loss`
9. `+ optional KD`

### 默认训练建议
```yaml
optimizer: adamw
weight_decay: 0.01
grad_clip: 1.0
amp: bf16
ema: true
ema_decay: 0.999
grad_accum: as_needed
```

### depth routing 稳定性建议
1. `gamma_depth` 先初始化为 0
2. 如果 alpha 太平：
   - 减小 temperature
   - 再考虑轻微 entropy regularization
3. 如果 alpha collapse 到 top layer：
   - 先去掉 entropy penalty
   - 再调大学习率或延迟解冻 adapter
4. 如果 depth module 导致动作整体漂移：
   - 加 top-layer anchor 正则
   - 或先只让它服务 residual head，不服务 coarse head

### 动作损失细节建议
1. translation / rotation / gripper 可以分开看 loss
2. grasp / ungrasp 特别不稳时，单独加 gripper weight
3. 最终 action 记得 clip 到 dataset 范围

### 必做训练检查
- coarse loss 和 final loss 是否同时下降
- residual norm 是否发散
- gate 是否 collapse 到全 0 或全 1
- depth entropy 是否全程不变

---

## 22. 第二十二步：实验必须按“从小到大”的顺序打，不要直接全 benchmark

### 目标
先拿最小正结果，再扩展。

### 推荐实验顺序
1. `BitVLA baseline`
2. `+ Static Depth Mix`
3. `+ Action-AttnRes-Lite (LLM-only)`
4. `+ Adapter`
5. `+ Action-AttnRes-Lite + Adapter`
6. `+ DetResidual`
7. `+ Flow(always-on)`
8. `+ Flow(gated)`
9. `+ Diffusion(always-on)`（对照）
10. `+ Dual-branch depth routing`（如果前面已经很稳）
11. `+ KD`（如果必要）
12. `+ Risky tail Block AttnRes graft`（最后再说）

### stop / go 规则
- 如果某模块在 Long 单 seed 上都没有任何正信号，不要急着扩全 benchmark
- 先看：
  - hard-state error
  - rollout qualitative
  - latency
  - 训练稳定性
  - depth weight 行为是否合理
- 只有有正信号，才扩到多 seed 和全 benchmark

---

## 23. 第二十三步：主 ablation matrix 提前定死

### 目标
不要等结果出来再临时想 ablation。

### 主 ablation 列表
1. `BitVLA`
2. `BitVLA + UniformAvgDepth`
3. `BitVLA + StaticDepthMix`
4. `BitVLA + Action-AttnRes-Lite (LLM-only)`
5. `BitVLA + Adapter`
6. `BitVLA + Action-AttnRes-Lite + Adapter`
7. `BitVLA + Action-AttnRes-Lite + Adapter + DetResidual`
8. `BitVLA + Action-AttnRes-Lite + Adapter + Flow(always-on)`
9. `BitVLA + Action-AttnRes-Lite + Adapter + Flow(gated)`
10. `BitVLA + Action-AttnRes-Lite + Adapter + Diffusion(always-on)`（可选）
11. `BitVLA + Dual-branch Depth Routing + Adapter + Flow(gated)`（可选）
12. `BitVLA + ... + KD`（可选）

### 局部 ablation

#### Depth routing 相关
- top-only
- uniform avg
- learned static mix
- Action-AttnRes-Lite
- chunk-shared vs per-slot
- LLM-only vs ViT-only vs dual-branch
- last 4 layers vs last 8 layers->4 blocks
- with / without gamma residual fusion

#### Adapter 相关
- ViT only
- LLM only
- ViT + LLM
- last 2 layers vs last 4 layers
- rank 4 / 8 / 16

#### Refiner 相关
- deterministic residual vs flow vs diffusion
- latent dim 32 / 64 / 128
- 1-step / 2-step / 4-step

#### Gate 相关
- no gate
- gate from residual norm label
- gate from coarse error label
- gate + depth entropy
- gate + disagreement
- different threshold tau

#### Loss 相关
- L1 vs Huber
- with / without residual reg
- with / without entropy regularization
- with / without gripper upweight

---

## 24. 第二十四步：效率评估必须单独做，不要混在 success 里

### 目标
证明你没有把 BitVLA 的最大卖点搞丢。

### 你必须单独测四种模式
1. `coarse-only`
2. `coarse + Action-AttnRes-Lite`
3. `always-on refiner`
4. `gated refiner`

### 你必须记录的指标
- avg latency
- p95 latency
- throughput
- peak VRAM
- 模型参数量增加
- checkpoint 大小增加
- 每个 episode 的 refinement 比例
- depth tap overhead

### 测量方法
1. 固定 batch size 和输入 shape
2. 先 warmup
3. 再测 100 或 500 次 query
4. 同一机器、同一环境、同一 dtype
5. 单独对比：
   - hook off / hook on
   - tap off / tap on
   - Action-AttnRes-Lite off / on

### 输出图
- success vs latency
- success vs memory
- refinement ratio vs success
- threshold tau vs latency/success Pareto
- depth routing overhead vs success gain

---

## 25. 第二十五步：做 OOD / robustness 分析，不要只看标准 benchmark

### 目标
让你的故事更像顶会，而不是“在一个固定 benchmark 上堆了个头”。

### 仿真里能做的 OOD
1. camera jitter / viewpoint shift
2. color / brightness perturbation
3. distractor texture
4. object pose perturbation
5. partial occlusion
6. instruction wording variation（如果 pipeline 支持）

### 你最关心的观察
- depth routing 是否主要帮助：
  - 接触前微调
  - 遮挡恢复
  - gripper 时序
  - 长时序误差积累后的 recovery
- 更早 / 中间深度是否在干扰或遮挡场景被更多使用

### 如果后面有真实机器人
优先复刻 BitVLA paper 里的任务类型：
- grasp
- flip
- put-into-basket
以及：
- unseen object
- visual distractor

---

## 26. 第二十六步：一定要做 failure case taxonomy

### 目标
即使涨点不夸张，只要你能清楚证明“改善集中发生在 deterministic low-bit decoder 最脆弱的状态”，故事也成立。

### 你要把失败拆成这些类
1. **coarse plan 对，接触不准**
2. **gripper 时序错**
3. **后半段 drift**
4. **遮挡后恢复失败**
5. **refiner 过修正**
6. **gate 漏触发**
7. **gate 误触发造成不必要计算**
8. **depth routing 无效或误路由**
   - 一直只看 top layer
   - 一直乱看浅层
   - entropy 很高但没帮助

### 每一类至少做
- 2~3 个定性视频
- 1 张动作曲线图
- 1 张 depth weight 图
- 1 段文字分析

---

## 27. 第二十七步：保留一个高风险支线 —— Tail Block AttnRes Graft（最后再做）

### 目标
保留你和另一个 AI 讨论里最激进但也最有潜力的思想：

> **如果 action-path-localized depth routing 已经证明有用，能不能只在 backbone 顶部少数 block 上试一个真正的 Block AttnRes graft？**

但请注意：
- 这不是主线；
- 这是高风险、后验增强版；
- 只有当前面已经有强正结果时才值得做。

### 为什么它不能做主线
1. 它已经开始改 backbone 了；
2. 和原始 AttnRes 一样，需要重新定义 block 内 / block 间 aggregation；
3. post-hoc graft 到已训练 VLA 上，不一定能学好；
4. 更容易带来 latency / memory overhead。

### 如果你真的要试，怎么做得最保守
#### 只改最后 2~4 个 LLM block
- 不碰前面大部分 low-bit backbone
- 不碰 ViT
- 只在最后几层试 tail-only Block AttnRes

#### 初始化必须保持 identity-like
- 初始时尽量接近原 residual path
- 不允许一插进去模型就行为巨变

#### 先做两个极简对照
1. tail block static mix
2. tail block Block AttnRes-like dynamic mix

### 这个支线只有在什么情况下值得写进论文
- 它在 mainline 已经成立的基础上，再给出额外但不大的增益；
- 或者它能帮助你论证：**真正重要的不是“整个 backbone 重写”，而是“顶部关键动作路径的 depth routing”。**

---

## 28. 第二十八步：把论文故事提前写出来，不要等最后再想 framing

### 目标
让实验从第一天开始为最终论文服务。

### 正确 framing
不要写成：
> 我们给 BitVLA 加了一个 flow / diffusion head，所以涨点了。

也不要写成：
> 我们把 Kimi 的 AttnRes 移植到了 BitVLA。

更好的写法是：

> **BitVLA 已经解决了低比特 backbone 的轻量与速度问题，但其动作输出端仍然依赖 top-layer-centered deterministic chunk regression。我们提出 action-path precision rescue：受 depth-wise selective aggregation 启发，我们设计 action-conditioned depth routing，使动作路径能够按内容读取多深度表示；随后通过轻量 adapter、deterministic residual correction 与 gated residual flow refinement，仅在动作关键路径与困难状态上恢复精度，从而在尽量保留 BitVLA 轻量优势的前提下提升动作精度和鲁棒性。**

### 最好强调的三到四个 claim
1. **精度恢复不需要均匀发生在整个 low-bit VLA 中，只需要发生在动作关键路径。**
2. **对 low-bit fast VLA 来说，多深度读出比只信 top layer 更合理，且 input-conditioned depth routing 比 static mixing 更有效。**
3. **residual generative refinement 比 full generative replacement 更适合 BitVLA 这类 fast chunk-decoding policy。**
4. **selective gating 可以把额外计算集中投给 hard states。**

### 论文命名建议
你最后的方法名最好不要直接叫 AttnRes。可以考虑：
- **Depth-Selective Action Refinement (DSAR)**
- **Action-Conditioned Depth Routing for BitVLA**
- **BitVLA-APR: Action-Path Precision Rescue**
- **BitVLA-DepthRefine**

---

## 29. 第二十九步：你最后必须产出的图和表，提前预留脚本

### 必做 Figure
1. 方法总览图
2. Success vs Latency Pareto
3. Success vs Memory Pareto
4. Long / OOD 分项柱状图
5. coarse vs refined 动作误差曲线
6. gate 激活率随 episode step 变化图
7. depth routing 权重可视化图
8. 失败案例可视化

### 必做 Table
1. 主 benchmark 结果
2. Uniform / Static / Action-AttnRes ablation
3. Adapter / DetResidual / Flow / Gate ablation
4. 效率表
5. OOD / hard-state 分析表

### 必做附录
- 训练配置
- 模块参数量
- 额外开销
- 更多 qualitative case
- depth routing 更多可视化

---

## 30. 第三十步：课程 deliverables 对齐清单（无时间线版）

### Proposal Presentation 前必须准备好的内容
1. BitVLA baseline 复现结果
2. 一个明确的 failure case taxonomy
3. `top-only vs static mix vs Action-AttnRes-Lite` 的初步对比
4. 方法图草稿
5. 你的核心 hypothesis 和 3 个主 claim

### Final Report and Code 前必须完成的内容
1. 可运行的 baseline 复现代码
2. 至少一条完整主线：
   - `BitVLA + Action-AttnRes-Lite + Adapter + DetResidual + Flow(gated)`
3. 主 ablation matrix
4. 效率分析
5. hard-state / OOD 分析
6. 失败案例分析

### Showcase 要准备的内容
1. baseline vs final method rollout 对比视频
2. hard-state 成功案例
3. 1~2 个失败案例和分析
4. 一页简洁网站或海报图

---

## 31. 最后的执行顺序（最实用的精简版）

如果你只想看最核心的执行顺序，就按这个来：

1. 跑通 BitVLA 官方 eval 和 Long baseline
2. 跑通 BitVLA Long finetune debug + full run
3. 搭好 logging / rollout / hard-state / efficiency 仪表盘
4. 实现 multi-depth tap + online summary
5. 先做 `top-only / uniform avg / static mix`
6. 做 `Action-AttnRes-Lite (LLM-only)`
7. 做 `Adapter-only`
8. 做 `Action-AttnRes-Lite + Adapter`
9. 做 `Deterministic Residual Head`
10. 做 `Residual Latent Flow Matching (always-on)`
11. 做 `Gate / Selective Refinement`
12. 做多 seed、全 benchmark、OOD、效率图
13. 如果有必要，再做 diffusion 对照、KD、dual-branch、tail Block AttnRes graft

---

## 32. 参考链接

- BitVLA: https://arxiv.org/abs/2506.07530
- BitVLA GitHub: https://github.com/ustcwhy/BitVLA
- Attention Residuals: https://arxiv.org/abs/2603.15031
- Attention Residuals GitHub: https://github.com/MoonshotAI/Attention-Residuals
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- Octo: https://arxiv.org/abs/2405.12213
- QLoRA: https://arxiv.org/abs/2305.14314

