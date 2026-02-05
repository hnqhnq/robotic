## 强化学习算法家族

DDPG、TD3、SAC、DrQ / DrQ-v2（视觉RL）

### 1. DDPG
定义：
DDPG = 用神经网络做“连续动作”的 Actor–Critic 强化学习算法
👉 DDPG = 连续动作版的 Q-learning



### 2. TD3
TD3 的目标非常明确：

既然 DDPG 最大问题是 Critic 不靠谱，那我就从 Critic 下手。

### 3. SAC
SAC 仍然是 Actor–Critic，但和 TD3 有三个本质区别。
1️⃣ Actor：从“确定性” → “随机策略”
2️⃣ Critic：仍然是双 Q（和 TD3 一样）
3️⃣ 多了一个“自动调温器”（Automatic Entropy Tuning）

### 4. DrQ / DrQ-v2
DrQ 的核心机制：数据增强 + 一致性约束
你可以把它写成：
基础骨架：SAC（常见）
关键改动：对像素输入做数据增强，并用增强后的观测训练 critic/actor

DrQ-v2 = 把 DrQ 的“视觉增强思想”保留，同时换成更简单、更稳定、更高效的 DDPG/TD3 风格训练范式。

### 5. 归一化
| 维度            | BatchNorm | LayerNorm | SimplexNorm |
| ------------- | --------- | --------- | ----------- |
| 归一化对象         | batch 内样本 | 单样本特征     | 单样本特征（概率化）  |
| 依赖 batch size | ✅ 强依赖     | ❌ 不依赖     | ❌ 不依赖       |
| 适合 RL         | ❌ 通常不适合   | ✅ 常用      | ✅（为 RL 设计）  |
| 抗非平稳性         | ❌         | ✅         | ✅✅          |
| 抗表征塌缩         | ❌         | ⚠️ 有限     | ✅           |
| 多模态友好         | ❌         | ⚠️        | ✅           |
| 可解释性          | 一般        | 一般        | ⭐⭐⭐⭐        |


## 其他
像素 / 本体输入
   ↓
Encoder（视觉 / 本体）
   ↓
Fusion（多模态）
   ↓
Latent Representation
   ↓
Actor / Critic（TD3）

