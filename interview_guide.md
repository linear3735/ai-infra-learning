# 面试话术指南 —— MiniMind 四阶段训练项目

> 本文档为 AI Infra / 大模型工程方向面试准备，帮助你在面试中清晰、有逻辑地介绍本项目。

---

## 一、1 分钟项目介绍（电梯演讲）

**版本 1（简洁版）**:
> 我基于 MiniMind2 从零搭建了一个 26M 参数的轻量级大语言模型，完整跑通了 Pretrain、SFT、DPO、LoRA 四个训练阶段。每个阶段我都对比了生成效果，验证了从"会说人话"到"能对话"到"对话得好"的递进关系。这个项目帮我深入理解了 LLM 的完整训练链路和每个阶段的工程细节。

**版本 2（技术版）**:
> 我实现了一个 26M 参数的 Decoder-only Transformer，使用 PyTorch DDP 进行分布式训练，采用 FP16 混合精度。我亲自完成了四个阶段的训练：预训练（自回归语言建模）、SFT（指令微调）、DPO（直接偏好优化，无需 Reward Model）、LoRA（低秩适配，只训练 1% 参数）。训练过程中我解决了 tokenizer pad_token 设置、DPO loss mask 处理、LoRA 参数注入等多个工程问题。

---

## 二、项目介绍结构化模板（3-5 分钟）

### 开场：项目背景与目标
- "我的目标是准备 AI Infra 工程师的实习，需要展示对 LLM 训练全链路的理解"
- "MiniMind2 是一个开源的轻量级 LLM 项目，参数规模 26M，适合个人 GPU 上完整复现"

### 核心工作 1：模型架构实现
- "标准 Decoder-only Transformer，与 LLaMA 架构一致：RoPE 位置编码 + RMSNorm + SwiGLU"
- "我自己理解了每个组件的作用，比如 RoPE 相比绝对位置编码的优势是相对位置感知能力"

### 核心工作 2：四阶段训练
- **Pretrain**: "用 127 万条通用文本进行自回归预训练，让模型学会语言建模"
- **SFT**: "用 90 万条指令数据微调，让模型理解对话格式"
- **DPO**: "用 1.7 万条偏好数据直接优化，不需要训练 Reward Model，核心是对数概率比的优化"
- **LoRA**: "只训练 262K 参数（1%），实现角色的快速注入"

### 核心工作 3：问题排查与工程优化
- "训练过程中遇到了 tokenizer pad_token_id 未设置导致的生成异常"
- "DPO 阶段处理了 loss mask 的 shape mismatch 问题"
- "LoRA 阶段解决了保存路径不存在导致的 RuntimeError"

### 收尾：收获与思考
- "最大的收获是理解了每个训练阶段的数据格式差异和优化目标"
- "深刻体会到 AI Infra 的核心不是写模型，而是搭起稳定、高效、可复现的训练 pipeline"

---

## 三、高频技术问题 Q&A

### Q1: 为什么选 MiniMind2？不直接用 LLaMA 或 GPT？

**回答要点**:
- MiniMind2 参数量 26M，单卡 4090 可以完整跑通四个阶段
- 工业级模型（如 LLaMA 7B/70B）需要多机多卡、需要大量数据，个人无法完整体验训练链路
- 我的目标是**理解训练链路的工程细节**，而非训练一个工业级模型
- 26M 模型已经足以展示四阶段训练的递进效果

---

### Q2: Pretrain 和 SFT 的本质区别是什么？

**回答要点**:
- **目标不同**: Pretrain 是语言建模（预测下一个 token），SFT 是指令跟随（学习 input→output 的映射）
- **数据不同**: Pretrain 是连续文本，SFT 是 (instruction, output) 成对数据
- **损失函数相同但意义不同**: 都是 CrossEntropyLoss，但 Pretrain 的 label 是输入的 shifted 版本，SFT 的 label 只计算 output 部分的 loss
- **效果差异**: Pretrain 模型"会说人话但不懂对话"，SFT 模型"能按指令回复"

---

### Q3: DPO 和 PPO 有什么区别？为什么选 DPO？

**回答要点**:
- **PPO**: 需要训练 Reward Model + Policy Model + Value Model，需要在线采样，流程复杂，训练不稳定
- **DPO**: 直接用偏好数据 `(chosen, rejected)` 对进行优化，不需要 Reward Model
- **DPO 核心公式**: `loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))`
- **工程优势**: DPO 实现简单、训练稳定、显存占用小。PPO 在工业界更常用是因为它能处理更复杂的奖励信号，但 DPO 对于入门理解 RLHF 足够了

---

### Q4: LoRA 的原理是什么？为什么只训练 1% 参数就能有效果？

**回答要点**:
- **核心思想**: 不直接微调权重矩阵 W，而是训练两个低秩矩阵 A 和 B，让 `W = W_0 + B @ A`
- **低秩假设**: 模型权重的变化（fine-tuning delta）是低秩的，可以用很小的矩阵近似
- **参数效率**: A 是 (in, r)，B 是 (r, out)，r=8 时参数量是原来的 `2*r*d / d^2 = 2r/d`，约为 1%
- **工程价值**: 多个 LoRA 可以共享同一个基座模型，切换任务只需换 LoRA 权重，部署效率高

---

### Q5: 训练过程中遇到过什么问题？如何解决的？

**回答要点**（准备 2-3 个）:

**问题 1: tokenizer pad_token_id 未设置**
- 现象: 模型生成时出现异常 token，输出不连贯
- 原因: MiniMind tokenizer 没有默认 pad_token，generate() 时 padding 用了 eos_token_id
- 解决: 显式设置 `tokenizer.pad_token = tokenizer.eos_token`

**问题 2: DPO loss mask shape mismatch**
- 现象: `token_log_probs * mask` 报维度不匹配错误
- 原因: mask 和 log_probs 的维度没有对齐（batch 维度处理不一致）
- 解决: 检查 `compute_log_probs` 函数中 mask 的 broadcast 逻辑，确保 `(B, T)` 和 `(B, T)` 正确相乘

**问题 3: LoRA 保存路径不存在**
- 现象: RuntimeError 提示目录不存在
- 原因: `out/lora/` 目录没有提前创建
- 解决: 使用 `os.makedirs(save_dir, exist_ok=True)` 确保目录存在

---

### Q6: 如果让你优化这个训练流程，你会怎么做？

**回答要点**:
1. **数据并行优化**: 目前用的是 DDP，可以尝试 DeepSpeed ZeRO-2/ZeRO-3 来进一步降低显存占用
2. **混合精度升级**: 从 FP16 升级到 BF16（如果硬件支持），数值稳定性更好
3. **训练速度**: 使用 FlashAttention 替换标准 Attention，降低显存占用并加速
4. **数据预处理**: 使用 `datasets` 库 + `DataCollator` 替代手写的 Dataset，更规范
5. **监控**: 接入 wandb / tensorboard，实时监控 loss curve 和生成样例
6. **Checkpoint 管理**: 实现自动保存最佳模型 + 定期备份

---

### Q7: 你对 AI Infra 的理解是什么？

**回答要点**:
- AI Infra 不仅仅是写模型代码，而是**让模型训练稳定、高效、可复现的一整套工程体系**
- 包括：数据 pipeline、训练框架、分布式策略、显存优化、监控告警、模型部署
- 核心价值：让算法研究员可以专注于算法创新，而不需要关心底层的工程细节
- 我认为好的 AI Infra 工程师需要同时具备：深度学习理论理解 + 分布式系统知识 + 问题排查能力

---

## 四、反问环节建议问题

面试结束时通常会有反问环节，准备 1-2 个好问题可以加分：

### 推荐问题

1. **"贵组的 AI Infra 团队目前主要负责哪些工作？是更偏训练框架还是推理优化？"**
   - 展示你对 AI Infra 细分领域的了解

2. **"贵组在大模型训练中主要使用什么框架？Megatron-LM、DeepSpeed 还是自研框架？"**
   - 展示你对工业级训练框架的了解

3. **"对于实习生，贵组更期待我入职后快速上手现有工具链，还是有空间参与新工具的开发？"**
   - 展示你的主动性和对成长空间的关注

4. **"贵组目前在大模型训练中最主要的工程挑战是什么？是显存、通信还是数据 pipeline？"**
   - 展示你对训练痛点的理解

### 不推荐的问题
- ❌ "实习生有转正机会吗？"（太功利，且通常有标准流程）
- ❌ "加班多吗？"（负面暗示）
- ❌ "工作内容具体是什么？"（应该面试前做功课）

---

## 五、简历描述建议

### 项目描述（简历版）

```
MiniMind 四阶段训练链路复现 | 个人项目
- 基于 MiniMind2 从零搭建 26M 参数 LLM，完整实现 Pretrain/SFT/DPO/LoRA 四阶段训练
- 使用 PyTorch DDP + FP16 混合精度在单卡 RTX 4090 上完成训练，掌握分布式训练基础
- 深入理解 DPO 算法的工程实现（无需 Reward Model 的偏好优化）
- 实现 LoRA 低秩适配，仅训练 1% 参数（262K）即可注入特定角色知识
- 独立排查并修复 tokenizer 异常、loss mask shape mismatch、路径错误等工程问题
```

### 技能关键词
- LLM Training Pipeline (Pretrain / SFT / RLHF / LoRA)
- PyTorch Distributed Training (DDP)
- Mixed Precision Training (FP16)
- Direct Preference Optimization (DPO)
- Parameter-Efficient Fine-Tuning (LoRA)
- Transformer Architecture (RoPE, RMSNorm, SwiGLU)

---

## 六、常见误区提醒

1. **不要说"我训练了一个大模型"** —— 26M 不是"大"模型，重点在于你**完整跑通了训练链路**
2. **不要过度夸大效果** —— 26M 模型的生成效果有限，重点在于你对每个阶段的理解和工程实现
3. **不要只谈理论** —— 面试 AI Infra 岗位，工程细节（如 DDP 怎么配置、FP16 怎么开、checkpoint 怎么存）比理论更重要
4. **不要贬低 MiniMind** —— 面试官可能也知道这个项目，重点是展示你从中**学到了什么、解决了什么问题**
