# MiniMind 从零构建与四阶段训练 — AI Infra 学习项目

> 基于 [MiniMind2](https://github.com/jingyaogong/minimind) 的完整 LLM 训练链路实践，用于 AI Infra / 大模型工程方向实习准备。

> **⚠️ 重要说明**: 本仓库中的训练脚本和模型代码来源于 [MiniMind2 官方仓库](https://github.com/jingyaogong/minimind)。用户在此基础上亲自完成了 **Pretrain → SFT → RLHF(DPO) → LoRA** 四个核心训练阶段的完整实验，并对各阶段模型的生成效果进行了系统对比。具体说明请查看 [NOTES.md](./NOTES.md)。

## 项目简介

本项目从零构建了一个 26M 参数规模的轻量级大语言模型，完整跑通了 **Pretrain → SFT → RLHF(DPO) → LoRA** 四个核心训练阶段，并对各阶段模型的生成效果进行了系统对比。

**目标岗位**: AI Infra 工程师 / 大模型训练工程师 / RL Infra 工程师

**核心能力展示**:
- 深入理解 LLM 完整训练链路（预训练、监督微调、偏好优化、参数高效微调）
- 掌握 PyTorch 分布式训练基础（DDP、混合精度、梯度累积、梯度裁剪）
- 理解 RLHF 中 DPO 算法的工程实现细节
- 具备从零搭建训练框架、排查训练问题的工程能力

---

## 模型架构

| 配置项 | 数值 |
|--------|------|
| 参数量 | 26.08 M |
| 隐藏维度 (dim) | 512 |
| 层数 (n_layers) | 8 |
| 注意力头数 | 8 |
| 最大序列长度 | 512 |
| 词汇表大小 | 6400 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| 激活函数 | SwiGLU |
| 训练精度 | FP16 (half) |

**架构特点**: 标准的 Decoder-only Transformer，采用 RoPE 相对位置编码 + RMSNorm + SwiGLU 激活函数，与 LLaMA 架构设计一致。

---

## 训练数据

| 阶段 | 数据集 | 规模 | 说明 |
|------|--------|------|------|
| Pretrain | `pretrain_t2t_mini.jsonl` | 127万条 | 通用文本预训练语料 |
| SFT | `sft_t2t_mini.jsonl` | 90万条 | 指令-回答对，有监督微调 |
| DPO | `dpo.jsonl` | 1.7万条 | 偏好对比数据（chosen/rejected） |
| LoRA | `lora_identity.jsonl` | 自定义 | 身份注入数据（角色扮演） |

---

## 四阶段训练流程

### Stage 1: 预训练 (Pretrain)
- **目标**: 学习语言建模能力，预测下一个 token
- **损失函数**: CrossEntropyLoss（标准的自回归语言建模）
- **训练效果**: 模型学会生成连贯文本，但**不具备对话能力**
- **关键观察**: 对 "你好" 的回复是 "我是一个小姑娘，名叫艾玛..." —— 模型不理解对话格式

### Stage 2: 监督微调 (SFT)
- **目标**: 学习对话格式，让模型能按指令回复
- **数据格式**: `{"instruction": "...", "output": "..."}`
- **训练效果**: 模型开始理解 `user/assistant` 角色，能给出基本回应
- **关键观察**: "你好" → "你好，我能为你做些什么呢？"（能对话了）

### Stage 3: 偏好优化 (DPO / RLHF)
- **目标**: 让模型输出更符合人类偏好（更有用、更无害、更礼貌）
- **算法**: **DPO (Direct Preference Optimization)**，无需训练单独的 Reward Model
- **核心思想**: 直接优化 `chosen` 相对于 `rejected` 的对数概率比
- **训练效果**: 回复更礼貌、更有结构、更主动提供帮助
- **关键观察**: 对天气问题会反问 "您所在的城市是哪里？"

### Stage 4: 参数高效微调 (LoRA)
- **目标**: 在保持基座模型不变的情况下，注入特定领域知识（身份/角色）
- **技术**: LoRA (Low-Rank Adaptation)，只训练 1% 的参数
- **配置**: rank=8, alpha=16
- **训练效果**: 回复最详细、最有结构，能分点说明

---

## 四阶段生成效果对比

详见 [compare_results.md](./compare_results.md) 获取完整的对比分析。

```
Prompt: 你好
  [Pretrain    ] 我是一个小姑娘，名叫艾玛。她有许多卖家可以提供在线聊天服务...
  [SFT         ] 你好，我能为你做些什么呢？
  [RLHF/DPO    ] 你好，我可以帮助你。请问有什么我可以帮助你的？
  [LoRA        ] 你好！欢迎再次为您服务。您有什么需要帮助的吗？

Prompt: 请问今天天气怎么样
  [Pretrain    ] 我想知道天气如何，这是如何影响我们的生活。今天的天气非常好...
  [SFT         ] 很抱歉，我无法获取您所在地的天气信息，建议您查看当地的天气预报...
  [RLHF/DPO    ] 您所在的城市是哪里？我可以帮您查询当地的天气预报。
  [LoRA        ] 今天的天气是晴朗的，温度适中，适合户外活动...

Prompt: 讲个笑话
  [Pretrain    ] Linda：您好，Linda。听说你在国内外旅行...
  [SFT         ] 好的，有一次我和朋友去看电影，结果自己错了路...
  [RLHF/DPO    ] 好的，给你讲个笑话。为什么猪会吃草？因为它觉得草就在天空中。
  [LoRA        ] "笑话"是一个常见的口头笑话，它叫做"笑话"...

Prompt: 如何学习编程
  [Pretrain    ] Iner: 嗨，Kate。我是David，一名程序员...
  [SFT         ] 学习编程的基本步骤包括选择编程语言，阅读相关书籍...
  [RLHF/DPO    ] 学习编程需要掌握基本的编程语言和编程思维方式...
  [LoRA        ] 学习编程是一个不断发展的过程...
        1. **阅读代码**：阅读...

Prompt: 人工智能是什么
  [Pretrain    ] 智能语音助手是一种使用自然语言处理技术构建...
  [SFT         ] 人工智能是计算机科学的一个分支...
  [RLHF/DPO    ] 人工智能是一种基于计算机技术的计算模型...
  [LoRA        ] 人工智能是一种模拟人类智能的技术...
```

**核心结论**: 四个阶段的训练效果呈现明显的递进关系——从"会说人话"到"能对话"到"对话得好"到"专业回复"。

---

## 项目结构

```
ai-infra-learning/
├── model/                      # 模型定义（源自 MiniMind2 官方）
│   ├── model_minimind.py       # MiniMind 模型主体 (MiniMindForCausalLM)
│   ├── LMConfig.py             # 模型配置类
│   ├── model_lora.py           # LoRA 注入与加载
│   ├── dataset.py              # 各阶段 Dataset 定义
│   └── minimind_tokenizer/     # Tokenizer
├── train_pretrain.py           # Stage 1: 预训练
├── train_full_sft.py           # Stage 2: 监督微调
├── train_dpo.py                # Stage 3: DPO 偏好优化
├── train_lora.py               # Stage 4: LoRA 微调
├── train_distillation.py       # 知识蒸馏 (未运行)
├── train_distill_reason.py     # 推理能力蒸馏 (未运行)
├── eval_model.py               # 模型评估脚本
├── final_compare.py            # 四阶段效果对比脚本
├── out/                        # 训练权重输出
│   ├── pretrain_512.pth
│   ├── full_sft_512.pth
│   ├── rlhf_512.pth
│   └── lora/
│       └── lora_identity_512.pth
└── dataset/                    # 训练数据
    ├── pretrain_t2t_mini.jsonl
    ├── sft_t2t_mini.jsonl
    └── dpo.jsonl
```

> **注意**: 由于模型权重文件较大，本项目仅包含代码。如需复现，请参考下方"如何复现"部分。

---

## 训练环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| 训练框架 | PyTorch 2.x |
| 分布式 | DDP (DistributedDataParallel) |
| 精度 | FP16 (Automatic Mixed Precision) |
| 优化器 | AdamW |
| 学习率调度 | Cosine Annealing + Warmup |

---

## 技术亮点（面试重点）

### 1. 完整的 LLM 训练链路工程能力
- 从零搭建 Pretrain → SFT → DPO → LoRA 完整 pipeline
- 理解每个阶段的**数据格式差异**、**损失函数设计**、**优化目标**
- 掌握各阶段权重加载与衔接（Pretrain 权重 → SFT 初始化 → DPO 初始化 → LoRA 基座）

### 2. DPO 算法的工程实现
- **无需 Reward Model**：直接用偏好数据 `(chosen, rejected)` 对优化
- **核心公式**: `loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))`
- **工程细节**: 正确的 `log_prob` 计算需要处理 padding mask，避免 pad token 污染损失
- **对比理解**: DPO vs PPO — DPO 更简单高效，不需要在线采样和奖励模型训练

### 3. LoRA 参数高效微调
- 只训练 1% 参数 (262K / 26M)，大幅降低显存和训练时间
- 理解 LoRA 的数学原理: `W = W_0 + B @ A`，其中 `A`、`B` 为低秩矩阵
- 工程实现: `apply_lora()` 注入 + `save_lora()`/`load_lora()` 只存增量参数

### 4. 分布式训练基础
- DDP (DistributedDataParallel) 多卡训练
- 混合精度训练 (FP16) 加速 + 省显存
- 梯度累积模拟大 batch size
- 梯度裁剪 (gradient clipping) 防止梯度爆炸

### 5. 问题排查能力
- 解决 `tokenizer.pad_token_id` 未设置导致的生成异常
- 修复 DPO 中 `token_log_probs * mask` 的 shape mismatch 问题
- 处理 LoRA 保存路径不存在导致的 RuntimeError
- 适配不同版本的 `generate()` API（位置参数 vs 关键字参数）

---

## 如何复现

```bash
# 1. 克隆 MiniMind2 官方仓库
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
bash scripts/download_data.sh

# 4. 四阶段训练（依次执行）
python train_pretrain.py --dim 512 --n_layers 8
python train_full_sft.py --dim 512 --n_layers 8
python train_dpo.py --dim 512 --n_layers 8
mkdir -p out/lora
python train_lora.py --dim 512 --n_layers 8

# 5. 效果对比
python final_compare.py
```

---

## 收获与思考

1. **数据质量 > 模型规模**: 1.26M 参数的模型经过 SFT 后已经能基本对话，说明高质量指令数据比盲目堆参数更重要。

2. **DPO 的简洁之美**: 相比 PPO 需要维护 Policy、Reward、Value 多个模型，DPO 只用一对偏好数据就能直接优化，工程实现更轻量。

3. **LoRA 的实用性**: 在 RL 方向，LoRA 可用于快速迭代不同 reward 函数的实验，不用每次都全量训练，这对 RL Infra 的工程效率至关重要。

4. **AI Infra 的核心**: 不是会写模型定义，而是能搭起**稳定、高效、可复现**的训练 pipeline，并在出问题时有能力定位和修复。

---

## 面试话术指南

详见 [interview_guide.md](./interview_guide.md) 获取完整的面试准备资料，包括项目介绍、技术问题 Q&A、以及反问环节建议。

---

## 后续计划

- [ ] 复现 MiniMind 的 **知识蒸馏** (Distillation) 和 **推理蒸馏** (Reasoning Distillation)
- [ ] 学习 **PPO + Reward Model** 的完整 RLHF 实现
- [ ] 尝试 **GRPO** (Group Relative Policy Optimization) 等新型 RL 算法
- [ ] 将训练框架适配到 **Megatron-LM / DeepSpeed** 等工业级分布式框架

---

## 参考

- [MiniMind GitHub](https://github.com/jingyaogong/minimind)
- [MiniMind 文档](https://jingyaogong.github.io/minimind/)
- [DPO Paper: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [LoRA Paper: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
