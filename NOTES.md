# 关于本项目的代码来源说明

## 代码来源

本项目中的训练脚本、模型定义代码（`model/` 目录下的文件）以及数据加载逻辑，**均来源于 [MiniMind2 官方仓库](https://github.com/jingyaogong/minimind)**。

MiniMind2 是一个由 **jingyaogong** 开发的开源轻量级大语言模型项目，采用 MIT 许可证，非常适合个人学习和研究使用。

## 用户完成的工作

虽然代码框架来自 MiniMind2 官方仓库，但用户亲自完成了以下核心工作：

1. **四阶段完整训练**
   - Stage 1: Pretrain（预训练）——在 AutoDL RTX 4090 上完成
   - Stage 2: SFT（监督微调）——理解指令数据格式并完成微调
   - Stage 3: DPO（直接偏好优化）——实现 RLHF 中的偏好优化阶段
   - Stage 4: LoRA（低秩适配）——实现参数高效微调

2. **训练过程中的工程实践**
   - 配置和调试 AutoDL 云端训练环境
   - 解决训练过程中遇到的各类工程问题（tokenizer 配置、loss mask 处理、路径管理等）
   - 对比分析四个阶段模型的生成效果差异

3. **效果对比与分析**
   - 使用相同的 prompt 测试四个阶段模型的输出
   - 系统记录和分析每个阶段的能力提升
   - 撰写对比报告总结训练效果

## 为什么这样组织仓库

由于以下原因，本仓库仅包含文档和说明文件，不包含完整的训练代码：

1. **AutoDL 环境限制**: 用户无法从 AutoDL 终端复制粘贴命令，也无法直接下载训练好的文件
2. **代码归属清晰**: MiniMind2 官方代码由原作者维护，保持代码的原始来源清晰透明
3. **项目重点**: 本项目的价值在于用户**亲自完成四阶段训练并理解每个阶段的工程细节**，而非重写一套训练代码

## 如何获取完整代码

如需复现本项目，请按以下步骤操作：

```bash
# 1. 克隆 MiniMind2 官方仓库
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
bash scripts/download_data.sh

# 4. 按照 README.md 中的四阶段训练命令依次执行训练
```

## 许可证

MiniMind2 官方代码采用 MIT 许可证。本仓库中的文档（README.md、compare_results.md、interview_guide.md 等）由用户独立撰写。

## 参考链接

- [MiniMind2 GitHub 仓库](https://github.com/jingyaogong/minimind)
- [MiniMind2 官方文档](https://jingyaogong.github.io/minimind/)
