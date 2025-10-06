LoRA 在 SST-2 上的参数高效微调
这是一个使用 LoRA（Low-Rank Adaptation）对 DistilBERT-base-uncased 模型进行参数高效微调的项目，在 GLUE SST-2（情感二分类）数据集上实现。项目包含两种实现：基于 Hugging Face PEFT 库的高级版本（LORA-peft.py）和从零自定义实现（LORA-self.py）。LoRA 通过低秩适配器仅更新少量参数（~0.1% 总参数），显著减少计算开销，同时保持高性能。

## 项目概述
- **任务**：情感分析（正面/负面，二分类）。
- **数据集**：GLUE SST-2（~67k 训练样本，872 验证）。
- **模型**：DistilBERT-base-uncased（66M 参数），仅 LoRA 适配器训练。
- **LoRA 配置**：rank=4-8, alpha=16, dropout=0.05；目标模块：q_lin/v_lin 或 query/value。
- **优化**：AdamW + FP16 AMP（PEFT 版）+ 梯度累积（有效 batch=16）。
- **框架**：PyTorch + Hugging Face Transformers（PEFT 版额外需 peft）。
- **评估**：验证集准确率/F1，参数压缩率。

## 环境要求

- Python 3.8+
- PyTorch 2.0+（支持 CUDA）
- Hugging Face Transformers 4.20+
- Datasets, Tqdm
- PEFT 0.4+（仅 PEFT 版）

## TODO

- 添加早停（early stopping）避免过拟合。
- 实验不同 rank/alpha 值或更多目标模块。
- 支持其他数据集（如 IMDB）或模型（如 BERT）。
- PEFT 版：使用 merge_and_unload() 合并适配器用于部署。
- 自定义版：添加验证循环和 dropout 支持。


