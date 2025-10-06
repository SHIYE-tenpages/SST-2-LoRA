import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW


#  LoRA 模块定义

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # LoRA 低秩矩阵
        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 冻结原始层参数
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 原始输出 + LoRA 修正项
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling



# 注入 LoRA 到模型

def inject_lora(model, target_modules=("query", "value"), r=4, alpha=16):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, LoRALinear(module, r=r, alpha=alpha))
    return model


# LoRA Adapter 保存与加载

def save_adapters(model, path="lora_adapters.pt"):
    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_state[name + ".lora_A"] = module.lora_A.state_dict()
            lora_state[name + ".lora_B"] = module.lora_B.state_dict()
    torch.save(lora_state, path)
    print(f"✅ LoRA adapters saved to {path}")


def load_adapters(model, path="lora_adapters.pt", map_location="cpu"):
    lora_state = torch.load(path, map_location=map_location)
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.lora_A.load_state_dict(lora_state[name + ".lora_A"])
            module.lora_B.load_state_dict(lora_state[name + ".lora_B"])
    print(f"✅ LoRA adapters loaded from {path}")



# 训练函数

def collate_fn(batch):
    return {key: torch.stack([x[key] for x in batch]) for key in batch[0]}


def train(model, train_loader, dev_loader, device, epochs=3, lr=2e-5):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model(**batch)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss = {total_loss / len(train_loader):.4f}")

    print("✅ Training done.")



# 数据与模型加载

if __name__ == "__main__":
    import math

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")

    def tokenize_fn(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

    encoded_dataset = dataset.map(tokenize_fn, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(encoded_dataset["train"], batch_size=16, shuffle=True)
    dev_loader = DataLoader(encoded_dataset["validation"], batch_size=16)

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lora_model = inject_lora(base_model, r=4, alpha=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练 LoRA
    train(lora_model, train_loader, dev_loader, device, epochs=1)

    # 保存 adapter
    save_adapters(lora_model, "my_lora.pt")

    # 可选：重新加载
    load_adapters(lora_model, "my_lora.pt")
