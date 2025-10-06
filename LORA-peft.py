
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.amp
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./LORA-peft"        # 你选的 A: 当前目录 ./lora_sst2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

BATCH_SIZE = 4        # per-step batch size
ACCUM_STEPS = 4       # accumulation -> effective batch = 16
EPOCHS = 1
LR = 2e-5

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def collate_fn(batch):
    # batch: list of dicts of tensors
    return {k: torch.stack([ex[k] for ex in batch]) for k in batch[0]}

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


# 数据集加载

def prepare_dataset(tokenizer, max_len=128):
    ds = load_dataset("glue", "sst2")

    def preprocess(ex):
        out = tokenizer(ex["sentence"], padding="max_length", truncation=True, max_length=max_len)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# 带LoRA模型的建立

def build_model_with_lora(model_name, r, alpha, lora_dropout):
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_lin", "v_lin"],  # DistilBERT typical names
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base, lora_config)
    return model, base


# 评估函数

def evaluate(model, dataloader, device):
    model.eval()
    acc_list = []
    loss_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            # 原：with autocast():
            # 新：
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
                outputs = model(**batch)
                logits = outputs.logits
                loss = F.cross_entropy(logits, batch["labels"]).item()
            acc = compute_accuracy(logits, batch["labels"])
            acc_list.append(acc)
            loss_list.append(loss)
    model.train()
    avg_acc = float(sum(acc_list) / len(acc_list)) if acc_list else 0.0
    avg_loss = float(sum(loss_list) / len(loss_list)) if loss_list else 0.0
    return avg_acc, avg_loss


# 保存adapter

def save_adapter(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # PEFT 的 model.save_pretrained 会保存 adapter 元数据和权重
    model.save_pretrained(out_dir)


# 训练流程 (AMP + grad accumulation)

def train(model, train_loader, dev_loader, device, epochs, lr, accum_steps, out_dir):
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device_type="cuda" if device == "cuda" else "cpu")
    
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        running_loss = 0.0
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # 原：with autocast():
            # 新：
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
                outputs = model(**batch)
                logits = outputs.logits
                loss = F.cross_entropy(logits, batch["labels"])
                loss = loss / accum_steps  # normalize for accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})
                running_loss = 0.0

        # 验证（evaluate函数也会修改，见下）
        val_acc, val_loss = evaluate(model, dev_loader, device)
        print(f"[Epoch {epoch+1}] Validation — acc: {val_acc:.4f} | avg_loss: {val_loss:.4f}")

        # 保存 adapter
        save_adapter(model, out_dir)
        print(f"Saved LoRA adapter to {out_dir}")


def load_adapter_for_infer(base_model_name, adapter_dir, device):
    base = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()
    return model

def infer(model, tokenizer, sentence, device):
    model.eval()
    inputs = tokenizer(sentence, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
            outputs = model(**inputs)
            logits = outputs.logits
            pred = int(torch.argmax(logits, dim=-1).cpu().item())
    return pred

# 主函数

def main():
    print("Device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = prepare_dataset(tokenizer, max_len=MAX_LEN)

    train_ds = ds["train"]
    dev_ds = ds["validation"]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model, base = build_model_with_lora(MODEL_NAME, r=LORA_R, alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    print("=== Trainable params (LoRA 应只占少部分) ===")
    model.print_trainable_parameters()

    train(model, train_loader, dev_loader, device=DEVICE, epochs=EPOCHS, lr=LR, accum_steps=ACCUM_STEPS, out_dir=OUTPUT_DIR)

    # inference demo: load adapter-only and predict
    print("Loading adapter for inference demo ...")
    inf_model = load_adapter_for_infer(MODEL_NAME, OUTPUT_DIR, device=DEVICE)
    example_sent = "I absolutely loved this movie. Highly recommend!"
    pred = infer(inf_model, tokenizer, example_sent, device=DEVICE)
    print(f"Example -> '{example_sent}'  Pred label: {pred}  (1=pos, 0=neg)")

if __name__ == "__main__":
    main()
