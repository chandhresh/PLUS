from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

# -------------------------------
# 1️⃣ Check GPU
# -------------------------------
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# -------------------------------
# 2️⃣ Load CSV dataset
# -------------------------------
dataset = load_dataset("csv", data_files="data/financial_sentiment.csv")
dataset = dataset["train"].train_test_split(test_size=0.2)

# -------------------------------
# 3️⃣ Detect columns
# -------------------------------
sample = dataset["train"][0]
TEXT_COL, LABEL_COL = None, None

for k in sample.keys():
    if k.lower() in ["sentence", "text", "content", "headline", "news"]:
        TEXT_COL = k
    if k.lower() in ["label", "sentiment", "class"]:
        LABEL_COL = k

print(f"✅ Using text column: {TEXT_COL}")
print(f"✅ Using label column: {LABEL_COL}")

# -------------------------------
# 4️⃣ Label mapping (CRITICAL)
# -------------------------------
LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def encode_labels(batch):
    batch["labels"] = [LABEL_MAP[str(x).lower()] for x in batch[LABEL_COL]]
    return batch

dataset = dataset.map(encode_labels, batched=True)

# -------------------------------
# 5️⃣ Tokenization
# -------------------------------
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -------------------------------
# 6️⃣ Load model
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# -------------------------------
# 7️⃣ Training arguments (Transformers 4.57+)
# -------------------------------
training_args = TrainingArguments(
    output_dir="finbert_trained",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),   # 🔥 GPU mixed precision
    logging_steps=50,
    report_to="none"
)

# -------------------------------
# 8️⃣ Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# -------------------------------
# 9️⃣ Train
# -------------------------------
trainer.train()

# -------------------------------
# 🔟 Save model
# -------------------------------
trainer.save_model("finbert_trained")
tokenizer.save_pretrained("finbert_trained")

print("🎉 FINBERT TRAINING COMPLETED SUCCESSFULLY")
