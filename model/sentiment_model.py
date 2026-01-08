import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer (safe)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# model — FORCE SAFETENSORS (this fixes the error)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    device_map=None,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True   # 🔥 THIS IS THE KEY
)

model.to(device)
model.eval()

LABELS = ["negative", "neutral", "positive"]


def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0]

    probabilities = {
        LABELS[i]: float(probs[i])
        for i in range(len(LABELS))
    }

    label = max(probabilities, key=probabilities.get)

    return {
        "label": label,
        "confidence": round(probabilities[label], 4),
        "probabilities": probabilities
    }
