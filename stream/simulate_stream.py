import json
import time
import csv
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

with open("data/news_raw.json", encoding="utf-8") as f:
    news = json.load(f)

with open("data/sentiment_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "title", "label", "score"])

for article in news:
    title = article.get("title", "")
    payload = {"text": title}

    response = requests.post(API_URL, json=payload)
    result = response.json()

    with open("data/sentiment_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            title[:60],
            result["label"],
            result["score"]
        ])

    print(title, "→", result)
    time.sleep(2)
