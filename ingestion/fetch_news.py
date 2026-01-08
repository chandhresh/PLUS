import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")

url = "https://newsdata.io/api/1/latest"
params = {
    "apikey": API_KEY,
    "q": "tesla OR stock OR market",
    "language": "en"
}

response = requests.get(url, params=params)
data = response.json()

os.makedirs("data", exist_ok=True)

with open("data/news_raw.json", "w", encoding="utf-8") as f:
    json.dump(data["results"], f, indent=2)

print("✅ News saved to data/news_raw.json")
