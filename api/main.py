# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

from model.sentiment_model import predict_sentiment

# ---------------------------------
# App initialization
# ---------------------------------
app = FastAPI(title="FinBERT Sentiment API")

# ---------------------------------
# Environment variable (News API)
# ---------------------------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

# ---------------------------------
# Request schema
# ---------------------------------
class TextRequest(BaseModel):
    text: str

# ---------------------------------
# Health check
# ---------------------------------
@app.get("/")
def health():
    return {"status": "FinBERT API running"}

# ---------------------------------
# Manual text prediction (existing)
# ---------------------------------
@app.post("/predict")
def predict(req: TextRequest):
    """
    Predict sentiment for a given financial text
    """
    try:
        return predict_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------
# Live news sentiment (FIXED)
# ---------------------------------
@app.get("/news")
def analyze_latest_news(
    query: str = "stock",
    language: str = "en",
    limit: int = 5
):
    """
    Fetch latest financial news and run sentiment analysis
    """
    try:
        # If no API key, use sample data for demo
        if not NEWSDATA_API_KEY:
            print("⚠️ NEWSDATA_API_KEY not set, using sample headlines")
            
            sample_headlines = [
                "Tech stocks rally as earnings beat expectations",
                "Federal Reserve signals potential interest rate cuts",
                "Oil prices decline amid global demand concerns",
                "Banking sector faces regulatory scrutiny",
                "Cryptocurrency market shows mixed signals",
                "Retail sales surge beyond predictions",
                "Manufacturing sector reports contraction",
                "Unemployment rate drops to historic low",
                "Trade tensions ease as negotiations progress",
                "Consumer confidence reaches five-year high"
            ]
            
            results = []
            for title in sample_headlines[:limit]:
                sentiment = predict_sentiment(title)
                results.append({
                    "title": title,
                    "sentiment": sentiment["label"],
                    "confidence": sentiment["confidence"],
                    "probabilities": sentiment.get("probabilities", {})
                })
            
            return {
                "query": query,
                "total_articles": len(results),
                "results": results,
                "note": "Using sample data (API key not configured)"
            }

        # Fetch real news with API key
        url = (
            f"https://newsdata.io/api/1/news"
            f"?apikey={NEWSDATA_API_KEY}"
            f"&q={query}"
            f"&language={language}"
        )

        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"News API error: {response.text}"
            )
        
        data = response.json()

        results = []
        articles = data.get("results", [])
        
        if not articles:
            return {
                "query": query,
                "total_articles": 0,
                "results": [],
                "message": "No articles found for this query"
            }

        for item in articles[:limit]:
            text = item.get("title", "")

            if text:
                sentiment = predict_sentiment(text)
                results.append({
                    "title": text,
                    "sentiment": sentiment["label"],
                    "confidence": sentiment["confidence"],
                    "probabilities": sentiment.get("probabilities", {})
                })

        return {
            "query": query,
            "total_articles": len(results),
            "results": results
        }
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="News API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch news: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")