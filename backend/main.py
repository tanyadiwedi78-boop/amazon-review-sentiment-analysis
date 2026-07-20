from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from groq import Groq
import os, sys

load_dotenv()
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ============ Text Cleaning ===============================================
lem = WordNetLemmatizer()
sw = set(stopwords.words("english"))
sw.update(["br", "one", "get", "also", "would", "like", "product"])


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)          
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lem.lemmatize(t) for t in tokens
            if t not in sw and len(t) > 2]
    return " ".join(tokens)


# ==================== Groq Client ===================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set -- AI insights will fail. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)


def analyze_with_groq(review_text: str, sentiment: str, confidence: float):
    prompt = f"""You are an expert Amazon Product review analyst.
Review: "{review_text}"
Sentiment: {sentiment} ({confidence}% confidence)

Write a 3-4 sentence analysis paragraph that MUST include:
1. Exact product category (e.g., Clothing/Beverages/Electronics/Food/etc)
2. Specific reasons WHY this review is {sentiment} (mention exact issues/positives from the review)
3. Which aspect affected the customer most
4. Business impact of this review

Rules:
- Be specific, mention exact problems/positives from the review
- No bullet points, no JSON
- Plain paragraph only
- Sound like a professional analyst
- Start with "This is a [category] product review..."

Now write the analysis for the given review:"""

    try:
        response = client.chat.completions.create(     
            model="openai/gpt-oss-20b",                  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Don't let a Groq failure crash the whole /predict endpoint --
        # the sentiment prediction itself still works even if AI insights fail.
        print(f"Groq API error: {e}")
        return f"(AI insights unavailable: {str(e)})"


app = FastAPI(title="Amazon Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_saved"

clf = joblib.load(MODEL_DIR / "sentiment_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "API Running", "groq_configured": GROQ_API_KEY is not None}


@app.post("/predict")
def predict_sentiment(review: ReviewRequest):
    cleaned = clean_text(review.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text became empty after cleaning")

    X = tfidf.transform([cleaned])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    sentiment = "Positive" if pred == 1 else "negative"
    confidence = round(float(max(prob)) * 100, 2)

    ai_insights = analyze_with_groq(review.text, sentiment, confidence)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "label": int(pred),
        "ai_insights": ai_insights,
    }


@app.post("/predict/batch")
def predict_batch(reviews: list[ReviewRequest]):
    results = []
    for r in reviews:
        cleaned = clean_text(r.text)
        X = tfidf.transform([cleaned])
        pred = clf.predict(X)[0]
        prob = clf.predict_proba(X)[0]
        results.append({
            "text": r.text,
            "sentiment": "Positive" if pred == 1 else "negative",
            "confidence": round(float(max(prob)) * 100, 2),
        })
    return {"results": results}