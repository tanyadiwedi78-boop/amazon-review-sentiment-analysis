from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

lem = WordNetLemmatizer()
sw  = set(stopwords.words("english"))
sw.update(["br","one","get","also","would","like","product"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lem.lemmatize(t) for t in tokens
            if t not in sw and len(t) > 2]
    return " ".join(tokens)

app = FastAPI(title="Amazon Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_saved"

clf   = joblib.load(MODEL_DIR / "sentiment_model.pkl")  # ← fix
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API Running ✅"}

@app.post("/predict")
def predict_sentiment(review: ReviewRequest):
    cleaned    = clean_text(review.text)
    X          = tfidf.transform([cleaned])
    pred       = clf.predict(X)[0]
    prob       = clf.predict_proba(X)[0]    # ← fix
    sentiment  = "Positive" if pred == 1 else "Negative"
    confidence = round(float(max(prob)) * 100, 2)
    return {
        "sentiment" : sentiment,
        "confidence": confidence,
        "label"     : int(pred)
    }

@app.post("/predict/batch")
def predict_batch(reviews: list[ReviewRequest]):
    results = []
    for r in reviews:
        cleaned = clean_text(r.text)        # ← fix (r nahi review)
        X       = tfidf.transform([cleaned])
        pred    = clf.predict(X)[0]
        prob    = clf.predict_proba(X)[0]   # ← fix
        results.append({
            "text"      : r.text,
            "sentiment" : "Positive" if pred == 1 else "Negative",
            "confidence": round(float(max(prob)) * 100, 2),
        })
    return {"results": results}