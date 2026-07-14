# 🛍️ Amazon Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP)** system that classifies Amazon food product reviews as **Positive** or **Negative**, trained on **5,68,454 real customer reviews**. Built entirely from scratch — covering text preprocessing, feature engineering, model training, and evaluation.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-91.2%25-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Model Performance](#-model-performance)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [NLP Pipeline — Step by Step](#-nlp-pipeline--step-by-step)
- [NLP Fundamentals Explained](#-nlp-fundamentals-explained)
- [Model Training](#-model-training)
- [Evaluation](#-evaluation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## Overview

Understanding customer sentiment at scale is a real-world problem faced by every e-commerce company. Manually reading lakhs of reviews isn't feasible — this project builds a **machine learning pipeline that automatically reads a review's raw text and predicts whether the customer felt positively or negatively** about the product.

This is a classic **binary text classification** problem, and this project solves it using traditional (non-deep-learning) NLP techniques — which are faster to train, easier to interpret, and still highly effective for this scale of data.

**Dataset:** Amazon Fine Food Reviews — 5,68,454 reviews, each with review text and a star rating (1–5), which was converted into a binary sentiment label (Positive / Negative).

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 91.2% |
| True Positives | 81,069 |
| True Negatives | 7,687 |
| False Positives | 1,558 |
| False Negatives | 14,844 |

The model correctly classifies sentiment **9 out of 10 times**, which is a strong result for a classical ML approach (no deep learning / transformers used).

---

## ✨ Key Features

- Large-scale NLP applied to **5,68,454 reviews**
- Full text preprocessing pipeline (cleaning, tokenization, stopword removal)
- **TF-IDF vectorization** to convert text into numerical features
- Complete **Exploratory Data Analysis (EDA)** with visualizations
- Model evaluation using **Confusion Matrix** and **ROC Curve**
- 91.2% accuracy on unseen test data

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| NLP | NLTK, TF-IDF |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## 🔄 NLP Pipeline — Step by Step

```
Raw Review Text
      ↓
Text Cleaning (lowercase, remove punctuation/HTML/special chars)
      ↓
Tokenization (splitting text into words)
      ↓
Stopword Removal (removing common words like "the", "is", "a")
      ↓
Stemming / Lemmatization (reducing words to root form)
      ↓
TF-IDF Vectorization (converting text → numeric feature vectors)
      ↓
Train-Test Split
      ↓
Model Training (Classification Algorithm)
      ↓
Evaluation (Accuracy, Confusion Matrix, ROC Curve)
      ↓
Sentiment Prediction (Positive / Negative)
```

---

## 📚 NLP Fundamentals Explained

This section explains **every core NLP concept** used in this project — written so that anyone new to NLP can understand the "why" behind each step, not just the "what."

### 1. Text Cleaning

Raw review text is messy — it contains HTML tags, punctuation, numbers, and inconsistent capitalization. Before a machine learning model can understand text, it needs to be normalized:
- Converting everything to **lowercase** so "Good" and "good" are treated the same
- Removing **punctuation, special characters, and HTML tags**
- Removing extra whitespace

**Why it matters:** Machine learning models don't understand grammar — they see everything as data. Without cleaning, the model would treat `"good"` and `"good!"` as two completely different words.

### 2. Tokenization

Tokenization is the process of breaking a sentence into individual units called **tokens** — usually words.

```
"This product is amazing" → ["This", "product", "is", "amazing"]
```

**Why it matters:** Models work with individual features, not full sentences. Tokenization is the first step that turns unstructured text into something structured enough to analyze.

### 3. Stopword Removal

Stopwords are extremely common words (**"the", "is", "a", "an", "in", "and"**) that appear frequently in almost every sentence but carry little to no sentiment information.

```
"This is a good product" → ["good", "product"]
```

**Why it matters:** Removing stopwords reduces noise and dimensionality, letting the model focus on words that actually carry sentiment (like "good," "bad," "excellent," "terrible").

### 4. Stemming / Lemmatization

Both techniques reduce words to their **root/base form**, so that different forms of a word are treated as the same feature.

- **Stemming** (used here via NLTK): chops off word endings using rules — fast but sometimes crude.
  ```
  "running", "runs", "ran" → "run" (approx.)
  "loved", "loving" → "lov"
  ```
- **Lemmatization**: uses vocabulary and grammar rules to return a real dictionary word (more accurate, slightly slower).

**Why it matters:** Without this step, "love," "loved," and "loving" would be treated as three completely unrelated words, even though they express the same sentiment. This step lets the model generalize better.

### 5. TF-IDF Vectorization (Term Frequency – Inverse Document Frequency)

Machine learning models require **numbers**, not words. TF-IDF is a technique that converts cleaned text into a numeric vector, weighing each word by importance:

- **Term Frequency (TF):** How often a word appears in a specific review.
- **Inverse Document Frequency (IDF):** How rare or common that word is *across all reviews*. Words that appear in almost every review (even after stopword removal) get a lower weight; rare, distinctive words get a higher weight.

```
TF-IDF(word) = TF(word in review) × IDF(word across all reviews)
```

**Why it matters:** A word like "amazing" that appears rarely but strongly signals sentiment gets a high score. A word that appears in nearly every review (but wasn't caught as a stopword) gets down-weighted, since it doesn't help distinguish positive from negative reviews.

### 6. Train-Test Split

The dataset is split into a **training set** (used to teach the model) and a **test set** (used to evaluate how well it generalizes to unseen reviews). This prevents the model from being evaluated on data it has already memorized.

### 7. Classification Model

Once every review is represented as a TF-IDF vector, it becomes a standard **supervised binary classification problem** — the model learns the relationship between word patterns and sentiment labels (Positive/Negative) from the training data, then predicts sentiment for new, unseen reviews.

---

## 🧠 Model Training

The cleaned and vectorized text (TF-IDF features) is fed into a classical machine learning classifier via **Scikit-learn**, trained on the majority of the 5,68,454 reviews and validated on a held-out test set. The choice of a classical ML model (over deep learning) makes the pipeline lightweight, fast to train, and easy to interpret — while still achieving strong accuracy at this data scale.

---

## 📊 Evaluation

Model performance isn't judged by accuracy alone — two additional tools are used:

- **Confusion Matrix:** Breaks predictions into True Positives, True Negatives, False Positives, and False Negatives — showing *exactly* what kind of mistakes the model makes (e.g., is it more likely to miss negative reviews than positive ones?).
- **ROC Curve (Receiver Operating Characteristic):** Plots the model's ability to distinguish between the two classes across different decision thresholds — a model that hugs the top-left corner of the curve is a strong classifier.

---

## ▶ How to Run

1. Open `amazon_sentiment_prediction_model.ipynb` in **Google Colab**
2. Click **Runtime → Run all**
3. The model will train automatically on the dataset
4. Results (accuracy, confusion matrix, ROC curve) will be generated at the end of the notebook

---

## 📁 Project Structure

```
├── amazon_sentiment_prediction_model.ipynb   # Full pipeline: EDA → preprocessing → training → evaluation
└── README.md
```

---

## 🚀 Future Improvements

- [ ] Deploy as a live web app (Streamlit/FastAPI) for real-time sentiment prediction
- [ ] Compare TF-IDF + classical ML against deep learning approaches (LSTM/BERT)
- [ ] Add multi-class sentiment (Positive / Neutral / Negative) instead of binary
- [ ] Experiment with n-grams (bigrams/trigrams) to capture phrase-level sentiment
- [ ] Hyperparameter tuning for further accuracy improvement

---

## 👤 Author

Built as part of an NLP/ML portfolio, demonstrating a complete text classification pipeline from raw text to a deployable, evaluated model.




