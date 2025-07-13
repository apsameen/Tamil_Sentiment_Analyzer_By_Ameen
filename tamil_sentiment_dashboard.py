# tamil_sentiment_dashboard.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt




# === 1. Load Data ===
X = [
    "அருமையான படம்",
    "மிக மோசமான அனுபவம்",
    "சாதாரணமாக இருந்தது",
    "இனிமையான இசை",
    "பிடிக்கவில்லை",
    "நன்றாக இருந்தது"
]

y = [
    "positive",
    "negative",
    "neutral",
    "positive",
    "negative",
    "neutral"
]

# === 2. Vectorize Tamil Text ===
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# === 3. Train Model ===
model = MultinomialNB()
model.fit(X_vec, y)

# === 4. Save Model and Vectorizer ===
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
