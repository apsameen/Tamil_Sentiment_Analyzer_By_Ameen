import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
