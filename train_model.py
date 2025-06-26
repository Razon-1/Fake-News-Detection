import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 1  # Fake
true["label"] = 0  # Real
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Use 'text' column if available, else 'title'
text_col = "text" if "text" in df.columns else "title"
X = df[text_col].astype(str)
y = df["label"]

# Build pipeline with Random Forest
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "model.pkl")
print("Model saved as model.pkl (Random Forest)")
