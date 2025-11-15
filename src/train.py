import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

DATA_PATH = os.getenv("DATA_PATH", "data/processed/train.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
# Simple label map for later Buy/Hold/Sell
LABEL_TO_SIGNAL = {"positive": "Buy", "neutral": "Hold", "negative": "Sell"}
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = df.dropna(subset=["text", "label"]).sample(frac=1.0, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42,
stratify=df["label"]
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9
        )),
        ("nb", MultinomialNB())
    ])
    # Calibrate to get better probability estimates (optional but useful)
    clf = CalibratedClassifierCV(pipe, cv=3)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, digits=3))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump({"model": clf, "label_to_signal": LABEL_TO_SIGNAL}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
