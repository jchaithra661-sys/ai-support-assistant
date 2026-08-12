"""
Intent classification: TF-IDF + Logistic Regression.

Why this instead of a transformer? On ~200 labeled tickets a linear model on TF-IDF
features is the correct choice (transformers overfit tiny datasets and need far more
data to beat a strong linear baseline). This module is written so swapping in a
transformer fine-tune later is a drop-in replacement — see README "Level up" section.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"


def load_tickets(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def train_intent_classifier(csv_path: str, save: bool = True) -> dict:
    df = load_tickets(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    if save:
        joblib.dump(pipeline, MODEL_PATH)

    return {"pipeline": pipeline, "accuracy": accuracy, "report": report}


def load_classifier(path: str = None):
    return joblib.load(path or MODEL_PATH)


def predict_intent(pipeline, text: str) -> dict:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = float(max(proba))
    return {"intent": pred, "confidence": round(confidence, 3),
            "all_scores": {c: round(float(p), 3) for c, p in zip(classes, proba)}}


if __name__ == "__main__":
    result = train_intent_classifier(str(Path(__file__).parent.parent / "data" / "tickets.csv"))
    print(f"Test accuracy: {result['accuracy']:.3f}\n")
    print(result["report"])
