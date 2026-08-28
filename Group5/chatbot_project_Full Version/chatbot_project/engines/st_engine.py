"""
Sentence-Transformers engine (Method B): every question is embedded with a
pretrained Sentence-BERT model, then a Logistic Regression classifier is
trained on top of those embeddings to predict one of the 126 intents in
data/complete_datasheet.csv.

This replaces the earlier nearest-neighbour version (which just took the
single closest training example by raw cosine similarity, ~0.71 top-1
accuracy on the held-out split) with a proper classifier head - the model
learns which directions in embedding space actually separate the 126
intents instead of trusting one nearest example every time, which matters
here since several intents (e.g. the ~15 "<subject>_book" location intents)
sit close together in embedding space.

Uses the same DATA_PATH / train_test_split(test_size=0.2, random_state=42,
stratify=...) as engines/svm_engine.py so all three methods are evaluated
on the *same* held-out test rows - that's what makes the accuracy/F1
numbers in app.py's sidebar actually comparable across A/B/C.
"""

import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "chatbot_datasheet.csv"
MODEL_PATH = BASE_DIR / "models" / "st_classifier.pkl"

# Same model name the original nearest-neighbour version used - if it's
# already been downloaded once on this machine, this won't re-download it.
MODEL_NAME = "all-MiniLM-L6-v2"

# Below this confidence, treat it as "didn't understand" rather than
# returning a (probably wrong) canned answer for the top-scoring intent.
CONFIDENCE_THRESHOLD = 0.35


def train_and_save():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["question"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    print(f"Loading sentence encoder ({MODEL_NAME})...")
    encoder = SentenceTransformer(MODEL_NAME)

    print("Encoding train/test questions...")
    X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=False)
    X_test_emb = encoder.encode(X_test.tolist(), show_progress_bar=False)

    classifier = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    classifier.fit(X_train_emb, y_train)

    y_pred = classifier.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("===== Sentence Transformer (embeddings + Logistic Regression) Evaluation =====")
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    answer_map = df.drop_duplicates("intent").set_index("intent")["answer"].to_dict()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "classifier": classifier,
            "answer_map": answer_map,
            # same 4 keys as engines/svm_engine.py's metrics dict, so the
            # sidebar can show Accuracy/Precision/Recall/F1 for B the same
            # way it already does for A and C.
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        }, f)

    print(f"\nModel saved to {MODEL_PATH}")
    return acc, precision, recall, f1


class STChatbot:
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.encoder = SentenceTransformer(MODEL_NAME)
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.answer_map = data["answer_map"]
        self.metrics = data["metrics"]
        self.threshold = threshold

    def get_response(self, text: str):
        emb = self.encoder.encode([text])
        probs = self.classifier.predict_proba(emb)[0]
        best_idx = int(probs.argmax())
        intent = self.classifier.classes_[best_idx]
        confidence = float(probs[best_idx])

        if confidence < self.threshold:
            return {
                "reply": "Sorry, I don't understand that question. Could you rephrase it?",
                "intent": "unknown",
                "confidence": confidence,
            }

        answer = self.answer_map.get(intent, "Sorry, I don't understand that question.")
        return {"reply": answer, "intent": intent, "confidence": confidence}


if __name__ == "__main__":
    train_and_save()
