import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "chatbot_datasheet.csv"
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"


def train_and_save():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["question"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("===== SVM + TF-IDF Evaluation =====")
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    answer_map = df.drop_duplicates("intent").set_index("intent")["answer"].to_dict()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "model": model,
            "answer_map": answer_map,
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        }, f)

    print(f"\nModel saved to {MODEL_PATH}")
    return acc, precision, recall, f1


class SVMChatbot:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.answer_map = data["answer_map"]
        self.metrics = data["metrics"]

    def get_response(self, text: str):
        vec = self.vectorizer.transform([text])
        intent = self.model.predict(vec)[0]
        confidence = max(self.model.predict_proba(vec)[0])
        answer = self.answer_map.get(intent, "Sorry, I don't understand that question.")
        return {"reply": answer, "intent": intent, "confidence": float(confidence)}


if __name__ == "__main__":
    train_and_save()
