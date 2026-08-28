import os
import uuid
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# dialogflow.csv do not include the book_type intent because it is separately run in the webhook
BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
DATA_PATH = BASE_DIR / "data" / "dialogflow.csv"          
METRICS_PATH = BASE_DIR / "models" / "dialogflow_metrics.pkl"

PROJECT_ID = "tarumtlibraryfaqbot-dpyr"  


class DialogflowChatbot:
    def __init__(self):
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"credentials.json not found"
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)

        from google.cloud import dialogflow 
        self.dialogflow = dialogflow
        self.session_client = dialogflow.SessionsClient()

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "rb") as f:
                self.metrics = pickle.load(f)
        else:
            self.metrics = None  

    def get_response(self, text: str, session_id: str, language_code: str = "en"):
        session = self.session_client.session_path(PROJECT_ID, session_id)
        text_input = self.dialogflow.TextInput(text=text, language_code=language_code)
        query_input = self.dialogflow.QueryInput(text=text_input)

        response = self.session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        reply = response.query_result.fulfillment_text or "Sorry, I didn't understand that."
        return {
            "reply": reply,
            "intent": response.query_result.intent.display_name or "unknown",
            "confidence": float(response.query_result.intent_detection_confidence),
        }


def new_session_id():
    return str(uuid.uuid4())

def train_and_save():

    df = pd.read_csv(DATA_PATH)

    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["intent"]
    )

    print(f"Loading Dialogflow chatbot, testing on {len(test_df)} questions...")
    bot = DialogflowChatbot()

    true_labels = []
    pred_labels = []

    for i, row in enumerate(test_df.itertuples(), 1):
        question = row.question
        true_intent = row.intent

        session_id = new_session_id()  
        result = bot.get_response(question, session_id)
        predicted_intent = result["intent"]

        true_labels.append(true_intent)
        pred_labels.append(predicted_intent)

        status = "✓" if predicted_intent == true_intent else "✗"
        print(f"[{i}/{len(test_df)}] {status} Q: {question}")
        print(f"    True: {true_intent} | Predicted: {predicted_intent}")

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    print("\n===== Dialogflow Evaluation =====")
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    METRICS_PATH.parent.mkdir(exist_ok=True)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nMetrics saved to {METRICS_PATH}")
    return acc, precision, recall, f1


if __name__ == "__main__":
    train_and_save()