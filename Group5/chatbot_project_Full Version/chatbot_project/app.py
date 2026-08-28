import streamlit as st
import uuid

st.set_page_config(page_title="TARUMT Library FAQ Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 TARUMT Library FAQ Chatbot")
st.caption("Library Chatbot Development")


st.markdown("""
<style>
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    text-align: right;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    text-align: right;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #f0f2f6;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


engine_choice = st.sidebar.radio(
    "Choose chatbot engine:",
    ["A) TF-IDF + SVM", "B) Sentence Transformer", "C) Dialogflow"],
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"A": [], "B": [], "C": []}

engine_key = engine_choice[0]  


@st.cache_resource
def load_svm_bot():
    from engines.svm_engine import SVMChatbot
    return SVMChatbot()


@st.cache_resource
def load_st_bot():
    from engines.st_engine import STChatbot
    return STChatbot()


@st.cache_resource
def load_dialogflow_bot():
    from engines.dialogflow_engine import DialogflowChatbot
    return DialogflowChatbot()


def get_bot_response(text: str):
    if engine_key == "A":
        bot = load_svm_bot()
        return bot.get_response(text)
    elif engine_key == "B":
        bot = load_st_bot()
        return bot.get_response(text)
    else:
        bot = load_dialogflow_bot()
        return bot.get_response(text, st.session_state.session_id)


for msg in st.session_state.chat_history[engine_key]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about library hours, wifi, exam schedule, fees, hostel...")

if user_input:
    st.session_state.chat_history[engine_key].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        result = get_bot_response(user_input)
        reply = result["reply"]
        with st.chat_message("assistant"):
            st.write(reply)
            st.caption(f"Engine: {engine_choice} | Intent: {result.get('intent')} | Confidence: {result.get('confidence', 0):.2f}")
        st.session_state.chat_history[engine_key].append({"role": "assistant", "content": reply})
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {e}")

with st.sidebar:
    st.divider()
    st.subheader("Model Metrics")
    if engine_key == "A":
        try:
            metrics = load_svm_bot().metrics
            st.write(f"Accuracy: {metrics['accuracy']:.2f}")
            st.write(f"Precision: {metrics['precision']:.2f}")
            st.write(f"Recall: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1']:.2f}")
        except Exception:
            st.info("Run train_models.py first.")

    elif engine_key == "B":
        try:
            metrics = load_st_bot().metrics
            st.write(f"Accuracy: {metrics['accuracy']:.2f}")
            st.write(f"Precision: {metrics['precision']:.2f}")
            st.write(f"Recall: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1']:.2f}")
        except Exception:
            st.info("Run train_models.py first.")

    elif engine_key == "C":
        try:
            metrics = load_dialogflow_bot().metrics
            st.write(f"Accuracy: {metrics['accuracy']:.2f}")
            st.write(f"Precision: {metrics['precision']:.2f}")
            st.write(f"Recall: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1']:.2f}")
        except Exception:
            st.info("Run dialogflow_engine.py first.")