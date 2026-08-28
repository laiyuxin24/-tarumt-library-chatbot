TAR UMT LIBRARY CHATBOT - HOW TO START (combined demo, 3 methods, 1 UI)
=========================================================================
One Streamlit app with a sidebar switch between the group's three chatbot
engines. Each engine uses its own dataset optimised for its approach:

  A) TF-IDF + SVM          -> engines/svm_engine.py
                              Dataset: data/chatbot_datasheet.csv
                              (126 intents, 1,597 questions)

  B) Sentence Transformer  -> engines/st_engine.py
                              Dataset: data/chatbot_datasheet.csv
                              (126 intents, 1,597 questions)

  C) Dialogflow            -> engines/dialogflow_engine.py
                              Dataset: data/dialogflow.csv
                              (41 intents, 492 questions)
                              Book location queries are handled
                              separately via webhook + book_subjects.csv
                              (61 subject entries)

NOTE: Engine C uses a smaller, separate dataset (dialogflow.csv) because
book type / location intents are handled dynamically via webhook
fulfillment rather than static classification. Using chatbot_datasheet.csv
for Engine C would produce misleadingly low accuracy scores as those
intents are intentionally excluded from its classification dataset.


STEP 1 - Install dependencies
--------------------------------------------------------
A requirements.txt is included in the project root. Run:

    pip install streamlit pandas scikit-learn sentence-transformers google-cloud-dialogflow


STEP 2 - Train the models
--------------------------------------------------------
Each engine loads a model file from models/ at runtime - these are not
included in the zip, so this step must be run once before the app works:

    python train_models.py

This trains ALL THREE engines in one go:
  - Engine A (TF-IDF + SVM) trains locally on chatbot_datasheet.csv
  - Engine B (Sentence Transformer) trains locally on chatbot_datasheet.csv
  - Engine C (Dialogflow) is evaluated via live API calls to Google
    Dialogflow using dialogflow.csv - requires internet connection.

Each engine prints Accuracy / Precision / Recall / F1 on a held-out 20%
test split (random_state=42, stratify=intent).

If you only want to (re)train one engine individually:

    python engines/svm_engine.py            (Engine A only)
    python engines/st_engine.py             (Engine B only)
    python engines/dialogflow_engine.py     (Engine C only)


STEP 3 - Wake up the Dialogflow Webhook (IMPORTANT for Engine C)
--------------------------------------------------------
Engine C uses a webhook deployed on Render cloud for dynamic book location
queries (e.g. "where is the computer science book?"). The webhook is
hosted at:

    https://library-chatbot-webhook.onrender.com

IMPORTANT: Render's free tier puts the service to sleep after 15 minutes
of inactivity. Before running the app, open the link above in your
browser to wake the service up.

  >> If the page displays "Not Found" - this is NORMAL and EXPECTED. <<

The webhook only defines a POST route (/webhook) for Dialogflow to call
internally. It does not have a GET route for browser access, so "Not
Found" simply means the service is awake and running correctly. It does
NOT mean something is broken.

If you skip this step and go straight to the app, Engine C's book
location queries will return the default response "Checking..." instead
of the actual floor and DDC information. Please wait about 30-60 seconds
after opening the link before starting the app to ensure the service is
fully awake. If you have already launched the Streamlit app before waking
the webhook, simply open the Render link first, wait for it to load, then
reload the Streamlit page in your browser - it will reconnect to the
webhook automatically without needing to restart the app.


STEP 4 - Run the app
--------------------------------------------------------
    python -m streamlit run app.py

Opens at http://localhost:8501. Use the "Choose chatbot engine" radio
button in the sidebar to switch between A / B / C. Each engine keeps its
own separate chat history within the same session. The sidebar also shows
that engine's evaluation metrics from Step 2.

NOTE: Chat history is session-based only. Closing or refreshing the
browser tab will clear the conversation history. This is expected
behaviour for a Streamlit application.


TROUBLESHOOTING
--------------------------------------------------------
- "Run train_models.py first." appears in the Model Metrics box
    -> The model file for that engine does not exist yet in models/, or
       it is in an older format. Re-run Step 2 for that engine.

- Engine C shows "Checking..." instead of book floor information
    -> The Render webhook is still asleep. Open the link in Step 3,
       wait 30-60 seconds, then try again.

- Engine C shows "Not Found" when opening the webhook link
    -> This is normal. See Step 3 explanation above.

- Engine B is slow on first run
    -> The sentence encoder model (~90 MB) is being downloaded for the
       first time. This only happens once and will be cached locally
       after the initial download.

- Changed engines/st_engine.py or any pickled model format
    -> The old models/*.pkl file is stale. Delete it and re-run Step 2
       for that engine.


PROJECT STRUCTURE
--------------------------------------------------------
chatbot_project/
  app.py                        Streamlit UI, engine switcher
  train_models.py               Trains and evaluates all 3 engines
  requirements.txt              All required Python dependencies
  credentials.json              Google Cloud service-account key
                                (already included - do not share or
                                re-zip into anything you distribute)
  data/
    chatbot_datasheet.csv       Dataset for Engine A and Engine B
                                (126 intents, 1,597 questions)
    dialogflow.csv              Dataset for Engine C evaluation
                                (41 intents, 492 questions)
  engines/
    svm_engine.py               A) TF-IDF + SVM
    st_engine.py                B) Sentence Transformer + Logistic Regression
    dialogflow_engine.py        C) Dialogflow
  models/                       Trained model files (generated by Step 2,
                                not included in zip)
  webhook_project/              Dialogflow fulfillment webhook
    webhook.py                  Flask webhook server
    book_subjects.csv           Book subject to floor/DDC lookup table
                                (61 subject entries)
                                Deployed on Render cloud:
                                https://library-chatbot-webhook.onrender.com
