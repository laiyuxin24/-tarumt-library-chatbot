from flask import Flask, request, jsonify
from pathlib import Path
import csv

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
subjects = {}
with open(BASE_DIR / "book_subjects.csv", encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        key = row['subject_keyword'].strip()
        subjects[key] = row


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)
    print("=== Incoming request ===")

    print(req)

    intent_name = req['queryResult']['intent']['displayName']
    params = req['queryResult']['parameters']
    book_type = str(params.get('book_type', '')).strip()

    info = subjects.get(book_type)

    if intent_name == 'book_location':
        if info:
            reply = f"{info['subject_keyword']} book are located on Level {info['floor_level']} under {info['ddc_number']}."
        else:
            reply = f"Sorry, couldn't find '{book_type}' in our subject catalog. It might be under General/Fiction (Open Shelf, no fixed floor)."

    elif intent_name == 'book_type':
        if info:
            reply = f"{info['subject_keyword']} books are located on Level {info['floor_level']} under DDC {info['ddc_number']}."
        else:
            reply = f"Sorry, couldn't find '{book_type}' in our subject catalog — try iDiscover for the exact title."

    else:
        reply = "Sorry, I didn't understand that."

    return jsonify({"fulfillmentText": reply})

if __name__ == '__main__':
    app.run(port=5000, debug=True)