#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv  

load_dotenv()

app = Flask(__name__)

# -------------------------------
# Load model once at startup
# -------------------------------
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    "Kushtrim/norbert3-large-norsk-sentiment-sst2",
    trust_remote_code=True,
    use_auth_token=hf_token
)

model = AutoModelForSequenceClassification.from_pretrained(
    "Kushtrim/norbert3-large-norsk-sentiment-sst2",
    trust_remote_code=True,
    torch_dtype="auto",
    use_auth_token=hf_token
)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)



# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("inputs", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Run sentiment analysis
    try:
        result = classifier(text)
    except Exception as e:
        return jsonify({"Error": "Kontakt- lsl007@uib.no"}), 500

    return jsonify(result)

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
