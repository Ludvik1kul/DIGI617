#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer,AutoModelForTokenClassification
from dotenv import load_dotenv  
from werkzeug.exceptions import HTTPException
import numpy as np
load_dotenv()

app = Flask(__name__)

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# -------------------------------
# Preload all models/pipelines
# -------------------------------
pipelines = {}

# 1. Kushtrim sentiment (sst2)
tokenizer1 = AutoTokenizer.from_pretrained(
    "Kushtrim/norbert3-large-norsk-sentiment-sst2",
    trust_remote_code=True,
    token=hf_token
)
model1 = AutoModelForSequenceClassification.from_pretrained(
    "Kushtrim/norbert3-large-norsk-sentiment-sst2",
    trust_remote_code=True,
    dtype="auto",
    token=hf_token
)
pipelines["sentiment1"] = pipeline("text-classification", model=model1, tokenizer=tokenizer1)

# 2. ltg/norbert3-large_sentence-sentiment
tokenizer2 = AutoTokenizer.from_pretrained(
    "ltg/norbert3-large_sentence-sentiment",
    trust_remote_code=True,
    token=hf_token
)
model2 = AutoModelForSequenceClassification.from_pretrained(
    "ltg/norbert3-large_sentence-sentiment",
    trust_remote_code=True,
    dtype="auto",
    token=hf_token
)
pipelines["sentiment2"] = pipeline("text-classification", model=model2, tokenizer=tokenizer2)

# 3. ltg/norbert3-large_TSA
tokenizer3 = AutoTokenizer.from_pretrained(
    "ltg/norbert3-large_TSA",
    trust_remote_code=True,
    token=hf_token
)

model3 = AutoModelForTokenClassification.from_pretrained(
    "ltg/norbert3-large_TSA",
    trust_remote_code=True,
    token=hf_token
)

pipelines["TSA"] = pipeline(
    "token-classification",
    model=model3,
    tokenizer=tokenizer3,
    aggregation_strategy="simple"
)
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
    model_key = data.get("model", "sentiment1")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if model_key not in pipelines:
        return jsonify({"error": "Unknown model"}), 400

    try:
        result = pipelines[model_key](text)

        # Special handling for TSA model (token classification)
        if model_key == "TSA":
            grouped = {}
            for item in result:
                label = item.get("entity_group", item.get("label", "UNK"))
                word = item.get("word")
                score = float(item.get("score", 0))

                if label not in grouped:
                    grouped[label] = []

                grouped[label].append({"word": word, "score": score})

            result = grouped

        # Convert numpy/torch floats to regular Python floats
        def to_python(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_python(v) for v in obj]
            return obj

        result = to_python(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)
# Global error handler to always return JSON
@app.errorhandler(Exception)
def handle_exception(e):
    # If it's a HTTPException, use its code and description
    if isinstance(e, HTTPException):
        return jsonify({
            "error": e.description
        }), e.code
    
    # Otherwise it's some internal server error
    return jsonify({
        "error": str(e)
    }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
