# api/hf.py
import os
import json
import requests

API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}

def handler(request):
    try:
        data = request.json()
        text = data.get("inputs", "")

        if not headers["Authorization"]:
            raise RuntimeError("HF_TOKEN not set in environment variables")

        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")

        result = response.json()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
