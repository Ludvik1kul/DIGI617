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
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "HF_TOKEN not set in environment variables"})
            }

        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        # Always wrap the response safely
        try:
            result = response.json()
        except Exception:
            # If HF returns plain text, wrap it as JSON
            result = {"raw_response": response.text}

        if response.status_code != 200:
            return {
                "statusCode": response.status_code,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Hugging Face API error",
                    "status_code": response.status_code,
                    "details": result
                })
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Backend crash: {str(e)}"})
        }
