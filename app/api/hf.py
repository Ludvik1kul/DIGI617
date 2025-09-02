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

        try:
            result = response.json()  # try JSON first
        except ValueError:
            # fallback if HF returned plain text / HTML
            return {
                "statusCode": response.status_code,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Non-JSON response from Hugging Face",
                    "status_code": response.status_code,
                    "details": response.text
                })
            }

        if response.status_code != 200:
            return {
                "statusCode": response.status_code,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Hugging Face API error",
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
