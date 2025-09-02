# api/hf.py
import os
import requests

API_URL = "https://router.huggingface.co/hf-inference/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

def handler(request):
    try:
        data = request.json()  # Get JSON body from frontend
        text = data.get("inputs", "")

        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": result
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }
