#!/usr/bin/env python3
"""
Simple GMI API test to understand the correct format
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

GMI_API_KEY = os.getenv("GMI_API_KEY")
print(f"API Key (first 20 chars): {GMI_API_KEY[:20]}...")

url = "https://api.gmi-serving.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GMI_API_KEY}"
}

# Try minimal payload
payload = {
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
}

print("Testing with minimal payload...")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Content: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print("❌ Error!")
        
except Exception as e:
    print(f"Exception: {e}")