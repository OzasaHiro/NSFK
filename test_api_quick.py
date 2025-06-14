#!/usr/bin/env python3
"""
Quick API test without running server
"""

import asyncio
import json
from api import app
from fastapi.testclient import TestClient

def test_api():
    """Test the API endpoints"""
    client = TestClient(app)
    
    print("NSFK API Quick Test")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n3. API structure test complete!")
    print("To test video analysis, run:")
    print("  python3 start_api.py")
    print("Then use: curl -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{\"url\":\"https://youtube.com/watch?v=egGAWaRwJgo\"}'")

if __name__ == "__main__":
    test_api()