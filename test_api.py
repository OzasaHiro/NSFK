#!/usr/bin/env python3
"""
Test script for NSFK API
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def test_analyze_endpoint(youtube_url: str):
    """Test the analyze endpoint"""
    print(f"\nTesting analyze endpoint with URL: {youtube_url}")
    
    payload = {
        "url": youtube_url
    }
    
    try:
        print("Sending analysis request...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Analysis Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== ANALYSIS RESULTS ===")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Duration: {result.get('duration', 'N/A')} seconds")
            print(f"Safety Score: {result.get('safety_score', 'N/A')}/100")
            print(f"Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print(f"Keywords: {', '.join(result.get('keywords', []))}")
            
            # Category scores
            category_scores = result.get('category_scores', {})
            if category_scores:
                print("\nCategory Scores:")
                for category, score in category_scores.items():
                    print(f"  {category}: {score}")
            
            # Risk factors
            risk_factors = result.get('risk_factors', [])
            if risk_factors:
                print(f"\nRisk Factors ({len(risk_factors)}):")
                for i, factor in enumerate(risk_factors[:3], 1):
                    print(f"  {i}. {factor}")
                    
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return False

def test_invalid_url():
    """Test with invalid URL"""
    print("\nTesting with invalid URL...")
    
    payload = {
        "url": "https://example.com/not-youtube"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return 422 (validation error)
        return response.status_code == 422
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("NSFK API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_check()
    root_ok = test_root_endpoint()
    
    if not health_ok:
        print("\n❌ Health check failed. Make sure the API server is running.")
        return
    
    # Test invalid URL
    invalid_url_ok = test_invalid_url()
    
    # Test with a real YouTube URL (short educational video)
    test_url = input("\nEnter a YouTube URL to test (or press Enter for default): ").strip()
    if not test_url:
        # Use a short, safe educational video as default
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - safe test video
        print(f"Using default test URL: {test_url}")
    
    analyze_ok = test_analyze_endpoint(test_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Root Endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
    print(f"Invalid URL Test: {'✅ PASS' if invalid_url_ok else '❌ FAIL'}")
    print(f"Video Analysis: {'✅ PASS' if analyze_ok else '❌ FAIL'}")
    
    if all([health_ok, root_ok, invalid_url_ok, analyze_ok]):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")