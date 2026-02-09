#!/usr/bin/env python3
"""
Quick test script to verify API is working locally before deploying to Vercel.
Usage: python test_api.py
"""

import requests
import os
import sys
from pathlib import Path

# Test configuration
LOCAL_API_URL = "http://localhost:8000/predict"
VERCEL_API_URL = "https://your-app-name.vercel.app/api/predict"  # Update after deployment

def test_local_api():
    """Test API running locally"""
    print("Testing local API endpoint...")
    print(f"API URL: {LOCAL_API_URL}")
    
    # Check if backend is running
    try:
        # Try to find a test image
        test_image_path = Path("test_image.jpg")
        if not test_image_path.exists():
            print("❌ No test_image.jpg found in current directory")
            print("   Please add a test image or use an existing one")
            return False
        
        # Make request
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(LOCAL_API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            print("✅ Local API is working!")
            result = response.json()
            print(f"✅ Response: {result['data']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to local API")
        print("   Make sure the backend is running:")
        print("   $ cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_vercel_api():
    """Test API running on Vercel (after deployment)"""
    print(f"\nTesting Vercel API endpoint...")
    print(f"API URL: {VERCEL_API_URL}")
    
    if "your-app-name" in VERCEL_API_URL:
        print("⚠️  Update VERCEL_API_URL with your actual Vercel domain first!")
        return False
    
    try:
        # Find test image
        test_image_path = Path("test_image.jpg")
        if not test_image_path.exists():
            print("❌ No test_image.jpg found")
            return False
        
        # Make request
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(VERCEL_API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            print("✅ Vercel API is working!")
            result = response.json()
            print(f"✅ Response: {result['data']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AuraSense AI - API Test Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "local":
            success = test_local_api()
        elif sys.argv[1] == "vercel":
            success = test_vercel_api()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python test_api.py [local|vercel]")
            sys.exit(1)
    else:
        print("\nTesting local API first...")
        success = test_local_api()
        
        if success:
            print("\n✅ All tests passed!")
            print("\nTo test Vercel after deployment:")
            print("  1. Update VERCEL_API_URL in this script")
            print("  2. Run: python test_api.py vercel")
        else:
            print("\n❌ Local API test failed")
            print("Please ensure:")
            print("  1. Backend is running: cd backend && python main.py")
            print("  2. Port 8000 is available")
            print("  3. Models are loaded correctly")
