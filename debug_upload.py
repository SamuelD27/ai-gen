#!/usr/bin/env python3
"""
Debug script to test media upload endpoint directly
Run this in Colab after starting the backend
"""

import requests
import io
from PIL import Image
import json

# Create a test JPEG image
def create_test_jpeg():
    img = Image.new('RGB', (512, 512), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return img_byte_arr

print("Creating test JPEG image...")
test_image = create_test_jpeg()

print("Testing media upload endpoint...")
print(f"URL: http://localhost:8000/api/media/upload")

try:
    files = {'file': ('test_upload.jpg', test_image, 'image/jpeg')}
    response = requests.post('http://localhost:8000/api/media/upload', files=files, timeout=30)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        print(f"\n✅ SUCCESS!")
        data = response.json()
        print(f"Response Data:")
        print(json.dumps(data, indent=2))
    else:
        print(f"\n❌ FAILED!")
        print(f"Response Text: {response.text}")

        # Try to parse as JSON
        try:
            error_data = response.json()
            print(f"Error Details:")
            print(json.dumps(error_data, indent=2))
        except:
            pass

except requests.exceptions.ConnectionError as e:
    print(f"\n❌ CONNECTION ERROR: Backend not running or not accessible")
    print(f"Error: {e}")

except requests.exceptions.Timeout as e:
    print(f"\n❌ TIMEOUT: Request took too long")
    print(f"Error: {e}")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR:")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error: {e}")

# Check if backend is running
print("\n" + "="*60)
print("Checking backend health...")
try:
    health = requests.get('http://localhost:8000/health', timeout=5)
    if health.status_code == 200:
        print("✅ Backend is running")
        print(f"Health check: {health.json()}")
    else:
        print(f"⚠️  Backend returned status {health.status_code}")
except:
    print("❌ Backend is not accessible")

# Check auth config
print("\n" + "="*60)
print("Checking auth configuration...")
try:
    auth_config = requests.get('http://localhost:8000/api/auth/config', timeout=5)
    if auth_config.status_code == 200:
        config = auth_config.json()
        print(f"Auth Config: {config}")
        if config.get('auth_enabled'):
            print("⚠️  WARNING: Auth is ENABLED (should be disabled)")
        else:
            print("✅ Auth is disabled")
    else:
        print(f"⚠️  Auth config returned status {auth_config.status_code}")
except Exception as e:
    print(f"❌ Could not check auth config: {e}")
