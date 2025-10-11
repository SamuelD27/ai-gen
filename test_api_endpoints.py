#!/usr/bin/env python3
"""
Comprehensive API endpoint tester for CharForge GUI
Tests all endpoints to ensure they work without authentication
"""

import requests
import json
import io
from PIL import Image

BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a test image in memory."""
    img = Image.new('RGB', (512, 512), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health():
    """Test health check endpoint."""
    print("Testing: GET /health")
    response = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    return response.status_code == 200

def test_auth_config():
    """Test auth configuration endpoint."""
    print("\nTesting: GET /api/auth/config")
    response = requests.get(f"{BASE_URL}/api/auth/config")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Response: {data}")
    print(f"  Auth Enabled: {data.get('auth_enabled')}")
    return response.status_code == 200 and data.get('auth_enabled') == False

def test_media_upload():
    """Test media file upload."""
    print("\nTesting: POST /api/media/upload")
    img_bytes = create_test_image()
    files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/api/media/upload", files=files)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Uploaded: {data.get('filename')}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_media_list():
    """Test listing media files."""
    print("\nTesting: GET /api/media/files")
    response = requests.get(f"{BASE_URL}/api/media/files")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Total files: {data.get('total')}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_media_delete(filename):
    """Test deleting a media file."""
    print(f"\nTesting: DELETE /api/media/files/{filename}")
    response = requests.delete(f"{BASE_URL}/api/media/files/{filename}")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print(f"  Message: {response.json().get('message')}")
        return True
    else:
        print(f"  Error: {response.text}")
        return False

def test_models():
    """Test models listing endpoint."""
    print("\nTesting: GET /api/models/models")
    response = requests.get(f"{BASE_URL}/api/models/models")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Checkpoints: {len(data.get('checkpoints', []))}")
        print(f"  LoRAs: {len(data.get('loras', []))}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_schedulers():
    """Test schedulers listing endpoint."""
    print("\nTesting: GET /api/models/schedulers")
    response = requests.get(f"{BASE_URL}/api/models/schedulers")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Schedulers: {len(data.get('schedulers', []))}")
        return True
    else:
        print(f"  Error: {response.text}")
        return False

def test_trainers():
    """Test trainers listing endpoint."""
    print("\nTesting: GET /api/models/trainers")
    response = requests.get(f"{BASE_URL}/api/models/trainers")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Trainers: {len(data.get('trainers', []))}")
        return True
    else:
        print(f"  Error: {response.text}")
        return False

def test_dataset_list():
    """Test listing datasets."""
    print("\nTesting: GET /api/datasets/datasets")
    response = requests.get(f"{BASE_URL}/api/datasets/datasets")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Total datasets: {data.get('total')}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_dataset_create(image_filename):
    """Test creating a dataset."""
    print("\nTesting: POST /api/datasets/datasets")
    payload = {
        "name": "Test Dataset",
        "trigger_word": "testchar",
        "caption_template": "a photo of {trigger} person",
        "auto_caption": False,
        "resize_images": True,
        "crop_images": True,
        "flip_images": False,
        "quality_filter": "basic",
        "selected_images": [image_filename]
    }
    response = requests.post(f"{BASE_URL}/api/datasets/datasets", json=payload)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Dataset ID: {data.get('id')}")
        print(f"  Name: {data.get('name')}")
        print(f"  Status: {data.get('status')}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_characters_list():
    """Test listing characters."""
    print("\nTesting: GET /api/training/characters")
    response = requests.get(f"{BASE_URL}/api/training/characters")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Total characters: {len(data)}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def test_inference_available_characters():
    """Test listing available characters for inference."""
    print("\nTesting: GET /api/inference/available-characters")
    response = requests.get(f"{BASE_URL}/api/inference/available-characters")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Available characters: {len(data)}")
        return data
    else:
        print(f"  Error: {response.text}")
        return None

def run_all_tests():
    """Run all API tests."""
    print("=" * 60)
    print("CharForge GUI API Endpoint Tests")
    print("=" * 60)

    results = {}

    # Basic tests
    results['health'] = test_health()
    results['auth_config'] = test_auth_config()

    # Media tests
    uploaded_file = test_media_upload()
    results['media_upload'] = uploaded_file is not None

    media_list = test_media_list()
    results['media_list'] = media_list is not None

    # Models tests
    results['models'] = test_models() is not None
    results['schedulers'] = test_schedulers()
    results['trainers'] = test_trainers()

    # Dataset tests
    dataset_list = test_dataset_list()
    results['dataset_list'] = dataset_list is not None

    if uploaded_file:
        dataset = test_dataset_create(uploaded_file['filename'])
        results['dataset_create'] = dataset is not None

        # Clean up: delete uploaded file
        if uploaded_file:
            test_media_delete(uploaded_file['filename'])

    # Training tests
    characters = test_characters_list()
    results['characters_list'] = characters is not None

    # Inference tests
    available_chars = test_inference_available_characters()
    results['inference_available'] = available_chars is not None

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test}")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
