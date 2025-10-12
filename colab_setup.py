"""
ai-gen Google Colab Setup Script
Run this in Google Colab to set up everything automatically.

Usage:
    !wget https://raw.githubusercontent.com/SamuelD27/ai-gen/main/colab_setup.py
    !python colab_setup.py
"""

import os
import sys
import subprocess
import time
import json

def run_command(cmd, description=""):
    """Run a shell command and display output"""
    if description:
        print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("=" * 70)
    print("🎨 ai-gen - Google Colab Setup")
    print("=" * 70)
    print("\nSetting up your environment with:")
    print("  • LoRA training (Flux models)")
    print("  • Ultra-realistic image generation")
    print("  • Web GUI with ngrok tunnel")
    print("  • All API keys pre-configured")
    print("\n" + "=" * 70 + "\n")

    # Check if we're in Colab
    try:
        import google.colab
        IS_COLAB = True
        print("✅ Running in Google Colab\n")
    except:
        IS_COLAB = False
        print("⚠️  Not running in Colab, but continuing anyway...\n")

    # Check for Pillow version conflict early
    try:
        from PIL import Image
        # Test if it actually works
        Image.new('RGB', (1, 1))
    except (ImportError, AttributeError) as e:
        print("⚠️  Pillow version conflict detected!")
        print("🔧 Fixing Pillow installation...\n")
        subprocess.run(["pip", "uninstall", "-y", "pillow"], capture_output=True)
        subprocess.run(["pip", "install", "--no-cache-dir", "--force-reinstall", "pillow==10.1.0"], capture_output=True)
        print("✅ Pillow fixed! Restarting script...\n")
        print("=" * 70)
        # Re-run the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return

    # Step 1: Check GPU
    print("🔍 Checking GPU...")
    run_command("nvidia-smi", "")

    # Step 2: Clone repository
    if not os.path.exists('/content/ai-gen'):
        print("\n📥 Cloning ai-gen repository...")
        run_command("git clone https://github.com/SamuelD27/ai-gen.git /content/ai-gen",
                   "Cloning repository")
    else:
        print("\n📁 Repository exists, pulling latest changes...")
        run_command("cd /content/ai-gen && git pull", "Updating repository")

    os.chdir('/content/ai-gen')
    print("✅ Repository ready!\n")

    # Step 3: Install Node.js (required for frontend)
    print("📦 Installing Node.js...\n")
    node_installed = run_command("which node", "")
    if not node_installed:
        print("Installing Node.js 20.x...")
        run_command("curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -", "")
        run_command("sudo apt-get install -y nodejs", "")
        print("✅ Node.js installed!")
    else:
        print("✅ Node.js already installed!")
    print("")
    # Step 4: Install Python dependencies
    print("📦 Installing Python dependencies (this may take a few minutes)...\n")
    run_command("pip install -q -r requirements.txt", "Installing main requirements")
    run_command("pip install -q -r charforge-gui/backend/requirements.txt",
               "Installing GUI backend requirements")
    run_command("pip install -q pyngrok", "Installing ngrok")
    print("\n✅ Dependencies installed!\n")

    # Install frontend dependencies now
    print("\n📦 Installing frontend dependencies...\n")
    os.chdir("/content/ai-gen/charforge-gui/frontend")

    # Clean install to fix rollup native module issue
    if os.path.exists("node_modules"):
        print("Cleaning old node_modules...")
        run_command("rm -rf node_modules package-lock.json", "")

    print("Installing Node modules (this may take 3-4 minutes)...")
    run_command("npm install --legacy-peer-deps", "Installing Node modules")
    print("✅ Frontend dependencies installed!\n")

    # Step 5: Setup environment variables
    print("🔑 Configuring API keys...\n")

    env_vars = {
        'HF_TOKEN': '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',
        'HF_HOME': '/content/.cache/huggingface',
        'CIVITAI_API_KEY': '68b35c5249f706b2fdf33a96314628ff',
        'GOOGLE_API_KEY': 'AIzaSyCkIlt1nCc5HDfKjrGvUHknmBj5PqdhTU8',
        'FAL_KEY': '93813d30-be3e-4bad-a0b2-dfe3a16fbb9d:8edebabc3800e0d0a6b46909f18045c8'
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # Create root .env
    with open('/content/ai-gen/.env', 'w') as f:
        f.write("# ai-gen Environment Variables\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    # Create backend .env
    with open('/content/ai-gen/charforge-gui/backend/.env', 'w') as f:
        f.write("# Backend Environment Variables\n")
        f.write("SECRET_KEY=colab-secret-key-change-in-production\n")
        f.write("DATABASE_URL=sqlite:///./database.db\n")
        f.write("ENABLE_AUTH=false\n")
        f.write("ALLOW_REGISTRATION=false\n")
        f.write("ENVIRONMENT=development\n")  # Enable detailed error messages
        f.write("COMFYUI_PATH=/content/ai-gen/ComfyUI\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print("✅ API keys configured!\n")

    # Step 6: Skip model downloads (will download on-demand)
    print("✅ Skipping model pre-download (models will be downloaded automatically when needed)\n")

    # Step 7: Setup ngrok
    print("🌐 Setting up ngrok tunnel...\n")
    run_command("pip install -q pyngrok", "Installing pyngrok")

    # Configure ngrok
    from pyngrok import ngrok
    ngrok.set_auth_token("33u4PSfJRAAdkBVl0lmMTo7LebK_815Q5PcJK6h68hM5PUAyM")
    print("✅ ngrok configured!\n")

    # Step 8: Start services
    print("🚀 Starting ai-gen GUI...\n")

    os.chdir('/content/ai-gen/charforge-gui')

    # Start backend in background with nohup
    print("🔧 Starting backend...")
    os.chdir('/content/ai-gen/charforge-gui/backend')
    with open('/tmp/backend.log', 'w') as log:
        backend_process = subprocess.Popen(
            ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'],
            stdout=log,
            stderr=subprocess.STDOUT
        )

    print("⏳ Waiting for backend to start...")
    time.sleep(5)

    # Check if backend is running
    try:
        import requests
        resp = requests.get('http://localhost:8000/health', timeout=2)
        print("✅ Backend is running!")
    except:
        print("⚠️  Backend might not be ready yet, continuing...")


    # Start frontend in background
    print("🎨 Starting frontend...")
    with open('/tmp/frontend.log', 'w') as log:
        frontend_process = subprocess.Popen(
            ['npm', 'run', 'dev', '--', '--host', '0.0.0.0', '--port', '5173'],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd='/content/ai-gen/charforge-gui/frontend'
        )

    print("⏳ Waiting for frontend to start...")

    # Monitor frontend startup
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)

        # Check if process died
        if frontend_process.poll() is not None:
            print(f"❌ Frontend process died! Exit code: {frontend_process.returncode}")
            print("📋 Last 30 lines of frontend log:")
            run_command("tail -30 /tmp/frontend.log", "")
            break

        # Check if port is listening using Python socket
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 5173))
            sock.close()
            if result == 0:
                print(f"✅ Frontend is listening on port 5173! (after {i+1}s)")
                break
        except Exception as e:
            pass  # Port not ready yet

        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}s)")

    # Final health check
    print("🔍 Checking if services are up...")
    try:
        import requests
        resp = requests.get('http://localhost:5173', timeout=2)
        print("✅ Frontend HTTP check passed!")
    except Exception as e:
        print(f"⚠️  Frontend check failed: {e}")
        print("📋 Checking process status...")
        run_command("ps aux | grep -E 'vite|npm'", "")
        print("📋 Last 30 lines of logs:")
        run_command("tail -30 /tmp/frontend.log", "")
        run_command("tail -20 /tmp/backend.log", "")

    # Run comprehensive API tests
    print("\n" + "=" * 70)
    print("🧪 Running API Tests")
    print("=" * 70 + "\n")

    import requests
    import io
    from PIL import Image

    test_results = {}

    # Test 1: Backend health
    print("1️⃣ Testing backend health...")
    try:
        resp = requests.get('http://localhost:8000/health', timeout=5)
        if resp.status_code == 200:
            print("   ✅ Backend health check passed")
            test_results['health'] = True
        else:
            print(f"   ❌ Backend returned {resp.status_code}")
            test_results['health'] = False
    except Exception as e:
        print(f"   ❌ Backend not responding: {e}")
        test_results['health'] = False

    # Test 2: Auth config
    print("\n2️⃣ Testing auth configuration...")
    try:
        resp = requests.get('http://localhost:8000/api/auth/config', timeout=5)
        if resp.status_code == 200:
            config = resp.json()
            if config.get('auth_enabled') == False:
                print(f"   ✅ Auth is disabled: {config}")
                test_results['auth'] = True
            else:
                print(f"   ⚠️  Auth is ENABLED (should be disabled): {config}")
                test_results['auth'] = False
        else:
            print(f"   ❌ Auth config returned {resp.status_code}")
            test_results['auth'] = False
    except Exception as e:
        print(f"   ❌ Auth config failed: {e}")
        test_results['auth'] = False

    # Test 3: Directory structure
    print("\n3️⃣ Checking directories...")
    dirs_to_check = [
        '/content/ai-gen/media',
        '/content/ai-gen/uploads',
        '/content/ai-gen/results'
    ]
    all_dirs_exist = True
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path} exists")
        else:
            print(f"   ❌ {dir_path} MISSING")
            all_dirs_exist = False
    test_results['directories'] = all_dirs_exist

    # Test 4: Media upload
    print("\n4️⃣ Testing media upload...")
    try:
        # Create test image using numpy for better compatibility
        import numpy as np
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        img_array[:, :] = [255, 0, 0]  # Red color
        img = Image.fromarray(img_array)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # Upload
        files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
        resp = requests.post('http://localhost:8000/api/media/upload', files=files, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Upload successful!")
            print(f"      Filename: {data.get('filename')}")
            print(f"      Size: {data.get('file_size')} bytes")
            print(f"      URL: {data.get('file_url')}")
            test_results['upload'] = True

            # Clean up test file
            try:
                requests.delete(f"http://localhost:8000/api/media/files/{data.get('filename')}", timeout=5)
                print(f"   🗑️  Test file cleaned up")
            except:
                pass
        else:
            print(f"   ❌ Upload failed: {resp.status_code}")
            print(f"      Response: {resp.text}")
            test_results['upload'] = False
    except Exception as e:
        import traceback
        print(f"   ❌ Upload test failed: {e}")
        print(f"   📋 Traceback: {traceback.format_exc()}")
        test_results['upload'] = False

    # Test 5: Media list
    print("\n5️⃣ Testing media list...")
    try:
        resp = requests.get('http://localhost:8000/api/media/files', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Media list works (found {data.get('total', 0)} files)")
            test_results['media_list'] = True
        else:
            print(f"   ❌ Media list failed: {resp.status_code}")
            test_results['media_list'] = False
    except Exception as e:
        print(f"   ❌ Media list failed: {e}")
        test_results['media_list'] = False

    # Test summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    print(f"\nPassed: {passed}/{total}")
    for test, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test}")

    if passed == total:
        print("\n🎉 All tests passed! System is fully operational!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("📋 Backend logs:")
        run_command("tail -50 /tmp/backend.log", "")

    # Create ngrok tunnel
    print("\n" + "=" * 70)
    print("🌐 Creating public tunnel...")
    print("=" * 70 + "\n")
    public_url = ngrok.connect(5173, bind_tls=True)

    print("\n" + "=" * 70)
    print("🎉 ai-gen is Running!")
    print("=" * 70)
    print(f"\n📱 Access your GUI here:\n")
    print(f"   🌐 {public_url}\n")
    print("💡 Click the link above to access your ai-gen interface")
    print("💡 On first visit, click 'Visit Site' on the ngrok warning page")
    print("\n🔧 Backend API: http://localhost:8000/docs")
    print("\n⚠️  Keep this script running! Stopping it will shut down the GUI.")
    print("=" * 70 + "\n")

    # Keep running
    try:
        print("✅ Services running. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.kill()
        frontend_process.kill()
        ngrok.kill()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()
