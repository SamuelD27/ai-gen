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
    if not os.path.exists("node_modules"):
        run_command("npm install", "Installing Node modules")
    print("✅ Frontend dependencies installed!\n")

    # Step 5: Setup environment variables
    print("🔑 Configuring API keys...\n")

    env_vars = {
        'HF_TOKEN': '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',
        'HF_HOME': '/content/.cache/huggingface',
        'CIVITAI_API_KEY': '68b35c5249f706b2fdf33a96314628ff',
        'GOOGLE_API_KEY': 'AIzaSyCkIlt1nCc5HDfKjrGvUHknmBj5PqdhTU8',
        'FAL_KEY': '93813d30-be3e-4bad-a0b2-dfe3a16fbb9d:8edebabc3800e0d0a6b46909f18045c8',
        'APP_PATH': '/content/ai-gen'
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
    time.sleep(15)

    # Check if frontend is running
    print("🔍 Checking if services are up...")
    try:
        import requests
        resp = requests.get('http://localhost:5173', timeout=2)
        print("✅ Frontend is running!")
    except Exception as e:
        print(f"⚠️  Frontend check failed: {e}")
        print("📋 Checking logs...")
        run_command("tail -20 /tmp/frontend.log", "")
        run_command("tail -20 /tmp/backend.log", "")

    # Create ngrok tunnel
    print("🌐 Creating public tunnel...\n")
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
