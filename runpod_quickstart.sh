#!/bin/bash

# ====================================================
# CharForge RunPod Quick-Start Script
# ====================================================
# One command to setup and launch CharForge on RunPod
# Usage: bash runpod_quickstart.sh
# ====================================================

set -e  # Exit on error

echo "🚀 CharForge RunPod Quick-Start"
echo "================================"
echo ""

# Navigate to workspace
cd /workspace 2>/dev/null || cd ~

# Install Node.js if not present (RunPod doesn't include it by default)
if ! command -v node &> /dev/null; then
    echo "📦 Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    echo "✅ Node.js installed: $(node --version)"
else
    echo "✅ Node.js already installed: $(node --version)"
fi

# Determine if we need to clone or just pull
REPO_DIR="ai-gen"

if [ -d "$REPO_DIR" ]; then
    echo "📁 Found existing $REPO_DIR directory"
    cd "$REPO_DIR"
    echo "📥 Pulling latest changes..."
    git pull || echo "⚠️  Could not pull (no git repo or no changes)"
else
    echo "📥 Cloning repository..."
    git clone https://github.com/SamuelD27/ai-gen.git
    cd "$REPO_DIR"
fi

echo ""
echo "🔧 Setting up environment..."

# Create root .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating root .env file..."
    cat > .env << 'ENVEOF'
# CharForge Environment Variables
# Your API Keys - Pre-configured

# Hugging Face
HF_TOKEN=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
HF_HOME=/workspace/.cache/huggingface

# CivitAI
CIVITAI_API_KEY=68b35c5249f706b2fdf33a96314628ff

# Google AI (for captioning)
GOOGLE_API_KEY=AIzaSyCkIlt1nCc5HDfKjrGvUHknmBj5PqdhTU8

# fal.ai
FAL_KEY=93813d30-be3e-4bad-a0b2-dfe3a16fbb9d:8edebabc3800e0d0a6b46909f18045c8

# RunPod (for cloud GPU)
RUNPOD_API_KEY=rpa_6YIMADVCWS5WR4HK02J4355MCA30CKVKK92JPDN91fudrf

# Application paths
APP_PATH=/workspace/ai-gen
ENVEOF
    echo "✅ Created root .env with API keys"
fi

# Create/activate virtual environment
if [ -d ".venv" ]; then
    echo "✅ Found existing virtual environment"
    source .venv/bin/activate
else
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install base requirements
echo "📦 Installing dependencies..."
if [ -f "base_requirements.txt" ]; then
    pip install -r base_requirements.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  No requirements file found, skipping..."
fi

# Run install script to download models
echo ""
echo "📥 Downloading models (this may take a while on first run)..."
if [ -f "install.py" ]; then
    python install.py || echo "⚠️  Install script had issues, continuing anyway..."
else
    echo "⚠️  install.py not found, skipping model download"
fi

# Setup GUI backend environment
echo ""
echo "🎨 Setting up GUI..."
cd charforge-gui/backend

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating backend .env file..."

    # Create .env with API keys
    cat > .env << 'ENVEOF'
# CharForge Environment Variables
SECRET_KEY=change-this-in-production-$(openssl rand -hex 32)
DATABASE_URL=sqlite:///./database.db

# API Keys
HF_TOKEN=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
HF_HOME=/workspace/.cache/huggingface
CIVITAI_API_KEY=68b35c5249f706b2fdf33a96314628ff
GOOGLE_API_KEY=AIzaSyCkIlt1nCc5HDfKjrGvUHknmBj5PqdhTU8
FAL_KEY=93813d30-be3e-4bad-a0b2-dfe3a16fbb9d:8edebabc3800e0d0a6b46909f18045c8
RUNPOD_API_KEY=rpa_6YIMADVCWS5WR4HK02J4355MCA30CKVKK92JPDN91fudrf

# Application paths
APP_PATH=/workspace/ai-gen
ENVEOF
    echo "✅ Created backend .env with API keys"
fi

# Install GUI dependencies
echo "📦 Installing GUI dependencies..."
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt -q

cd ..

# Setup frontend
echo "🎨 Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies (this may take a few minutes)..."
    npm install
else
    echo "✅ Node.js dependencies already installed"
fi

cd ../..

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "🚀 Starting CharForge GUI..."
echo ""

# Detect RunPod environment
RUNPOD_POD_ID=${RUNPOD_POD_ID:-}
RUNPOD_PUBLIC_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "")

# Launch GUI
cd charforge-gui

# Make start script executable
chmod +x start-dev.sh

# Start the GUI in background
echo "🚀 Starting backend and frontend..."
bash start-dev.sh &
GUI_PID=$!

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

echo ""
echo "================================"
echo "🎉 CharForge is Running!"
echo "================================"
echo ""

if [ -n "$RUNPOD_POD_ID" ]; then
    # Try to construct RunPod proxy URLs
    POD_HOST=$(hostname)
    echo "📱 RunPod Access URLs:"
    echo ""
    echo "   🌐 Frontend (Main GUI):"
    echo "      https://${POD_HOST}-5173.proxy.runpod.net"
    echo ""
    echo "   🔧 Backend (API Docs):"
    echo "      https://${POD_HOST}-8000.proxy.runpod.net/docs"
    echo ""
    echo "   💡 If the above URLs don't work:"
    echo "      1. Go to your RunPod dashboard"
    echo "      2. Click the 'Connect' button on your pod"
    echo "      3. Look for 'HTTP Service [Port 5173]' or 'HTTP Service [Port 8000]'"
    echo "      4. Click on those links to access the GUI"
    echo ""
    echo "   📖 See RUNPOD_ACCESS.md for detailed access instructions"
else
    echo "📱 Local Access:"
    echo "   Frontend: http://localhost:5173"
    echo "   Backend:  http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    if [ -n "$RUNPOD_PUBLIC_IP" ]; then
        echo ""
        echo "   Public:   http://${RUNPOD_PUBLIC_IP}:5173"
    fi
fi

echo ""
echo "🛑 To stop: Press Ctrl+C"
echo ""

# Keep script running
wait $GUI_PID
