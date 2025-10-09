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

# Get IP address for remote access
RUNPOD_POD_ID=$(hostname | cut -d'-' -f1)
RUNPOD_PUBLIC_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "unknown")

# Launch GUI
cd charforge-gui

# Make start script executable
chmod +x start-dev.sh

# Start the GUI
bash start-dev.sh &

# Wait for services to start
sleep 5

echo ""
echo "================================"
echo "🎉 CharForge is Running!"
echo "================================"
echo ""
echo "📱 Access URLs:"
echo "   Local:  http://localhost:5173"
echo "   Public: http://${RUNPOD_PUBLIC_IP}:5173"
echo ""
echo "🔧 API Documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "💡 RunPod Connection:"
echo "   Use RunPod's 'Connect' button to access via HTTP"
echo "   Or use port forwarding for direct access"
echo ""
echo "🛑 To stop: Press Ctrl+C or close this terminal"
echo ""

# Keep script running
wait
