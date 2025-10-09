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

# Determine if we need to clone or just pull
if [ -d "CharForgex" ]; then
    echo "📁 Found existing CharForgex directory"
    cd CharForgex
    echo "📥 Pulling latest changes..."
    git pull || echo "⚠️  Could not pull (no git repo or no changes)"
else
    echo "📥 Cloning CharForgeX..."
    # If you have a git repo, uncomment and update:
    # git clone https://github.com/yourusername/CharForgeX.git
    # cd CharForgeX

    # For now, assume code is already present
    if [ -d "CharForgex" ]; then
        cd CharForgex
    else
        echo "❌ CharForgeX directory not found. Please upload your code first."
        exit 1
    fi
fi

echo ""
echo "🔧 Setting up environment..."

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

    # Copy from root .env if it exists
    if [ -f "../../.env" ]; then
        cp ../../.env .env
        echo "✅ Copied API keys from root .env"
    else
        # Create basic .env
        cat > .env << 'ENVEOF'
SECRET_KEY=change-this-in-production-$(openssl rand -hex 32)
DATABASE_URL=sqlite:///./database.db
HF_TOKEN=${HF_TOKEN}
HF_HOME=/workspace/.cache/huggingface
CIVITAI_API_KEY=${CIVITAI_API_KEY}
GOOGLE_API_KEY=${GOOGLE_API_KEY}
FAL_KEY=${FAL_KEY}
RUNPOD_API_KEY=${RUNPOD_API_KEY}
ENVEOF
        echo "⚠️  Created basic .env - please add your API keys!"
    fi
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
