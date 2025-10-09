#!/bin/bash
# ai-gen Installation Script

set -e

echo "================================"
echo "ai-gen Installation"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"
echo ""

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠ CUDA not detected. CPU-only mode will be used."
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch
echo "Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # Install CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # Install CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo "✓ PyTorch installed"
echo ""

# Install main requirements
echo "Installing ai-gen dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p workspace cache output/{images,videos,loras}
echo "✓ Directories created"
echo ""

# Copy example configs
echo "Setting up configuration files..."
if [ ! -f "config.yaml" ]; then
    cp config.yaml.example config.yaml
    echo "✓ Created config.yaml"
else
    echo "✓ config.yaml already exists"
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env"
    echo ""
    echo "⚠ IMPORTANT: Edit .env and add your API keys!"
else
    echo "✓ .env already exists"
fi
echo ""

# Test installation
echo "Testing installation..."
python -c "
import torch
import diffusers
import transformers
print('✓ Core libraries imported successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  Diffusers: {diffusers.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""

echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys (HF_TOKEN, etc.)"
echo "2. Edit config.yaml to customize settings"
echo "3. Activate environment: source venv/bin/activate"
echo "4. Run GUI: python gui/app.py"
echo "5. Or use CLI: python -m cli.main --help"
echo ""
echo "See README_NEW.md for detailed documentation"
echo ""
