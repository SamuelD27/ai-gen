# CharForge - AI-Powered Character LoRA Creation

> **✨ RECENTLY UPDATED:** API keys configured, safety filters disabled by default, one-command RunPod setup!
> **📖 See [QUICK_START.md](QUICK_START.md) for the new quick start guide**
> **📋 See [CHANGES.md](CHANGES.md) for full list of changes**

<div align="center">

**No content restrictions. Maximum creative freedom. Production-ready.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Train photorealistic LoRAs and generate ultra-realistic images/videos with state-of-the-art AI models*

[Quick Start](#quick-start) •
[Features](#features) •
[Installation](#installation) •
[Documentation](#documentation) •
[Examples](#examples)

</div>

---

## ⚠️ Important Notice

**CharForgeX has NO CONTENT RESTRICTIONS by default.**

- No safety filters
- No content moderation
- No telemetry or tracking
- No watermarks

**You have complete creative freedom and full control.**

*Users are responsible for compliance with local laws and ethical use.*

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# Clone and enter directory
cd CharForgeX

# Run automated installer
bash install.sh

# Edit .env with your API keys
nano .env  # Add HF_TOKEN and other keys

# Activate environment
source venv/bin/activate
```

### Your First LoRA (30-60 minutes)

```bash
# 1. Prepare dataset (10-50 images of same person)
mkdir -p datasets/my_person
# Add your images to datasets/my_person/

# 2. Process dataset
python -m cli.main dataset clean datasets/my_person datasets/my_person_cleaned
python -m cli.main dataset preprocess datasets/my_person_cleaned datasets/my_person_processed
python -m cli.main dataset caption datasets/my_person_processed

# 3. Train LoRA (30-40 min on RTX 4090)
python -m cli.main train lora datasets/my_person_processed my_person --model flux --steps 1000

# 4. Generate images
python -m cli.main generate image "portrait of a person, professional photography" \
  --lora ./output/loras/my_person/my_person.safetensors \
  --num-images 4

# 5. Generate video
python -m cli.main generate video "person smiling at camera" \
  --lora ./output/loras/my_person/my_person.safetensors \
  --num-frames 16
```

### Or Use the GUI

```bash
python gui/app.py
# Access at http://localhost:7860
```

---

## ✨ Features

### Core Capabilities

- 🎨 **Multi-Model Support**: Flux, SDXL, SD3, AnimateDiff, HotShotXL
- 🧹 **Advanced Dataset Processing**: Face detection, quality filtering, smart cropping
- 📝 **Multi-Backend Captioning**: BLIP2, CLIP-Interrogator, CogVLM, ensemble
- 🏋️ **LoRA Training**: Photorealistic character training with auto-optimization
- 🔧 **LoRA Utilities**: Merging, stacking, benchmarking, weight interpolation
- 🖼️ **High-Fidelity Images**: ControlNet, IPAdapter, regional prompting
- 🎬 **Video Generation**: AnimateDiff/HotShotXL with motion control
- 🎵 **Audio Sync**: Facial motion and audio-driven animation
- 📈 **Frame Interpolation**: RIFE/AMT for smooth 60fps video
- ☁️ **Cloud Integration**: RunPod for scalable GPU provisioning
- 💻 **Dual Interface**: Professional CLI + intuitive Gradio GUI

### No Restrictions

✅ **No safety filters** - Complete creative freedom
✅ **No telemetry** - Your data stays local
✅ **No watermarks** - Clean, professional outputs
✅ **No limits** - Generate anything you can imagine
✅ **Full control** - Access to all parameters and settings

---

## 📦 Installation

### System Requirements

**Hardware:**
- GPU: NVIDIA with 12GB+ VRAM (24GB+ recommended for video)
- RAM: 32GB+ recommended
- Storage: 100GB+ free space

**Software:**
- Python 3.10 or higher
- CUDA 12.0+ (for GPU acceleration)
- Linux, macOS, or Windows WSL2

### Quick Install

```bash
bash install.sh
```

The installer will:
1. Check Python version
2. Detect CUDA/GPU
3. Create virtual environment
4. Install PyTorch + dependencies
5. Set up directories
6. Create config files

### Manual Install

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p workspace cache output/{images,videos,loras}

# Copy configs
cp config.yaml.example config.yaml
cp .env.example .env
```

### Configuration

Edit `.env` with your API keys:

```bash
HF_TOKEN=your_huggingface_token
CIVITAI_API_KEY=your_civitai_key  # Optional
GOOGLE_API_KEY=your_google_key    # Optional for enhanced captioning
RUNPOD_API_KEY=your_runpod_key    # Optional for cloud compute
```

---

## 📚 Documentation

### Core Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 30-minute tutorial for your first LoRA
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and technical overview
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - What's been built and why
- **[ROADMAP.md](ROADMAP.md)** - Future development plans
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrate from original CharForge

### Quick Links

- [Features](#features) - What CharForgeX can do
- [CLI Reference](#cli-reference) - Command-line usage
- [GUI Guide](#gui-interface) - Web interface tutorial
- [Configuration](#configuration-guide) - Settings and options
- [Troubleshooting](#troubleshooting) - Common issues
- [Examples](#examples) - Real-world usage

---

## 🎯 Examples

### High-Quality Portrait

```bash
python -m cli.main generate image \
  "professional portrait, studio lighting, 8k uhd, highly detailed" \
  --lora ./output/loras/person/person.safetensors \
  --resolution 2048 \
  --steps 50 \
  --lora-strength 0.85
```

### Batch Generation

```bash
# Create prompts file
cat > prompts.txt << EOF
portrait in a park, natural lighting
professional headshot, studio lighting
casual photo, urban background
artistic portrait, dramatic lighting
EOF

# Generate for each prompt
while read prompt; do
  python -m cli.main generate image "$prompt" \
    --lora ./output/loras/person/person.safetensors \
    --num-images 4
done < prompts.txt
```

### Video with LoRA

```bash
python -m cli.main generate video \
  "person looking at camera and smiling, professional lighting" \
  --lora ./output/loras/person/person.safetensors \
  --num-frames 32 \
  --fps 8 \
  --resolution 512 \
  --output ./output/videos/smile.mp4
```

### LoRA Merging

```bash
python -m cli.main lora merge \
  ./output/loras/person1/person1.safetensors \
  ./output/loras/person2/person2.safetensors \
  ./output/loras/merged.safetensors \
  --weights 0.6,0.4
```

---

## 💻 CLI Reference

### Dataset Commands

```bash
# Clean dataset (face detection, quality filter, deduplication)
cli dataset clean INPUT_DIR OUTPUT_DIR [OPTIONS]
  --min-resolution INT      # Minimum image resolution (default: 512)
  --require-faces           # Only keep images with faces
  --quality-filter          # Enable quality assessment

# Preprocess dataset (resize, crop, augment)
cli dataset preprocess INPUT_DIR OUTPUT_DIR [OPTIONS]
  --resolution INT          # Target resolution (default: 1024)
  --smart-crop              # Use face-aware cropping
  --augment                 # Create augmented versions

# Caption images
cli dataset caption INPUT_DIR [OPTIONS]
  --backend STR             # blip2, clip_interrogator, cogvlm, ensemble
  --batch-size INT          # Batch size (default: 4)
```

### Training Commands

```bash
# Train LoRA
cli train lora DATASET_DIR OUTPUT_NAME [OPTIONS]
  --model STR               # flux, sdxl, sd3 (default: flux)
  --steps INT               # Training steps (default: 1000)
  --batch-size INT          # Batch size (default: 1)
  --lr FLOAT                # Learning rate (default: 1e-4)
  --rank INT                # LoRA rank (default: 16)
  --resolution INT          # Training resolution (default: 512)
```

### Generation Commands

```bash
# Generate images
cli generate image PROMPT [OPTIONS]
  --lora PATH               # Path to LoRA weights
  --model STR               # flux, sdxl (default: flux)
  --num-images INT          # Number of images (default: 4)
  --resolution INT          # Image resolution (default: 1024)
  --steps INT               # Inference steps (default: 30)
  --lora-strength FLOAT     # LoRA strength (default: 0.8)

# Generate video
cli generate video PROMPT [OPTIONS]
  --lora PATH               # Path to LoRA weights
  --num-frames INT          # Number of frames (default: 16)
  --fps INT                 # Frames per second (default: 8)
  --resolution INT          # Frame resolution (default: 512)
  --output PATH             # Output video path
```

### LoRA Commands

```bash
# Merge LoRAs
cli lora merge LORA1 LORA2 OUTPUT [OPTIONS]
  --weights STR             # Comma-separated weights (e.g., "0.6,0.4")

# Benchmark LoRA
cli lora benchmark LORA_PATH [OPTIONS]
```

### Cloud Commands

```bash
# Provision GPU instance
cli cloud provision [OPTIONS]
  --gpu-type STR            # RTX4090, A100, H100
  --region STR              # US-OR, US-CA, EU-RO

# Check status
cli cloud status

# Terminate instance
cli cloud terminate
```

---

## 🖥️ GUI Interface

### Launch GUI

```bash
python gui/app.py
```

Access at `http://localhost:7860`

### GUI Features

**Dataset Processing Tab:**
- Clean datasets visually
- Configure preprocessing
- Batch captioning

**Training Tab:**
- Configure LoRA training
- Monitor progress
- View training logs

**Image Generation Tab:**
- Generate with prompts
- Apply LoRAs
- Download results

**Video Generation Tab:**
- Prompt-to-video
- LoRA-to-video
- Motion control

**LoRA Utilities Tab:**
- Merge LoRAs
- Benchmark performance
- Manage weights

**Settings Tab:**
- Configure compute mode
- Cloud integration
- System preferences

---

## ⚙️ Configuration Guide

Configuration is managed via `config.yaml`:

```yaml
# Project settings
project_name: my_project
work_dir: ./workspace
cache_dir: ./cache

# NO RESTRICTIONS (default)
enable_safety_checker: false
enable_watermark: false
enable_telemetry: false

# Compute settings
compute:
  mode: local  # local, cloud, hybrid
  device: cuda
  precision: bf16
  use_xformers: true

# Dataset processing
dataset:
  enable_face_detection: true
  target_resolution: 1024
  captioning_backend: blip2

# Training
training:
  model_backend: flux-dev
  lora_rank: 16
  steps: 1000
  learning_rate: 0.0001

# Image generation
image_generation:
  width: 1024
  height: 1024
  num_inference_steps: 30
  lora_weight: 0.8

# Video generation
video_generation:
  backend: animatediff
  num_frames: 16
  fps: 8
  use_interpolation: true
```

See [config.yaml.example](config.yaml.example) for all options.

---

## 🐛 Troubleshooting

### Out of VRAM

**Symptoms:** CUDA out of memory errors

**Solutions:**
- Reduce `batch_size` to 1
- Lower `resolution` (try 512 or 768)
- Enable gradient checkpointing in config
- Use 8-bit optimizer

### Poor LoRA Quality

**Symptoms:** Generated images don't match training data

**Solutions:**
- Increase training steps (1500-2000)
- Use more diverse images (20-50)
- Try higher LoRA rank (24-32)
- Increase lora-strength (0.9-1.0)

### Slow Training

**Symptoms:** Training takes too long

**Solutions:**
- Enable xformers in config
- Use bf16 precision
- Cache latents (enabled by default)
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Blurry Images

**Symptoms:** Generated images lack detail

**Solutions:**
- Increase inference steps (40-50)
- Use higher resolution (1536-2048)
- Adjust guidance scale (6-10)
- Enable face restoration

---

## 📊 Performance

### Benchmarks (RTX 4090)

| Operation | Time | Throughput |
|-----------|------|------------|
| Dataset Captioning (1000 images) | ~30min | 33 img/min |
| Flux LoRA Training (1000 steps) | ~40min | 25 steps/min |
| Image Generation (1024x1024) | ~8s | 4 images/batch |
| Video Generation (16 frames) | ~45s | 1.3 videos/min |

### Optimization Tips

1. **Enable xformers** - 30% faster training/inference
2. **Use 8-bit optimizer** - 40% less VRAM
3. **Enable gradient checkpointing** - 50% less VRAM
4. **Cache latents** - 3x faster after first epoch
5. **Use bf16 precision** - Best quality/speed tradeoff

---

## 🤝 Contributing

This is a personal project focused on creative freedom. Feel free to fork and modify for your needs.

To request features:
1. Check [ROADMAP.md](ROADMAP.md) for planned features
2. Open an issue with [Feature Request] tag
3. Describe use case and benefits

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## ⚖️ Disclaimer

**This tool is designed for creative and research purposes.**

- Users are responsible for compliance with local laws
- No content filtering is applied by default
- Use responsibly and ethically
- Respect intellectual property and privacy rights

---

## 🙏 Acknowledgments

- **Flux**: Black Forest Labs
- **AnimateDiff**: GuoYuWei et al.
- **Diffusers**: Hugging Face
- **LoRA**: Microsoft Research
- **BLIP2**: Salesforce Research

---

## 📞 Support

- **Documentation**: See [docs](#documentation)
- **Examples**: Check [examples/](examples/)
- **Issues**: Open GitHub issue
- **Quick Help**: See [QUICKSTART.md](QUICKSTART.md)

---

<div align="center">

**Built for creators who demand full control over their AI media generation pipeline.**

🎨 Train • 🖼️ Generate • 🎬 Animate • 🚀 Create

*No restrictions. Maximum freedom.*

</div>
