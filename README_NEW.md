# CharForgeX - Professional LoRA Training & Ultra-Realistic Media Generation

> **No content restrictions. Maximum creative freedom. Production-ready system for training photorealistic LoRAs and generating high-fidelity images/videos.**

## Features

### 🎯 Core Capabilities

- **Multi-Model Support**: Flux, SDXL, SD3, AnimateDiff, HotShotXL
- **Advanced Dataset Processing**: Cleaning, face detection, quality filtering, smart cropping
- **Multi-Backend Captioning**: BLIP2, CLIP-Interrogator, CogVLM, ensemble mode
- **LoRA Training**: Flux/SDXL/SD3 with automatic hyperparameter optimization
- **LoRA Utilities**: Merging, stacking, benchmarking, weight interpolation
- **High-Fidelity Image Generation**: ControlNet, IPAdapter, regional prompting
- **Video Generation**: AnimateDiff/HotShotXL with facial motion and audio sync
- **Temporal Interpolation**: RIFE/AMT for smooth video output
- **Cloud Integration**: RunPod for scalable GPU provisioning
- **Dual Interface**: Gradio GUI + comprehensive CLI

### ⚠️ No Restrictions

- **No Safety Filters**: Zero hardcoded content restrictions
- **No Telemetry**: Your data stays local
- **No Watermarks**: Clean outputs
- **Full Control**: Complete access to all parameters

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CharForgeX
cd CharForgeX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (HF_TOKEN, etc.)
```

### Basic Usage

#### 1. Process Dataset

```bash
# Clean dataset
python -m cli.main dataset clean ./raw_images ./cleaned_images --require-faces

# Preprocess (resize + smart crop)
python -m cli.main dataset preprocess ./cleaned_images ./processed_images --resolution 1024

# Caption images
python -m cli.main dataset caption ./processed_images --backend blip2
```

#### 2. Train LoRA

```bash
python -m cli.main train lora ./processed_images my_character \
  --model flux \
  --steps 1000 \
  --batch-size 1 \
  --lr 1e-4 \
  --rank 16 \
  --resolution 512
```

#### 3. Generate Images

```bash
python -m cli.main generate image "portrait of a person, photorealistic, high quality" \
  --lora ./output/loras/my_character/my_character.safetensors \
  --num-images 4 \
  --resolution 1024 \
  --steps 30 \
  --lora-strength 0.8
```

#### 4. Generate Videos

```bash
python -m cli.main generate video "person walking in a park, cinematic" \
  --lora ./output/loras/my_character/my_character.safetensors \
  --num-frames 16 \
  --fps 8 \
  --resolution 512
```

### GUI Interface

```bash
python gui/app.py
```

Access at `http://localhost:7860`

## Advanced Features

### LoRA Merging

```bash
python -m cli.main lora merge \
  ./loras/character1.safetensors \
  ./loras/character2.safetensors \
  ./loras/merged.safetensors \
  --weights 0.6,0.4
```

### Cloud Mode (RunPod)

```yaml
# config.yaml
compute:
  mode: cloud

cloud:
  enabled: true
  provider: runpod
  api_key: your_runpod_api_key
  gpu_type: RTX4090
  region: US-OR
  max_cost_per_hour: 0.50
```

```bash
# Provision cloud instance
python -m cli.main cloud provision --gpu-type RTX4090

# Run training on cloud
python -m cli.main train lora ./dataset my_lora --model flux --steps 1000
```

### Video with Facial Motion

```python
from generation.video.animatediff import AnimateDiffGenerator

generator = AnimateDiffGenerator()
generator.load()
generator.load_lora("./loras/my_character.safetensors")

frames = generator.generate(
    prompt="person talking",
    num_frames=32,
    width=512,
    height=512,
)

# Save video
generator.save_video(frames, "./output/video.mp4", fps=8)
```

### ControlNet Image Generation

```python
from generation.image.flux_gen import FluxGenerator

generator = FluxGenerator()
generator.load()
generator.load_lora("./loras/my_character.safetensors")

images = generator.generate(
    prompt="portrait of a person, photorealistic",
    width=1024,
    height=1024,
    num_images=4,
    lora_scale=0.8,
)

generator.save_images(images, "./output/images")
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

### Module Structure

```
CharForgeX/
├── core/              # Configuration system
├── dataset/           # Dataset processing
│   ├── cleaning.py
│   ├── preprocessing.py
│   └── captioning/
├── training/          # LoRA training engines
├── generation/        # Media generation
│   ├── image/
│   └── video/
├── cloud/             # RunPod integration
├── cli/               # CLI interface
├── gui/               # Gradio GUI
└── utils/             # Utilities
```

## Configuration

All settings can be configured via `config.yaml`:

```yaml
# Example configuration
project_name: my_project
work_dir: ./workspace
cache_dir: ./cache

# No restrictions
enable_safety_checker: false
enable_watermark: false
enable_telemetry: false

compute:
  mode: local
  device: cuda
  precision: bf16
  use_xformers: true

dataset:
  enable_face_detection: true
  enable_deduplication: true
  enable_quality_filter: true
  min_resolution: 512
  target_resolution: 1024
  captioning_backend: blip2

training:
  model_backend: flux-dev
  lora_rank: 16
  steps: 1000
  batch_size: 1
  learning_rate: 1e-4
  resolution: 512

image_generation:
  model_backend: flux-dev
  width: 1024
  height: 1024
  num_inference_steps: 30
  lora_weight: 0.8

video_generation:
  backend: animatediff
  width: 512
  height: 512
  num_frames: 16
  fps: 8
  use_interpolation: true
  target_fps: 24
```

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 12GB+ VRAM (24GB+ recommended for video)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for models and datasets

### Software

- Python 3.10+
- CUDA 12.0+
- PyTorch 2.1+

See [requirements.txt](requirements.txt) for complete dependencies.

## CLI Reference

### Dataset Commands

```bash
# Clean dataset
cli dataset clean INPUT_DIR OUTPUT_DIR [OPTIONS]

# Preprocess dataset
cli dataset preprocess INPUT_DIR OUTPUT_DIR [OPTIONS]

# Caption dataset
cli dataset caption INPUT_DIR [OPTIONS]
```

### Training Commands

```bash
# Train LoRA
cli train lora DATASET_DIR OUTPUT_NAME [OPTIONS]
```

### Generation Commands

```bash
# Generate images
cli generate image PROMPT [OPTIONS]

# Generate videos
cli generate video PROMPT [OPTIONS]
```

### LoRA Commands

```bash
# Merge LoRAs
cli lora merge LORA1 LORA2 OUTPUT [OPTIONS]

# Benchmark LoRA
cli lora benchmark LORA_PATH [OPTIONS]
```

### Cloud Commands

```bash
# Provision instance
cli cloud provision [OPTIONS]

# Check status
cli cloud status

# Terminate instance
cli cloud terminate
```

## Performance

### Benchmarks (Single RTX 4090)

| Task | Time | Throughput |
|------|------|------------|
| Dataset Captioning (1000 images) | ~30min | 33 img/min |
| Flux LoRA Training (1000 steps) | ~40min | 25 steps/min |
| Image Generation (1024x1024) | ~8s | 4 images/batch |
| Video Generation (16 frames) | ~45s | 1.3 videos/min |

### Optimization Tips

1. **Enable xformers**: 30% faster training/inference
2. **Use 8-bit optimizer**: 40% less VRAM
3. **Enable gradient checkpointing**: 50% less VRAM
4. **Cache latents**: 3x faster training after first epoch
5. **Use bf16 precision**: Best quality/speed tradeoff

## Cloud Deployment

### RunPod Setup

```bash
# Set API key
export RUNPOD_API_KEY=your_key

# Provision GPU
python -m cli.main cloud provision \
  --gpu-type RTX4090 \
  --region US-OR

# Sync dataset
python -m cloud.runpod.sync upload ./dataset

# Train on cloud
python -m cli.main train lora /workspace/dataset my_lora

# Download LoRA
python -m cloud.runpod.sync download ./output/loras
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Lower resolution
- Use 8-bit optimizer

### Slow Training

- Enable xformers
- Use bf16 precision
- Cache latents
- Increase batch size (if VRAM allows)

### Poor LoRA Quality

- Increase training steps
- Improve dataset quality
- Try different learning rates
- Use higher LoRA rank

## Examples

See [examples/](examples/) for:

- Sample datasets
- Training configurations
- Generation prompts
- LoRA merging recipes
- Video generation workflows

## Contributing

This is a personal project designed for maximum creative freedom. Feel free to fork and modify for your needs.

## License

MIT License - See [LICENSE](LICENSE)

## Disclaimer

This tool is designed for creative and research purposes. Users are responsible for compliance with local laws and platform terms of service. No content filtering is applied by default - use responsibly.

## Acknowledgments

- Flux: Black Forest Labs
- AnimateDiff: GuoYuWei et al.
- Diffusers: Hugging Face
- LoRA Training: ostris/ai-toolkit

---

**Built for creators who demand full control over their AI media generation pipeline.**
