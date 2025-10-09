# ai-gen - Professional LoRA Training & Ultra-Realistic Media Generation

## System Architecture

### Core Principles
- **No Content Restrictions**: Zero hardcoded filters or safety checks
- **Maximum Realism**: State-of-the-art models for photorealistic output
- **Flexibility**: Easy switching between local and cloud compute
- **Modularity**: Each component works independently and composably
- **Production-Ready**: CLI for automation, GUI for interactive use

### Module Overview

```
ai-gen/
├── core/                       # Core system components
│   ├── config.py              # Unified configuration system
│   ├── models.py              # Model registry and management
│   └── compute.py             # Local/cloud compute abstraction
│
├── dataset/                    # Dataset processing pipeline
│   ├── cleaning.py            # Face detection, deduplication, quality filtering
│   ├── preprocessing.py       # Resizing, cropping, augmentation
│   └── captioning/            # Multi-backend captioning
│       ├── blip2.py           # BLIP2 captioner
│       ├── clip_interrogator.py
│       ├── cogvlm.py          # CogVLM for detailed captions
│       └── ensemble.py        # Multi-model caption fusion
│
├── training/                   # LoRA training engines
│   ├── flux_trainer.py        # Flux LoRA training
│   ├── sdxl_trainer.py        # SDXL LoRA training
│   ├── sd3_trainer.py         # SD3 LoRA training
│   ├── merging.py             # LoRA merging and stacking
│   ├── benchmarking.py        # Automated quality benchmarking
│   └── hyperopt.py            # Hyperparameter optimization
│
├── generation/                 # Media generation pipelines
│   ├── image/                 # Image generation
│   │   ├── flux_gen.py        # Flux image generation
│   │   ├── sdxl_gen.py        # SDXL image generation
│   │   ├── controlnet.py      # ControlNet integration
│   │   ├── ipadapter.py       # IPAdapter for style/face control
│   │   └── regional.py        # Regional prompting
│   │
│   └── video/                 # Video generation
│       ├── animatediff.py     # AnimateDiff pipeline
│       ├── hotshotxl.py       # HotShotXL pipeline
│       ├── motion/            # Motion control
│       │   ├── face_motion.py # Facial animation
│       │   └── audio_sync.py  # Audio-driven motion
│       └── interpolation.py   # Frame interpolation (RIFE/AMT)
│
├── cloud/                      # Cloud compute integration
│   ├── runpod/                # RunPod integration
│   │   ├── client.py          # RunPod API client
│   │   ├── provisioning.py    # GPU provisioning
│   │   └── sync.py            # Model/dataset syncing
│   └── orchestrator.py        # Local/cloud job orchestration
│
├── cli/                        # Command-line interface
│   ├── dataset.py             # Dataset management commands
│   ├── train.py               # Training commands
│   ├── generate.py            # Generation commands
│   └── lora.py                # LoRA utilities (merge, benchmark)
│
├── gui/                        # Gradio GUI
│   ├── app.py                 # Main Gradio application
│   ├── tabs/                  # Individual feature tabs
│   │   ├── dataset_tab.py
│   │   ├── training_tab.py
│   │   ├── image_gen_tab.py
│   │   └── video_gen_tab.py
│   └── components/            # Reusable UI components
│
└── utils/                      # Shared utilities
    ├── image.py               # Image processing utilities
    ├── video.py               # Video processing utilities
    └── logging.py             # Logging and monitoring
```

## Key Features

### 1. Dataset Processing
- **Cleaning**: Face detection, duplicate removal, blur/quality filtering
- **Preprocessing**: Smart cropping, resizing, aspect ratio handling
- **Captioning**: Multiple backends (BLIP2, CLIP-Interrogator, CogVLM)
- **Augmentation**: Optional augmentations for training diversity

### 2. LoRA Training
- **Multi-Backend**: SDXL, Flux, SD3 support
- **Advanced Features**: LoRA merging, stacking, weight interpolation
- **Benchmarking**: Automated quality evaluation with FID/CLIP scores
- **Hyperparameter Tuning**: Automated optimization

### 3. Image Generation
- **High-Fidelity**: ControlNet, IPAdapter, regional prompting
- **Multi-Model**: Support for SDXL, Flux, SD3
- **Advanced Control**: Face swapping, pose control, style transfer
- **Batch Processing**: Efficient parallel generation

### 4. Video Generation
- **Prompt-to-Video**: AnimateDiff, HotShotXL
- **LoRA-to-Video**: Apply trained LoRAs to video generation
- **Motion Control**: Facial animation, audio-driven motion
- **Temporal Consistency**: Frame interpolation and smoothing

### 5. Cloud Integration
- **RunPod Support**: Seamless GPU provisioning
- **Auto-Scaling**: Dynamic resource allocation
- **Model Sync**: Automatic model/dataset synchronization
- **Cost Optimization**: Intelligent spot instance usage

### 6. User Interfaces
- **CLI**: Full-featured command-line for automation
- **Gradio GUI**: Interactive web interface
- **API**: REST API for custom integrations

## Configuration System

### Local Mode
```yaml
mode: local
compute:
  device: cuda
  precision: bf16
  optimization: xformers
```

### Cloud Mode
```yaml
mode: cloud
compute:
  provider: runpod
  gpu_type: RTX4090
  region: US-OR
  max_cost_per_hour: 0.50
```

## No Content Restrictions

This system is designed for creative freedom:
- No safety checkers or content filters
- No hardcoded model restrictions
- Full control over generation parameters
- No telemetry or usage tracking

## Performance Targets

- **Dataset Processing**: 1000 images/minute (captioning)
- **LoRA Training**: 30-60 minutes (1000 steps, Flux, single GPU)
- **Image Generation**: 4-8 images/minute (1024x1024, Flux)
- **Video Generation**: 1-2 videos/minute (24 frames, AnimateDiff)
