# Migration Guide: Old CharForge → CharForgeX

## Overview

This guide helps you transition from the original CharForge implementation to the new CharForgeX system.

## Key Differences

### Philosophy
- **Old**: Safety-first with hardcoded filters
- **New**: User freedom with optional safety

### Architecture
- **Old**: Monolithic scripts
- **New**: Modular, extensible architecture

### Interfaces
- **Old**: Basic CLI scripts
- **New**: Full CLI + Gradio GUI

### Features
- **Old**: Flux training + basic inference
- **New**: Multi-model support + video generation

## File Mapping

### Configuration

| Old | New | Notes |
|-----|-----|-------|
| `.env` | `.env` + `config.yaml` | Split into env vars and config |
| Hardcoded paths | `config.yaml` | All paths configurable |
| `scripts/character_lora.yaml` | `config.yaml` training section | Unified config |

### Training

| Old | New | Notes |
|-----|-----|-------|
| `train_character.py` | `cli/main.py train lora` | CLI interface |
| N/A | `training/flux_trainer.py` | Module structure |
| `scripts/run_ai_toolkit.sh` | Integrated into Python | No shell scripts needed |

### Inference

| Old | New | Notes |
|-----|-----|-------|
| `test_character.py` | `cli/main.py generate image` | CLI interface |
| `inference/safety.py` | Removed (optional) | No forced safety checks |
| `test_character.py` | `generation/image/flux_gen.py` | Module structure |

### Dataset Processing

| Old | New | Notes |
|-----|-----|-------|
| `scripts/run_captioner.sh` | `cli/main.py dataset caption` | Python implementation |
| N/A | `dataset/cleaning.py` | New feature |
| N/A | `dataset/preprocessing.py` | New feature |

### GUI

| Old | New | Notes |
|-----|-----|-------|
| `charforge-gui/` (Vue.js) | `gui/app.py` (Gradio) | Simplified, fully functional |
| Complex backend API | Direct Python integration | No API server needed |

## Migration Steps

### 1. Backup Old Installation

```bash
# Create backup
cp -r CharForge CharForge_backup
```

### 2. Set Up CharForgeX

```bash
# Install new system
cd CharForgeX
bash install.sh
```

### 3. Migrate Configuration

```bash
# Copy your old .env
cp ../CharForge/.env .env

# Edit config.yaml with your settings
cp config.yaml.example config.yaml
nano config.yaml
```

### 4. Migrate Datasets

```bash
# Old location: CharForge/scratch/{character}/sheet/
# New location: CharForgeX/workspace/datasets/{character}/

# Copy datasets
mkdir -p workspace/datasets
cp -r ../CharForge/scratch/*/sheet workspace/datasets/
```

### 5. Migrate LoRAs

```bash
# Old location: CharForge/scratch/{character}/
# New location: CharForgeX/output/loras/{character}/

# Copy LoRAs
mkdir -p output/loras
find ../CharForge/scratch -name "*.safetensors" -exec cp {} output/loras/ \;
```

### 6. Migrate Generated Content

```bash
# Old location: CharForge/scratch/{character}/output/
# New location: CharForgeX/output/images/

# Copy outputs
mkdir -p output/images
cp -r ../CharForge/scratch/*/output/* output/images/
```

## Command Equivalents

### Training

**Old:**
```bash
python train_character.py \
  --name "character" \
  --input "image.png" \
  --steps 1000 \
  --batch_size 1 \
  --lr 8e-4
```

**New:**
```bash
# First process dataset
python -m cli.main dataset preprocess \
  ./raw_images \
  ./processed_images

python -m cli.main dataset caption \
  ./processed_images

# Then train
python -m cli.main train lora \
  ./processed_images \
  character \
  --model flux \
  --steps 1000 \
  --batch-size 1 \
  --lr 0.0008
```

### Image Generation

**Old:**
```bash
python test_character.py \
  --character_name "character" \
  --prompt "portrait of a person" \
  --lora_weight 0.73 \
  --batch_size 4 \
  --safety_check
```

**New:**
```bash
python -m cli.main generate image \
  "portrait of a person" \
  --lora ./output/loras/character/character.safetensors \
  --lora-strength 0.73 \
  --num-images 4
  # No --safety-check flag needed, disabled by default
```

## Feature Replacements

### Safety Checking

**Old:** Always enabled by default
```python
# test_character.py
parser.add_argument("--safety_check", action="store_true", default=True)
```

**New:** Disabled by default, optional
```python
# config.yaml
enable_safety_checker: false  # Default is false
```

To enable safety in new system:
```yaml
# config.yaml
enable_safety_checker: true
```

### Prompt Optimization

**Old:**
```bash
python test_character.py \
  --prompt "simple prompt" \
  --do_optimize_prompt
```

**New:**
```bash
# Built into generation pipeline, configurable
python -m cli.main generate image \
  "simple prompt" \
  --optimize-prompt  # If implemented
```

### Face Enhancement

**Old:**
```bash
python test_character.py \
  --face_enhance
```

**New:**
```yaml
# config.yaml
image_generation:
  face_restoration: true
```

## Configuration Migration

### Environment Variables

**Keep these the same:**
```bash
HF_TOKEN=...
HF_HOME=...
CIVITAI_API_KEY=...
GOOGLE_API_KEY=...
FAL_KEY=...
```

### New Config File

Create `config.yaml` from your old settings:

```yaml
# Equivalent to old train_character.py defaults
training:
  steps: 800  # Old: --steps
  batch_size: 1  # Old: --batch_size
  learning_rate: 0.0008  # Old: --lr 8e-4
  lora_rank: 8  # Old: --rank_dim
  resolution: 512  # Old: --train_dim

# Equivalent to old test_character.py defaults
image_generation:
  width: 1024  # Old: --test_dim
  height: 1024
  num_inference_steps: 30
  lora_weight: 0.73  # Old: --lora_weight

# New feature: disable safety (old had it enabled)
enable_safety_checker: false
```

## Workflow Migration

### Old Workflow

```bash
# 1. Train
python train_character.py --name char --input img.png

# 2. Generate
python test_character.py --character_name char --prompt "portrait"
```

### New Workflow

```bash
# 1. Process dataset
python -m cli.main dataset clean ./raw ./cleaned
python -m cli.main dataset preprocess ./cleaned ./processed
python -m cli.main dataset caption ./processed

# 2. Train
python -m cli.main train lora ./processed char --model flux

# 3. Generate
python -m cli.main generate image "portrait" --lora ./output/loras/char/char.safetensors
```

Or use the GUI:

```bash
python gui/app.py
# All steps in web interface
```

## ComfyUI Integration

### Old
```bash
python scripts/run_comfy.py
bash scripts/symlink_loras.sh
```

### New
ComfyUI is optional. The new system is standalone but you can still integrate:

```bash
# Symlink your LoRAs to ComfyUI if needed
ln -s $(pwd)/output/loras /path/to/ComfyUI/models/loras/charforgex
```

## GUI Migration

### Old GUI (Vue.js + FastAPI)

```bash
cd charforge-gui
bash start-dev.sh
# Complex setup with frontend build, backend server, database
```

### New GUI (Gradio)

```bash
python gui/app.py
# Single command, all-in-one interface
```

**Benefits of new GUI:**
- No build step required
- No separate backend server
- No database setup needed
- Instant startup
- All features integrated

## API Migration

### Old System
Had a REST API server in `charforge-gui/backend/`

### New System
Direct Python module imports - no API needed for local use.

If you need an API, add in Phase 8 (see ROADMAP.md).

For now, use:
```python
from generation.image.flux_gen import FluxGenerator

gen = FluxGenerator()
gen.load()
images = gen.generate("prompt")
```

## Troubleshooting Migration

### "Can't find LoRA"

**Problem:** Old LoRA paths don't work

**Solution:**
```bash
# Find your LoRAs
find ../CharForge -name "*.safetensors"

# Copy to new location
cp path/to/old.safetensors ./output/loras/
```

### "Captions missing"

**Problem:** Old dataset doesn't have captions

**Solution:**
```bash
# Recaption with new system
python -m cli.main dataset caption ./workspace/datasets/old_dataset
```

### "Different image quality"

**Problem:** New images look different

**Solution:**
```yaml
# Adjust config.yaml to match old defaults
image_generation:
  num_inference_steps: 30  # Match old steps
  guidance_scale: 7.5
  lora_weight: 0.73  # Match old lora_weight
```

### "Safety checker blocks content"

**Problem:** Safety checker enabled when you don't want it

**Solution:**
```yaml
# config.yaml
enable_safety_checker: false  # Default in new system
```

## Benefits of Migration

### What You Gain

1. **No Forced Restrictions**
   - Safety disabled by default
   - Full creative control

2. **Better Organization**
   - Clear module structure
   - Unified configuration
   - Easier to extend

3. **More Features**
   - Video generation
   - Dataset cleaning
   - Smart preprocessing
   - Multiple captioning backends

4. **Easier Usage**
   - Gradio GUI
   - Unified CLI
   - Better docs

5. **Future-Proof**
   - Modular architecture
   - Cloud-ready
   - Easy to extend

### What You Keep

1. **Your Data**
   - All datasets can be migrated
   - All LoRAs work with new system
   - All images/outputs preserved

2. **Your Workflow**
   - Similar commands available
   - Same underlying models
   - Compatible file formats

3. **Your API Keys**
   - Same .env file
   - Same HuggingFace access
   - Same external services

## Rollback Plan

If you need to go back to old system:

```bash
# Restore backup
mv CharForge CharForgeX_new
mv CharForge_backup CharForge
cd CharForge

# Reactivate old environment
source .venv/bin/activate
```

Your old system is unchanged in the backup.

## Getting Help

1. Check [QUICKSTART.md](QUICKSTART.md) for new workflow
2. Read [README_NEW.md](README_NEW.md) for full docs
3. See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for overview
4. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

## Recommended Approach

1. **Week 1**: Install new system, test with one character
2. **Week 2**: Migrate all datasets and LoRAs
3. **Week 3**: Switch to new system full-time
4. **Week 4**: Archive old system

Keep old system as backup for first month.

## Quick Migration Checklist

- [ ] Back up old installation
- [ ] Install CharForgeX
- [ ] Copy .env file
- [ ] Create config.yaml
- [ ] Migrate datasets
- [ ] Migrate LoRAs
- [ ] Test image generation
- [ ] Test video generation
- [ ] Verify quality matches old system
- [ ] Switch to new workflows
- [ ] Archive old installation

---

**Welcome to CharForgeX! Enjoy your unrestricted creative freedom.** 🎨🚀
