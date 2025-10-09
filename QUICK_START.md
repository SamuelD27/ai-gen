# CharForge Quick Start Guide

## 🎯 What's New

Your CharForge system has been enhanced with:

✅ **API Keys Pre-configured** - All your API keys are set up
✅ **No Safety Restrictions** - Safety checks disabled by default (opt-in if needed)
✅ **One-Command RunPod Setup** - Launch everything with a single script
✅ **Upload Already Works** - GUI has full image upload capability
✅ **Optimized Configuration** - Best settings for performance

---

## 🚀 Quick Start (Local)

### 1. First Time Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not done)
pip install -r base_requirements.txt

# Download models (first time only)
python install.py
```

### 2. Launch GUI

```bash
cd charforge-gui
bash start-dev.sh
```

Access at: **http://localhost:5173**

### 3. Use CharForge

1. **Upload Images**: Go to Media tab, drag & drop your images
2. **Create Dataset**: Select uploaded images, create dataset with trigger word
3. **Train LoRA**: Go to Characters, create character, start training
4. **Generate Images**: Go to Inference, select character, enter prompt
5. **Generate Videos**: (Coming soon with AnimateDiff integration)

---

## ☁️ RunPod Quick Start

### One-Command Launch

```bash
# On RunPod terminal, run:
bash runpod_quickstart.sh
```

This will:
1. ✅ Setup virtual environment
2. ✅ Install all dependencies
3. ✅ Download required models
4. ✅ Configure API keys
5. ✅ Launch GUI automatically

### Accessing on RunPod

After launch, you'll see:
```
🎉 CharForge is Running!
📱 Access URLs:
   Local:  http://localhost:5173
   Public: http://<your-pod-ip>:5173
```

**To access from your computer:**
1. Click "Connect" button in RunPod dashboard
2. Select "HTTP Service"
3. Use port 5173

---

## 🔑 API Keys

All your API keys are pre-configured in `.env`:

- ✅ **HuggingFace**: For Flux models
- ✅ **CivitAI**: For community models
- ✅ **Google AI**: For advanced captioning
- ✅ **fal.ai**: For upscaling services
- ✅ **RunPod**: For cloud GPU provisioning

---

## 🎨 Using the GUI

### Upload Images

1. Navigate to **Media** tab
2. Click "Upload" or drag & drop images
3. Supported formats: JPG, PNG, WEBP, BMP
4. Max size: 50MB per image

### Create Dataset

1. Go to **Datasets** tab
2. Click "Create Dataset"
3. Select uploaded images
4. Enter:
   - Dataset name
   - Trigger word (e.g., "ohwx person")
   - Caption template
5. Click "Create"

### Train LoRA

1. Go to **Characters** tab
2. Click "Create Character"
3. Select your dataset
4. Configure training (or use defaults):
   - Steps: 800-1000
   - Learning rate: 8e-4
   - Batch size: 1
   - Rank: 8-16
5. Click "Start Training"

### Generate Images

1. Go to **Inference** tab
2. Select trained character
3. Enter prompt
4. Adjust settings:
   - LoRA weight: 0.7-0.8
   - Steps: 30
   - Resolution: 1024x1024
5. Click "Generate"

**No safety checks by default!** Generate freely.

---

## 💻 CLI Usage

### Train LoRA

```bash
python train_character.py \
  --name "my_character" \
  --input "path/to/image.jpg" \
  --steps 1000 \
  --batch_size 1 \
  --lr 8e-4 \
  --rank_dim 16
```

### Generate Images (No Safety Check)

```bash
python test_character.py \
  --character_name "my_character" \
  --prompt "portrait of a person, professional photography" \
  --batch_size 4 \
  --lora_weight 0.75
  # No --safety_check flag needed (disabled by default)
```

### Enable Safety Check (Optional)

```bash
python test_character.py \
  --character_name "my_character" \
  --prompt "your prompt here" \
  --safety_check
  # ^ Add this flag if you want safety checking
```

---

## 🔧 Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# Hugging Face
HF_TOKEN=your_token_here
HF_HOME=/workspace/.cache/huggingface  # Change cache location

# API Keys
CIVITAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
FAL_KEY=your_key
RUNPOD_API_KEY=your_key
```

### Training Parameters

Edit `scripts/character_lora.yaml` for advanced training config:

```yaml
train:
  batch_size: 1      # Increase for faster training (if VRAM allows)
  steps: 1000        # More steps = better quality
  lr: 8e-4          # Learning rate

network:
  linear: 16         # LoRA rank (higher = more capacity)
  linear_alpha: 16   # Usually same as rank
```

---

## 🎯 Workflow Examples

### High-Quality Portrait LoRA

1. **Collect 20-50 diverse images**
   - Different angles, expressions, lighting
   - High resolution (1024px+)
   - Clear face visibility

2. **Upload & Create Dataset**
   - Upload all images to GUI
   - Create dataset with trigger word

3. **Train with Optimal Settings**
   ```bash
   python train_character.py \
     --name "character" \
     --input "reference.jpg" \
     --steps 1500 \
     --rank_dim 24 \
     --lr 5e-4
   ```

4. **Generate Images**
   ```bash
   python test_character.py \
     --character_name "character" \
     --prompt "professional portrait, studio lighting, detailed" \
     --batch_size 4 \
     --lora_weight 0.8 \
     --test_dim 1024
   ```

### Quick Character Test

1. **Single reference image**
2. **Fast training**:
   ```bash
   python train_character.py \
     --name "test" \
     --input "image.jpg" \
     --steps 500 \
     --rank_dim 8
   ```
3. **Generate tests**:
   ```bash
   python test_character.py \
     --character_name "test" \
     --prompt "portrait" \
     --batch_size 8
   ```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solutions:**
- Reduce batch_size to 1
- Lower training resolution (--train_dim 512)
- Enable gradient checkpointing in config

### "Safety check blocking content"

**This should NOT happen** (safety is disabled by default)

If it does:
```bash
# Verify safety is disabled:
grep "safety_check" test_character.py
# Should show: default=False
```

### GUI won't start

```bash
# Check dependencies
cd charforge-gui/backend
pip install -r requirements.txt

cd ../frontend
npm install

# Restart
cd ..
bash start-dev.sh
```

### RunPod can't access GUI

1. Check ports are exposed: 5173, 8000
2. Use RunPod's "Connect" button → HTTP Service
3. Or use TCP port forwarding

---

## 📊 Performance Tips

### Faster Training

1. **Enable caching**: Keep `cache_latents_to_disk: true` in config
2. **Use bf16**: Already enabled in default config
3. **Optimize batch size**: Use 1 for 12GB VRAM, 2 for 24GB+

### Better Quality

1. **More training steps**: 1500-2000 for best results
2. **Higher rank**: Use 24-32 for complex characters
3. **Better dataset**: 30+ diverse, high-quality images
4. **Adjust LoRA weight**: Try 0.7-0.9 during generation

### Faster Generation

1. **Fewer steps**: 25-30 is good balance
2. **Lower resolution**: 768x768 for testing, 1024x1024 for finals
3. **Batch generation**: Generate 4-8 images at once

---

## 🎓 Best Practices

### Dataset Quality

- ✅ 15-50 images optimal
- ✅ High resolution (1024px+)
- ✅ Variety: angles, expressions, lighting
- ✅ Clear, sharp images
- ❌ Avoid: blurry, low-res, covered faces

### Training Settings

- **Quick test**: 500 steps, rank 8
- **Good quality**: 1000 steps, rank 16
- **Best quality**: 1500+ steps, rank 24-32

### Prompting

- Be specific: "professional portrait, studio lighting, sharp focus"
- Reference quality: "high quality, detailed, 8k uhd"
- Control style: "cinematic, dramatic lighting, bokeh"
- **No restrictions** - generate whatever you want!

---

## 🆘 Getting Help

### Check Logs

**GUI Logs:**
```bash
# Backend logs
cd charforge-gui/backend
tail -f app.log

# Training logs
tail -f /path/to/scratch/<character>/timing.log
```

**CLI Logs:**
Training outputs directly to terminal

### Common Issues

1. **Import errors**: `pip install -r requirements.txt`
2. **Model not found**: `python install.py`
3. **CUDA errors**: Check `nvidia-smi`, reduce batch size
4. **API errors**: Verify API keys in `.env`

---

## 🚀 Advanced Usage

### Custom Model Integration

Add custom models by editing:
- `scripts/character_lora.yaml` - Training config
- `helpers.py` - Model loading logic

### RunPod Automation

Create custom RunPod templates with:
1. Pre-installed CharForge
2. Pre-cached models
3. Auto-start script

### Batch Processing

Process multiple characters:
```bash
for name in char1 char2 char3; do
    python train_character.py \
        --name "$name" \
        --input "images/$name.jpg" \
        --steps 1000
done
```

---

## 📝 Summary

**You're ready to:**
1. ✅ Upload images via GUI
2. ✅ Train photorealistic LoRAs
3. ✅ Generate unrestricted images
4. ✅ Run on RunPod with one command
5. ✅ Scale to cloud GPUs as needed

**Key Files:**
- `.env` - Your API keys (already configured)
- `runpod_quickstart.sh` - One-command RunPod setup
- `test_character.py` - Safety disabled by default
- `charforge-gui/` - Full web interface

**No restrictions. Full creative freedom. Let's create!** 🎨🚀
