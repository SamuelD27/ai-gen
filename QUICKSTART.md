# ai-gen Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone and enter directory
cd ai-gen

# 2. Run installation script
bash install.sh

# 3. Edit environment variables
nano .env
# Add your HF_TOKEN and other API keys

# 4. Activate environment
source venv/bin/activate
```

## Your First LoRA (30-60 minutes)

### Step 1: Prepare Your Dataset (5 min)

```bash
# Create dataset directory
mkdir -p datasets/my_person

# Add 10-50 images of the same person to:
# datasets/my_person/
```

### Step 2: Clean & Preprocess (5 min)

```bash
# Clean dataset (removes low quality, duplicates)
python -m cli.main dataset clean \
  datasets/my_person \
  datasets/my_person_cleaned \
  --require-faces

# Preprocess (smart crop + resize to 1024px)
python -m cli.main dataset preprocess \
  datasets/my_person_cleaned \
  datasets/my_person_processed \
  --resolution 1024 \
  --smart-crop

# Caption images
python -m cli.main dataset caption \
  datasets/my_person_processed \
  --backend blip2
```

### Step 3: Train LoRA (30-40 min on RTX 4090)

```bash
python -m cli.main train lora \
  datasets/my_person_processed \
  my_person_lora \
  --model flux \
  --steps 1000 \
  --batch-size 1 \
  --lr 1e-4 \
  --rank 16 \
  --resolution 512
```

Your LoRA will be saved to: `./output/loras/my_person_lora/`

### Step 4: Generate Images (1-2 min)

```bash
python -m cli.main generate image \
  "portrait of a person, professional photography, high quality, detailed" \
  --lora ./output/loras/my_person_lora/my_person_lora.safetensors \
  --num-images 4 \
  --resolution 1024 \
  --steps 30 \
  --lora-strength 0.8
```

Images saved to: `./output/images/`

### Step 5: Generate Video (2-3 min)

```bash
python -m cli.main generate video \
  "person smiling and looking at camera, professional lighting" \
  --lora ./output/loras/my_person_lora/my_person_lora.safetensors \
  --num-frames 16 \
  --fps 8 \
  --resolution 512 \
  --lora-strength 0.8 \
  --output ./output/videos/my_video.mp4
```

## Using the GUI

Much easier for interactive work:

```bash
# Start GUI
python gui/app.py

# Access at: http://localhost:7860
```

The GUI provides:
- Visual dataset processing
- Interactive training configuration
- Real-time image/video generation
- LoRA management tools

## Common Workflows

### High-Quality Portrait Generation

```bash
# Use higher steps and resolution
python -m cli.main generate image \
  "close-up portrait, professional photography, studio lighting, 8k uhd, highly detailed" \
  --lora ./output/loras/my_person_lora/my_person_lora.safetensors \
  --num-images 1 \
  --resolution 2048 \
  --steps 50 \
  --lora-strength 0.85
```

### Batch Generation with Different Prompts

```python
# Create prompts.txt with one prompt per line
cat > prompts.txt << EOF
portrait in a park, natural lighting
professional headshot, studio lighting
casual photo, urban background
artistic portrait, dramatic lighting
EOF

# Generate for each prompt
while read prompt; do
    python -m cli.main generate image "$prompt" \
      --lora ./output/loras/my_person_lora/my_person_lora.safetensors \
      --num-images 4
done < prompts.txt
```

### Merging Multiple LoRAs

```bash
# Merge two character LoRAs
python -m cli.main lora merge \
  ./output/loras/person1/person1.safetensors \
  ./output/loras/person2/person2.safetensors \
  ./output/loras/merged.safetensors \
  --weights 0.6,0.4
```

## Troubleshooting

### Out of VRAM
- Reduce `--batch-size` to 1
- Lower `--resolution` (try 512 or 768)
- Enable gradient checkpointing in config.yaml

### LoRA doesn't capture likeness
- Increase training steps (try 1500-2000)
- Use more diverse training images (20-50 images)
- Try higher LoRA rank (24 or 32)
- Increase lora-strength (0.9-1.0)

### Training is slow
- Check CUDA is enabled: `python -c "import torch; print(torch.cuda.is_available())"`
- Enable xformers in config.yaml
- Use bf16 precision
- Cache latents (enabled by default)

### Images are blurry
- Increase inference steps (40-50)
- Use higher resolution (1536 or 2048)
- Try different guidance scale (6-10)
- Enable face restoration in config.yaml

## Tips for Best Results

### Dataset Quality
- 15-30 images is optimal
- Variety: different angles, expressions, lighting
- High resolution: 1024px+ minimum
- Clear face visibility in most images
- Remove sunglasses/heavy makeup variations

### Training Parameters
- **Flux**: 800-1200 steps, rank 16, lr 1e-4
- **SDXL**: 1500-2500 steps, rank 32, lr 5e-5
- Start with defaults, adjust based on results

### Generation Prompts
- Be specific about style, lighting, composition
- Reference quality: "high quality, detailed, professional"
- Avoid generic prompts
- Experiment with lora-strength (0.6-1.0)

### Prompt Examples

```
# Portrait
"portrait of a person, professional photography, studio lighting, sharp focus, 8k uhd, highly detailed, bokeh background"

# Environmental
"person standing in a modern office, natural window lighting, professional attire, photorealistic, depth of field"

# Artistic
"cinematic portrait of a person, dramatic lighting, film grain, 35mm photograph, shallow depth of field"

# Casual
"candid photo of a person smiling, outdoor setting, golden hour lighting, natural, authentic moment"
```

## Next Steps

1. Experiment with different prompts and settings
2. Try video generation with motion control
3. Merge LoRAs for creative combinations
4. Explore ControlNet for precise composition
5. Set up cloud compute for faster training

See [README_NEW.md](README_NEW.md) for complete documentation.
