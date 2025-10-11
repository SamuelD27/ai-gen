# 🚀 Google Colab Quick Start

The easiest way to use ai-gen is through Google Colab - no installation, free GPU, just click and run!

## 📋 One-Click Setup

1. **Open the notebook**:
   - Click here: [Open in Colab](https://colab.research.google.com/github/SamuelD27/ai-gen/blob/main/ai_gen_colab.ipynb)

2. **Enable GPU**:
   - Go to `Runtime` → `Change runtime type`
   - Select `T4 GPU` (free tier) or `A100` (Pro)
   - Click `Save`

3. **Run all cells**:
   - Click `Runtime` → `Run all`
   - Or manually run each cell with the ▶️ button

4. **Wait for setup** (~10-15 minutes first time):
   - Installing dependencies
   - Downloading models
   - Starting services

5. **Get your URL**:
   - After the last cell runs, you'll see: `🌐 https://XXXXX.ngrok-free.app`
   - Click that URL to access your GUI!

## 🎨 What You Can Do

### Train a LoRA
1. Upload 15-30 images of your subject
2. Create a dataset
3. Start training (takes 20-40 minutes)
4. Download your trained LoRA

### Generate Images
1. Select your LoRA
2. Enter a prompt
3. Generate ultra-realistic images
4. Download results

### No Restrictions
- Safety filters disabled by default
- Full creative freedom
- No content moderation

## 💡 Tips

### Free vs Pro Colab

| Feature | Free | Pro |
|---------|------|-----|
| GPU | T4 (16GB) | A100 (40GB) |
| Session | ~12 hours | ~24 hours |
| Training speed | 1x | 3-4x faster |
| Cost | Free | $10/month |

### Saving Your Work

**Option 1: Download to your computer**
- Navigate to `/content/ai-gen/output/`
- Right-click files → Download

**Option 2: Save to Google Drive**
```python
# Run this in a new cell
from google.colab import drive
drive.mount('/content/drive')

# Copy outputs to Drive
!cp -r /content/ai-gen/output /content/drive/MyDrive/ai-gen-output
```

**Option 3: Push to GitHub**
```python
# In a new cell
!cd /content/ai-gen/output && \
  git init && \
  git add . && \
  git commit -m "My outputs" && \
  git remote add origin YOUR_REPO_URL && \
  git push -u origin main
```

### GPU Out of Memory?

If you get OOM errors during training:

```python
# Edit the training config (in GUI or via CLI)
# Reduce these values:
- batch_size: 1 (instead of 2)
- resolution: 512 (instead of 1024)
- gradient_accumulation_steps: 4 (helps with small batches)
```

### Extending Session Time

Colab disconnects after inactivity:

```python
# Run this in a new cell to keep session alive
import time
from IPython.display import display, Javascript

while True:
    display(Javascript('window.keepAlive = setInterval(() => {console.log("Keeping alive")}, 60000)'))
    time.sleep(600)
```

## 🔧 CLI Usage in Colab

You can also use the CLI directly in notebook cells:

### Train a LoRA
```python
!cd /content/ai-gen && python train_character.py \
    --name "john_doe" \
    --trigger_word "johndoe" \
    --images_path "./datasets/john_doe" \
    --steps 1000 \
    --learning_rate 1e-4
```

### Generate Images
```python
!cd /content/ai-gen && python test_character.py \
    --lora "john_doe" \
    --prompt "johndoe wearing a business suit in an office" \
    --num_images 4 \
    --steps 30
```

### Batch Generation
```python
prompts = [
    "johndoe at the beach",
    "johndoe wearing sunglasses",
    "johndoe in winter clothes",
    "johndoe professional headshot"
]

for prompt in prompts:
    !cd /content/ai-gen && python test_character.py \
        --lora "john_doe" \
        --prompt "{prompt}" \
        --num_images 2
```

## 📊 Monitoring

### Check GPU Usage
```python
!nvidia-smi
```

### Check Disk Space
```python
!df -h
```

### View Training Logs
```python
!tail -f /content/ai-gen/output/loras/LORA_NAME/training.log
```

## 🐛 Troubleshooting

### "ngrok URL not working"
```python
# In a new cell, manually check ngrok
import requests
response = requests.get('http://localhost:4040/api/tunnels')
print(response.json())
```

### "Services not starting"
```python
# Check if processes are running
!ps aux | grep -E "uvicorn|node"

# Restart services
!cd /content/ai-gen/charforge-gui/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 &
!cd /content/ai-gen/charforge-gui/frontend && npm run dev -- --host 0.0.0.0 --port 5173 &
```

### "Can't download models"
```python
# Check HuggingFace token
import os
print(f"HF_TOKEN: {os.environ.get('HF_TOKEN', 'NOT SET')}")

# Manually download a model
!huggingface-cli download black-forest-labs/FLUX.1-dev --token $HF_TOKEN
```

### "Out of storage"
```python
# Clear cache and outputs
!rm -rf /content/.cache/*
!rm -rf /content/ai-gen/output/images/*
!rm -rf /content/ai-gen/workspace/*
```

## ❓ FAQ

**Q: Is this really free?**
A: Yes! Google Colab's free tier includes GPU access. Pro tier ($10/month) gives faster GPUs and longer sessions.

**Q: Will my files be saved?**
A: Colab storage is temporary. Download your outputs or save to Google Drive before session ends.

**Q: Can I use my own models?**
A: Yes! Upload them to `/content/ai-gen/models/` or provide URLs in the config.

**Q: How long does training take?**
A: On T4 GPU: 30-60 minutes for 1000 steps. On A100: 10-20 minutes.

**Q: What if my session disconnects?**
A: Just re-run the notebook. Models are cached and will download faster the second time.

## 🔗 Links

- [Main Repository](https://github.com/SamuelD27/ai-gen)
- [Open Colab Notebook](https://colab.research.google.com/github/SamuelD27/ai-gen/blob/main/ai_gen_colab.ipynb)
- [Full Documentation](README.md)
- [Issue Tracker](https://github.com/SamuelD27/ai-gen/issues)

---

**Ready to start?** → [Open in Colab](https://colab.research.google.com/github/SamuelD27/ai-gen/blob/main/ai_gen_colab.ipynb)
