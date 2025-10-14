# MASUKA Setup Guide

Complete guide for setting up and running MASUKA (AI Generation Platform with LoRA Training).

## Table of Contents
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Local Setup](#local-setup)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Features](#features)

---

## Quick Start (Google Colab)

### 1. Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Change runtime to **GPU (T4 or A100)**
   - Runtime → Change runtime type → GPU

### 2. Run Setup
```python
# Clone and setup MASUKA
!git clone https://github.com/SamuelD27/ai-gen.git /content/ai-gen
%cd /content/ai-gen
!python masuka_colab_setup.py
```

### 3. Access the GUI
- After setup completes, you'll see an ngrok URL
- Click the URL to access the web interface
- Default user is pre-configured (no login required)

### 4. Usage
1. **Upload Images**: Go to Media tab, upload training images
2. **Create Dataset**: Select images and create a dataset
3. **Create Character**: Choose dataset and trigger word
4. **Train LoRA**: Start training (takes 5-15 minutes)
5. **Generate Images**: Use trained character for inference

---

## Local Setup

### Prerequisites
- Python 3.10+
- Node.js 20+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# 1. Clone repository
git clone https://github.com/SamuelD27/ai-gen.git
cd ai-gen

# 2. Setup backend
cd charforge-gui/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup frontend
cd ../frontend
npm install

# 4. Configure environment
cp ../../.env.example .env
# Edit .env with your API keys
```

### Running Locally

Terminal 1 (Backend):
```bash
cd charforge-gui/backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (Frontend):
```bash
cd charforge-gui/frontend
npm run dev
```

Access at: http://localhost:5173

---

## Database Migrations

After pulling updates from the repository, you may need to run database migrations to update your database schema.

### Running Migrations

**Google Colab:**
```python
# Run Cell 3 in the Colab notebook
# This will automatically apply all pending migrations
```

**Local Setup:**
```bash
cd charforge-gui/backend
python run_migrations.py
```

### Migration Commands

```bash
# Run all migrations
python run_migrations.py

# List available migrations
python run_migrations.py --list

# Run a specific migration
python run_migrations.py --single add_error_message_column.py
```

### When to Run Migrations

Run migrations when:
- You pull new code from the repository
- You see database-related errors (e.g., "no such column")
- You're setting up a fresh installation
- Documentation mentions schema changes

**Note**: Migrations are safe to run multiple times. If a migration has already been applied, it will be skipped automatically.

---

## Configuration

### Required API Keys

1. **HuggingFace Token** (Required for model downloads)
   - Get from: https://huggingface.co/settings/tokens
   - Add to `.env`: `HF_TOKEN=hf_your_token_here`

2. **CivitAI API Key** (Optional, for additional models)
   - Get from: https://civitai.com/user/account
   - Add to `.env`: `CIVITAI_API_KEY=your_key_here`

3. **Google API Key** (Optional, for Gemini captioning)
   - Get from: https://console.cloud.google.com/apis/credentials
   - Add to `.env`: `GOOGLE_API_KEY=your_key_here`

4. **Video Generation APIs** (Optional)
   - CometAPI: https://www.comet.com/
   - OpenAI: https://platform.openai.com/api-keys

### Directory Structure
```
ai-gen/
├── charforge-gui/
│   ├── backend/        # FastAPI backend
│   │   ├── app/
│   │   │   ├── api/    # API endpoints
│   │   │   ├── core/   # Database, config, auth
│   │   │   ├── services/ # Business logic
│   │   │   └── utils/  # Validation utilities
│   │   └── database.db
│   └── frontend/       # Vue 3 frontend
│       ├── src/
│       └── dist/
├── scratch/           # Training workspaces
├── media/            # Uploaded images
├── results/          # Generated outputs
├── loras/            # Trained LoRA models
└── .env             # Configuration
```

---

## Troubleshooting

### Common Issues

#### 1. Training Fails Immediately
**Symptoms**: Training starts but fails at 0%

**Solutions**:
- Check that dataset has images
- Verify HF_TOKEN is valid
- Ensure GPU has enough VRAM (8GB minimum)
- Check logs in Colab output

#### 2. "Failed to load datasets" Error
**Cause**: Database migration needed

**Solution**:
```python
# In Colab
!cd /content/ai-gen/charforge-gui/backend && python -c "from app.core.database import Base, engine; Base.metadata.create_all(engine)"
```

#### 3. "no such column: training_sessions.error_message"
**Symptoms**: Error when deleting characters or viewing training sessions

**Cause**: Database schema is outdated (missing error_message column)

**Solution**:
```python
# In Colab - Run Cell 3 to apply migrations
# Or run manually:
!cd /content/ai-gen/charforge-gui/backend && python run_migrations.py
```

```bash
# Local setup
cd charforge-gui/backend
python run_migrations.py
```

**Why this happens**: The error_message column was added to track detailed error information for failed training sessions. If your database was created before this update, you need to run migrations to add the column.

#### 4. Training Session Stuck
**Symptom**: Session shows "pending" forever

**Solution**:
- Use "Cancel Training" button in UI
- Or run cleanup cell in Colab notebook

#### 5. Out of Memory
**Solutions**:
- Reduce batch size to 1
- Reduce train_dim to 256
- Use smaller image resolution
- Restart Colab session

#### 6. No GPU Detected
**Solution**:
- Runtime → Change runtime type → T4 GPU or A100 GPU
- Restart runtime

### Getting Help

1. Check the logs:
   - Colab: Scroll through cell output
   - Local: Check terminal where backend is running

2. Common log locations:
   ```
   Backend logs: stdout/stderr from uvicorn
   Training logs: Check work_dir in scratch/
   ```

3. Enable debug mode:
   ```bash
   # In .env
   ENVIRONMENT=development
   ```

---

## Features

### LoRA Training
- **Dataset-based training**: Upload multiple images
- **Single image training**: Quick character creation
- **Configurable parameters**: Steps, learning rate, batch size
- **Progress tracking**: Real-time training progress
- **Persistent storage**: All data saved to Google Drive (Colab)

### Image Generation
- **Multiple model support**: Flux.1, SD3.5, SDXL
- **LoRA integration**: Use trained characters
- **Batch generation**: Generate multiple images
- **Quality controls**: Face enhancement, safety filters
- **Prompt optimization**: AI-enhanced prompts

### Video Generation (Beta)
- **Multiple providers**: Sora 2, VEO3, Runway Gen-4
- **Character consistency**: Use trained LoRAs in videos
- **Customizable**: Duration, resolution, motion intensity
- **High quality**: Up to 4K resolution

### Dataset Management
- **Bulk upload**: Upload multiple images at once
- **Auto-captioning**: AI-generated captions
- **Image processing**: Resize, crop, flip
- **Quality filtering**: Remove low-quality images
- **Trigger words**: Define character identifiers

### Additional Features
- **No authentication**: Quick start, no signup needed
- **Persistent storage**: Google Drive integration (Colab)
- **Modern UI**: Clean, responsive interface
- **Error recovery**: Auto-cleanup of stuck sessions
- **API documentation**: OpenAPI/Swagger at `/docs`

---

## Advanced Configuration

### Training Parameters

```python
# Recommended for different scenarios:

# Quick test (3-5 minutes)
steps = 200
batch_size = 1
learning_rate = 1e-3

# Standard quality (10-15 minutes)
steps = 800
batch_size = 1
learning_rate = 8e-4

# High quality (30+ minutes)
steps = 1500
batch_size = 2
learning_rate = 5e-4
```

### GPU Memory Usage

| Configuration | VRAM Required |
|--------------|---------------|
| Minimal (256px, bs=1) | 6GB |
| Standard (512px, bs=1) | 10GB |
| High Quality (1024px, bs=1) | 16GB |
| Ultra (1024px, bs=2) | 24GB+ |

### Database Backup

```bash
# Backup database
cp database.db database_backup.db

# Restore database
cp database_backup.db database.db
```

---

## Development

### Running Tests
```bash
cd charforge-gui/backend
pytest tests/
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

See LICENSE file for details.

## Credits

- Built with FastAPI, Vue 3, and CharForge
- Uses Flux.1, SD3.5, and SDXL models
- Video generation via CometAPI, OpenAI, Google

---

## Support

For issues and questions:
1. Check troubleshooting section above
2. Review logs for error messages
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System info (GPU, OS, etc.)
