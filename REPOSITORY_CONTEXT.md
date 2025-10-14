# MASUKA Repository - Complete Context Summary

**Last Updated:** 2025-01-14
**Repository:** [SamuelD27/ai-gen](https://github.com/SamuelD27/ai-gen)
**Version:** 2.0.0
**Status:** Production-ready with active development

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Technology Stack](#technology-stack)
4. [Core Components](#core-components)
5. [Recent Changes & Fixes](#recent-changes--fixes)
6. [Database Architecture](#database-architecture)
7. [Deployment](#deployment)
8. [Known Issues & Solutions](#known-issues--solutions)
9. [Development Status](#development-status)

---

## 🎯 Project Overview

**MASUKA** is a production-ready AI generation platform that combines cutting-edge image and video generation capabilities with an intuitive web interface. It's designed for artists, content creators, and AI enthusiasts.

### Key Features

- **Direct Image Generation** - No training required
  - Flux.1 Dev & Schnell (best-in-class quality)
  - Stable Diffusion 3.5 Large & Medium
  - Playground v2.5, SDXL Base
  - Full parameter control (steps, guidance, dimensions, seed)
  - Batch generation up to 4 images

- **Video Generation**
  - Sora 2 integration (via CometAPI)
  - Google VEO3 support
  - Runway Gen-4 API
  - Up to 60 seconds of high-quality video

- **Custom LoRA Training**
  - Fast training (800-2000 steps, 5-15 minutes)
  - Automatic dataset curation
  - AI-powered captioning (BLIP-2, Florence-2)
  - 20-30 images for optimal results

- **Modern Web Interface**
  - Vue 3 + TypeScript frontend
  - FastAPI + Python backend
  - Real-time progress tracking
  - Generation history with metadata

- **Cloud Deployment**
  - Google Colab optimized
  - One-click deployment
  - ngrok tunneling
  - Persistent storage (Google Drive integration)

---

## 📁 Repository Structure

```
CharForgex/
├── charforge-gui/          # Main web application
│   ├── backend/           # FastAPI backend (632KB)
│   │   ├── app/
│   │   │   ├── api/       # API endpoints
│   │   │   │   ├── auth.py          # Authentication
│   │   │   │   ├── training.py      # LoRA training
│   │   │   │   ├── inference.py     # Image generation
│   │   │   │   ├── video.py         # Video generation
│   │   │   │   ├── datasets.py      # Dataset management
│   │   │   │   ├── models.py        # Model management
│   │   │   │   ├── settings.py      # User settings
│   │   │   │   └── media.py         # Media handling
│   │   │   ├── core/      # Core functionality
│   │   │   │   ├── database.py      # SQLAlchemy models
│   │   │   │   ├── config.py        # Configuration
│   │   │   │   ├── auth.py          # Auth logic
│   │   │   │   ├── security.py      # Security middleware
│   │   │   │   └── middleware.py    # Custom middleware
│   │   │   ├── services/  # Business logic
│   │   │   │   ├── settings_service.py
│   │   │   │   └── dataset_import.py
│   │   │   └── main.py    # FastAPI app entry
│   │   ├── migrations/    # Database migrations
│   │   │   ├── add_error_message_column.py
│   │   │   └── add_dataset_tables.py
│   │   └── run_migrations.py
│   └── frontend/          # Vue.js frontend
│       ├── src/
│       │   ├── components/  # Vue components
│       │   ├── views/       # Page views
│       │   ├── stores/      # Pinia stores
│       │   └── router/      # Vue Router
│       └── package.json     # Frontend dependencies
│
├── ai_toolkit/            # LoRA training toolkit
│   └── # Flux LoRA training implementation
│
├── core/                  # Core utilities
│   ├── prompt_optimizer.py
│   ├── validation_utils.py
│   └── settings.py
│
├── training/              # Training workflows
│   ├── train_lora.py
│   └── config/
│
├── inference/             # Image generation
│   ├── generate.py
│   └── models/
│
├── generation/            # Generation utilities
│   └── prompt_helpers.py
│
├── dataset/               # Dataset processing
│   ├── prepare_dataset.py
│   ├── caption_images.py
│   └── validate.py
│
├── scripts/               # Utility scripts
│   ├── setup_colab.py
│   └── install_deps.py
│
├── cloud/                 # Cloud deployment
│   ├── colab_setup.py
│   └── runpod_setup.py
│
├── cli/                   # Command-line interface
│   └── masuka_cli.py
│
├── gui/                   # Legacy Gradio interface (32KB)
│   └── app.py
│
├── LoRACaptioner/         # Caption generation
├── ComfyUI_AutoCropFaces/ # Face cropping utility
└── MV_Adapter/            # Multi-view adapter

# Root configuration files
├── masuka_colab_setup.py  # Main Colab setup script
├── ai_gen_colab.ipynb     # Colab notebook
├── requirements-masuka.txt # Python dependencies
├── .env.example           # Environment template
├── .gitignore            # Git ignore patterns
└── README.md             # Main documentation
```

### File Statistics
- **Python files:** 79
- **Markdown docs:** 578
- **Total size:** ~235MB (was ~425MB before node_modules cleanup)

---

## 🛠️ Technology Stack

### Backend
- **Framework:** FastAPI 0.115+
- **Database:** SQLite with SQLAlchemy ORM
- **Authentication:** JWT tokens, optional auth mode
- **Security:** Rate limiting, CORS, security headers
- **Python:** 3.10+
- **ML Framework:** PyTorch 2.5.1+

### Frontend
- **Framework:** Vue 3 with TypeScript
- **Build Tool:** Vite
- **UI Library:** Tailwind CSS
- **Components:** HeadlessUI, Heroicons
- **HTTP Client:** Axios
- **State Management:** Pinia

### AI Models
- **Image Generation:**
  - Flux.1 Dev/Schnell (Black Forest Labs)
  - Stable Diffusion 3.5 Large/Medium (Stability AI)
  - Playground v2.5
  - SDXL Base

- **Video Generation:**
  - Sora 2 (OpenAI via Comet)
  - Google VEO3
  - Runway Gen-4

- **Captioning:**
  - BLIP-2 (Salesforce)
  - Florence-2 (Microsoft)

- **Training:**
  - Custom Flux LoRA training via ai_toolkit
  - Supports 800-2000 steps
  - Configurable rank dimensions (4-128)

### Deployment
- **Platforms:** Google Colab, RunPod, Local
- **Tunneling:** ngrok for external access
- **Storage:** Google Drive for persistence
- **Container:** Docker support available

---

## 🔧 Core Components

### 1. Backend API (`charforge-gui/backend/`)

#### Database Models ([database.py:16-119](charforge-gui/backend/app/core/database.py#L16-L119))
```python
- User: Authentication and user management
- Character: Character definitions for training
- TrainingSession: LoRA training state and history
- InferenceJob: Image generation jobs
- AppSettings: User preferences and settings
- Dataset: Multi-image dataset management
- DatasetImage: Individual images in datasets
```

#### API Endpoints
1. **Authentication** ([auth.py](charforge-gui/backend/app/api/auth.py))
   - POST `/auth/login` - User login
   - POST `/auth/register` - User registration
   - Optional auth mode for single-user setups

2. **Training** ([training.py](charforge-gui/backend/app/api/training.py))
   - POST `/training/characters` - Create character
   - GET `/training/characters` - List characters
   - DELETE `/training/characters/{id}` - Delete character
   - POST `/training/characters/{id}/train` - Start training
   - POST `/training/characters/{id}/cancel` - Cancel training
   - GET `/training/training` - List all training sessions
   - GET `/training/characters/{id}/training` - Character training history

3. **Inference** ([inference.py](charforge-gui/backend/app/api/inference.py))
   - POST `/inference/generate` - Generate images
   - GET `/inference/jobs` - List generation jobs
   - GET `/inference/jobs/{id}` - Job status

4. **Video** ([video.py](charforge-gui/backend/app/api/video.py))
   - POST `/video/generate` - Generate video
   - GET `/video/jobs` - List video jobs
   - GET `/video/jobs/{id}` - Video job status

5. **Datasets** ([datasets.py](charforge-gui/backend/app/api/datasets.py))
   - POST `/datasets/` - Create dataset
   - GET `/datasets/` - List datasets
   - POST `/datasets/{id}/images` - Upload images
   - PUT `/datasets/{id}/images/{img_id}` - Update caption
   - DELETE `/datasets/{id}` - Delete dataset

6. **Models** ([models.py](charforge-gui/backend/app/api/models.py))
   - GET `/models/` - List available models
   - POST `/models/download` - Download model
   - GET `/models/installed` - List installed models

7. **Settings** ([settings.py](charforge-gui/backend/app/api/settings.py))
   - GET `/settings/{key}` - Get setting
   - PUT `/settings/{key}` - Update setting
   - GET `/settings/` - Get all settings

#### Security ([security.py](charforge-gui/backend/app/core/security.py))
- Rate limiting: 100 requests/minute per IP
- CORS with configurable origins
- Security headers (CSP, X-Frame-Options, etc.)
- JWT token authentication
- Password hashing with bcrypt

#### Configuration ([config.py](charforge-gui/backend/app/core/config.py))
```python
Settings:
- DATABASE_URL: SQLite database path
- SECRET_KEY: JWT signing key
- ENABLE_AUTH: Toggle authentication
- ALLOW_REGISTRATION: Allow new users
- MEDIA_DIR: Media storage path
- API keys: HF_TOKEN, CIVITAI_API_KEY, GOOGLE_API_KEY, etc.
```

### 2. Frontend Application (`charforge-gui/frontend/src/`)

#### Key Components
- **CharacterGrid**: Character management interface
- **TrainingForm**: LoRA training configuration
- **GenerationPanel**: Image generation interface
- **VideoGeneration**: Video creation interface
- **DatasetManager**: Dataset upload and management
- **ModelSelector**: Model selection and download
- **HistoryViewer**: Generation history browser

#### Views
- **Dashboard**: Main overview
- **Training**: Character and LoRA training
- **Generate**: Image generation
- **Video**: Video generation
- **Datasets**: Dataset management
- **Settings**: User preferences

#### State Management (Pinia)
- `authStore`: User authentication state
- `modelsStore`: Available models
- `trainingStore`: Training sessions
- `generationStore`: Generation jobs
- `settingsStore`: User settings

### 3. Training System (`ai_toolkit/` & `training/`)

#### Training Pipeline
1. **Dataset Preparation** ([dataset/prepare_dataset.py](dataset/prepare_dataset.py))
   - Image validation and resizing
   - Face detection and cropping
   - Quality filtering

2. **Captioning** ([dataset/caption_images.py](dataset/caption_images.py))
   - BLIP-2 or Florence-2 captioning
   - Caption templates
   - Trigger word injection

3. **Training** ([training/train_lora.py](training/train_lora.py))
   - Flux LoRA training via ai_toolkit
   - Progress tracking and logging
   - Early stopping and validation

4. **Validation** ([core/validation_utils.py](core/validation_utils.py))
   - Image quality checks
   - Caption validation
   - Dataset statistics

### 4. Deployment System

#### Colab Setup ([masuka_colab_setup.py](masuka_colab_setup.py))
The main setup script handles:
1. **Environment Detection**: Detects Colab vs local
2. **Dependency Installation**: Installs all required packages
3. **Model Downloads**: Downloads AI models
4. **Configuration**: Creates .env files with correct paths
5. **Database Migration**: Runs schema migrations
6. **Service Launch**: Starts backend and frontend
7. **Tunnel Creation**: Sets up ngrok tunnel
8. **Health Checks**: Validates everything is running

#### Colab Notebook ([ai_gen_colab.ipynb](ai_gen_colab.ipynb))
Single-cell setup that:
1. Mounts Google Drive for persistence
2. Downloads latest code from GitHub
3. Runs masuka_colab_setup.py
4. Validates database schema
5. Shows access URL

---

## 🔄 Recent Changes & Fixes

### Latest Commits (Last 10)

1. **CLEANUP: Remove node_modules from repository (192MB)** [16c1ee3a]
   - Removed 17,422 node_modules files
   - Updated .gitignore with proper patterns
   - Reduced repo size from 425MB to 235MB

2. **FIX: Database migration timing** [5a2ca7dd]
   - Fixed migrations running before .env creation
   - Updated setup script to run migrations at correct time
   - Simplified Colab notebook flow

3. **FIX: Import all models in migration** [4570cfde]
   - Fixed migration not creating tables properly
   - Now imports all model classes before create_all()

4. **FIX: Git clone error handling** [d02581c6]
   - Added verification after git clone
   - Improved error messages
   - Smart clone/pull logic

5. **FIX: Robust migration with table creation** [3ecd1ce3]
   - Migration creates tables if they don't exist
   - Added Drive mounting to Colab notebook

6. **FIX: Syntax error in Colab notebook** [030db708]
   - Fixed multi-line string issues in IPython
   - Use temp files for validation scripts

7. **FIX: Single-cell setup** [0ea67a06]
   - Consolidated Colab setup into one cell
   - Added automatic migrations
   - Graceful error handling

8. **FIX: Add database migration system** [7a1de0bb]
   - Created migration infrastructure
   - Added error_message column to training_sessions
   - Created run_migrations.py script

### Major Improvements (Recent Session)

#### Database Migration System
- **Problem:** Column `error_message` missing from existing databases
- **Solution:** Created migration system with idempotent scripts
- **Files:**
  - [add_error_message_column.py](charforge-gui/backend/migrations/add_error_message_column.py)
  - [run_migrations.py](charforge-gui/backend/run_migrations.py)
  - Updated [masuka_colab_setup.py:560-603](masuka_colab_setup.py#L560-L603)

#### Graceful Error Handling
- **Problem:** 500 errors when column missing
- **Solution:** Use `getattr()` with defaults in API
- **Location:** [training.py:726](charforge-gui/backend/app/api/training.py#L726)
```python
"error_message": getattr(session, 'error_message', None)
```

#### Colab Setup Simplification
- **Problem:** Multiple cells were confusing and error-prone
- **Solution:** Single comprehensive cell with clear steps
- **Benefits:**
  - No manual intervention needed
  - Clear progress indicators
  - Automatic error recovery

#### Repository Cleanup
- **Problem:** node_modules (192MB) committed to repo
- **Solution:** Removed from git, added to .gitignore
- **Impact:** Repo size reduced by 45%

---

## 💾 Database Architecture

### Schema Overview

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    hashed_password VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Characters table
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    input_image_path VARCHAR,        -- Optional, can use dataset
    dataset_id INTEGER,               -- Link to dataset
    trigger_word VARCHAR,             -- Trigger word for LoRA
    work_dir VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'created', -- created, training, completed, failed
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

-- Training sessions table
CREATE TABLE training_sessions (
    id INTEGER PRIMARY KEY,
    character_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    steps INTEGER DEFAULT 800,
    batch_size INTEGER DEFAULT 1,
    learning_rate FLOAT DEFAULT 8e-4,
    train_dim INTEGER DEFAULT 512,
    rank_dim INTEGER DEFAULT 8,
    pulidflux_images INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'pending', -- pending, running, completed, failed, cancelled
    progress FLOAT DEFAULT 0.0,
    log_file VARCHAR,
    error_message TEXT,              -- NEW: Error details for failed sessions
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Inference jobs table
CREATE TABLE inference_jobs (
    id INTEGER PRIMARY KEY,
    character_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    optimized_prompt TEXT,
    lora_weight FLOAT DEFAULT 0.73,
    test_dim INTEGER DEFAULT 1024,
    batch_size INTEGER DEFAULT 4,
    num_inference_steps INTEGER DEFAULT 30,
    do_optimize_prompt BOOLEAN DEFAULT 1,
    fix_outfit BOOLEAN DEFAULT 0,
    safety_check BOOLEAN DEFAULT 1,
    face_enhance BOOLEAN DEFAULT 0,
    status VARCHAR DEFAULT 'pending',
    output_paths TEXT,               -- JSON array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

-- App settings table
CREATE TABLE app_settings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    key VARCHAR NOT NULL,
    value TEXT,
    is_encrypted BOOLEAN DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR NOT NULL,
    trigger_word VARCHAR NOT NULL,
    caption_template TEXT,
    auto_caption BOOLEAN DEFAULT 1,
    resize_images BOOLEAN DEFAULT 1,
    crop_images BOOLEAN DEFAULT 1,
    flip_images BOOLEAN DEFAULT 0,
    quality_filter VARCHAR DEFAULT 'basic',
    image_count INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'created',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Dataset images table
CREATE TABLE dataset_images (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL,
    filename VARCHAR NOT NULL,
    original_filename VARCHAR NOT NULL,
    caption TEXT,
    processed BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
```

### Migration System

Migrations are located in [charforge-gui/backend/migrations/](charforge-gui/backend/migrations/)

#### Running Migrations

**Automatically (Colab):**
Migrations run automatically during setup at Step 8

**Manually (Local):**
```bash
cd charforge-gui/backend
python run_migrations.py
```

#### Creating New Migrations

1. Create new file in `migrations/` (e.g., `add_new_column.py`)
2. Implement `upgrade()` and `downgrade()` functions
3. Use SQLAlchemy text() for SQL commands
4. Make it idempotent (safe to run multiple times)

Example:
```python
from sqlalchemy import text
from app.core.database import engine

def upgrade():
    """Add new column."""
    with engine.connect() as conn:
        # Check if column exists
        result = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM pragma_table_info('table_name')
            WHERE name = 'column_name'
        """))

        exists = result.fetchone()[0] > 0

        if not exists:
            conn.execute(text("""
                ALTER TABLE table_name
                ADD COLUMN column_name TYPE
            """))
            conn.commit()
            print("✓ Added column_name column")
        else:
            print("✓ column_name already exists")

def downgrade():
    """Remove new column."""
    # SQLite doesn't support DROP COLUMN
    print("⚠ Manual downgrade required")

if __name__ == "__main__":
    upgrade()
```

---

## 🚀 Deployment

### Google Colab (Recommended for Testing)

**Option 1: Notebook**
1. Open [ai_gen_colab.ipynb](ai_gen_colab.ipynb) in Colab
2. Enable GPU (T4 or A100)
3. Run the single cell
4. Wait 10-15 minutes for setup
5. Access via ngrok URL

**Option 2: One-Liner**
```python
!wget https://raw.githubusercontent.com/SamuelD27/ai-gen/main/masuka_colab_setup.py && python masuka_colab_setup.py
```

**Environment Variables (Required):**
- `HF_TOKEN`: Hugging Face token for model downloads
- `FAL_KEY`: Fal.ai API key for video generation
- `GOOGLE_API_KEY`: Google API key for captioning
- `CIVITAI_API_KEY`: CivitAI key (optional)
- `COMET_API_KEY`: Comet API key for Sora (optional)

**Google Drive Structure:**
```
/content/drive/MyDrive/MASUKA/
├── database/
│   └── masuka.db          # Persistent database
├── media/
│   ├── uploads/           # Uploaded images
│   ├── results/           # Generated images
│   └── loras/             # Trained LoRA files
└── logs/                  # Training logs
```

### Local Development

**Requirements:**
- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Node.js 18+ (for frontend)

**Setup:**
```bash
# Clone repository
git clone https://github.com/SamuelD27/ai-gen.git
cd ai-gen

# Install Python dependencies
pip install -r requirements-masuka.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run migrations
cd charforge-gui/backend
python run_migrations.py

# Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
cd charforge-gui/frontend
npm install
npm run dev
```

### Docker Deployment

**Build:**
```bash
cd charforge-gui
docker-compose up -d
```

**Configuration:**
Edit `docker-compose.yml` to set environment variables

### RunPod Deployment

Use [cloud/runpod_setup.py](cloud/runpod_setup.py) for RunPod deployment
- Includes automatic setup
- GPU configuration
- Persistent storage setup

---

## ⚠️ Known Issues & Solutions

### 1. Database Migration Errors

**Issue:** `no such column: training_sessions.error_message`

**Cause:** Database created before migration system was added

**Solution:**
```bash
cd charforge-gui/backend
python run_migrations.py
```

**Status:** ✅ Fixed in latest version

### 2. Node Modules Size

**Issue:** Repository was 425MB due to committed node_modules

**Cause:** node_modules wasn't in .gitignore

**Solution:** Fixed in commit 16c1ee3a
- Removed node_modules from repo
- Updated .gitignore
- Frontend dependencies must be installed with `npm install`

**Status:** ✅ Fixed

### 3. Colab Setup Failures

**Issue:** Git clone failures or database errors

**Cause:** Network issues or old cached code

**Solution:**
- Notebook now verifies git clone success
- Automatic retry logic
- Clear error messages

**Status:** ✅ Fixed in commits d02581c6, 3ecd1ce3

### 4. Training Session Cleanup

**Issue:** Old "running" sessions blocking new training

**Solution:** Setup script automatically cancels stuck sessions
- Runs during migration step
- Sets status to "failed"
- Resets character status

**Status:** ✅ Automated

### 5. Frontend Dependencies

**Issue:** Frontend requires `npm install` after clone

**Reason:** node_modules no longer in repo (correct behavior)

**Solution:**
```bash
cd charforge-gui/frontend
npm install
```

**Status:** ✅ Working as intended

---

## 📊 Development Status

### Completed Features ✅

- [x] Character-based LoRA training
- [x] Multi-image dataset support
- [x] Direct image generation (no training)
- [x] Video generation (Sora, VEO3, Runway)
- [x] AI-powered captioning
- [x] Modern web interface
- [x] Google Colab deployment
- [x] Database migration system
- [x] Graceful error handling
- [x] Security middleware
- [x] Rate limiting
- [x] Optional authentication
- [x] API documentation
- [x] Repository cleanup

### In Progress 🚧

- [ ] Enhanced model management UI
- [ ] Batch training support
- [ ] Advanced dataset augmentation
- [ ] Training progress visualization
- [ ] Generation queue management

### Planned Features 📋

- [ ] ControlNet support
- [ ] IP-Adapter integration
- [ ] Automated hyperparameter tuning
- [ ] Multi-GPU training
- [ ] Cloud storage backends (S3, GCS)
- [ ] WebSocket for real-time updates
- [ ] Mobile-responsive UI improvements
- [ ] Video-to-video generation
- [ ] Model merging tools

### Documentation Status 📚

- [x] README.md - Main documentation
- [x] SETUP_GUIDE.md - Setup instructions
- [x] COMPREHENSIVE_REVIEW.md - Code review
- [x] REPOSITORY_CONTEXT.md - This document
- [x] ARCHITECTURE.md - System architecture
- [x] QUICKSTART.md - Quick start guide
- [ ] API_REFERENCE.md - API documentation (partial)
- [ ] CONTRIBUTING.md - Contribution guidelines (todo)

---

## 🔗 Key Files Reference

### Configuration
- [.env.example](.env.example) - Environment template
- [.gitignore](.gitignore) - Git ignore patterns
- [requirements-masuka.txt](requirements-masuka.txt) - Python dependencies

### Backend
- [app/main.py](charforge-gui/backend/app/main.py) - FastAPI entry point
- [app/core/database.py](charforge-gui/backend/app/core/database.py) - Database models
- [app/core/config.py](charforge-gui/backend/app/core/config.py) - Configuration
- [app/core/security.py](charforge-gui/backend/app/core/security.py) - Security middleware

### Frontend
- [src/main.ts](charforge-gui/frontend/src/main.ts) - Vue entry point
- [src/router/index.ts](charforge-gui/frontend/src/router/index.ts) - Routes
- [package.json](charforge-gui/frontend/package.json) - Dependencies

### Deployment
- [masuka_colab_setup.py](masuka_colab_setup.py) - Colab setup script
- [ai_gen_colab.ipynb](ai_gen_colab.ipynb) - Colab notebook
- [docker-compose.yml](charforge-gui/docker-compose.yml) - Docker config

### Documentation
- [README.md](README.md) - Main docs
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Setup guide
- [COMPREHENSIVE_REVIEW.md](COMPREHENSIVE_REVIEW.md) - Code review
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture

---

## 📝 Notes for Claude Desktop

### What to Focus On

1. **Recent Database Changes**
   - Migration system was just added
   - error_message column added to training_sessions
   - Migrations run automatically during setup

2. **Repository Cleanup**
   - node_modules removed (just happened)
   - .gitignore updated with proper patterns
   - Frontend requires `npm install` after clone

3. **Colab Setup**
   - Single-cell notebook for easy deployment
   - Automatic migrations and validation
   - Google Drive integration for persistence

4. **API Structure**
   - FastAPI backend with comprehensive endpoints
   - Optional authentication system
   - Security middleware with rate limiting

5. **Frontend Stack**
   - Vue 3 + TypeScript
   - Tailwind CSS for styling
   - Pinia for state management

### Common Tasks You Might Help With

1. **Debugging migration issues**
   - Check [migrations/](charforge-gui/backend/migrations/)
   - Verify database schema
   - Run migrations manually

2. **API endpoint modifications**
   - Endpoints in [app/api/](charforge-gui/backend/app/api/)
   - Models in [app/core/database.py](charforge-gui/backend/app/core/database.py)
   - Use graceful error handling patterns

3. **Frontend component development**
   - Components in [src/components/](charforge-gui/frontend/src/components/)
   - Views in [src/views/](charforge-gui/frontend/src/views/)
   - Follow Vue 3 Composition API patterns

4. **Setup script improvements**
   - Main script: [masuka_colab_setup.py](masuka_colab_setup.py)
   - Notebook: [ai_gen_colab.ipynb](ai_gen_colab.ipynb)
   - Focus on error handling and user experience

5. **Documentation updates**
   - Keep docs in sync with code changes
   - Update examples when APIs change
   - Add migration notes when schema changes

### Architecture Patterns to Follow

1. **Database Migrations**
   - Always idempotent (safe to run multiple times)
   - Check column/table existence before changes
   - Clear success/skip messages

2. **API Error Handling**
   - Use graceful degradation with `getattr()`
   - Return 500 only for critical errors
   - Log errors for debugging

3. **Frontend State**
   - Use Pinia stores for global state
   - Props for component communication
   - Emit events for parent updates

4. **Security**
   - Validate all inputs
   - Use parameterized queries
   - Respect rate limits
   - Honor CORS configuration

---

## 🎯 Quick Command Reference

### Development
```bash
# Backend
cd charforge-gui/backend
uvicorn app.main:app --reload

# Frontend
cd charforge-gui/frontend
npm run dev

# Run migrations
python run_migrations.py

# Run tests (when implemented)
pytest
```

### Git
```bash
# Pull latest
git pull origin main

# Check status
git status

# Commit changes
git add .
git commit -m "Description"
git push origin main
```

### Database
```bash
# Access SQLite database
sqlite3 charforge-gui/backend/database.db

# Check tables
.tables

# Check schema
.schema training_sessions

# Exit
.exit
```

---

**Generated:** 2025-01-14
**By:** Claude Code
**Version:** Repository Context Summary v1.0
