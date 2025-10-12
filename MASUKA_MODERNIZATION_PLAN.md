# MASUKA - Complete Modernization Plan

## Executive Summary

Transform "CharForge" into "MASUKA" - a production-ready, bulletproof AI generation platform with:
- Latest AI models (Flux.1, Stable Diffusion 3.5)
- Video generation (Sora 2, VEO3, Runway Gen-4 APIs)
- Zero dependency conflicts
- Modern, polished UI
- Robust error handling
- Google Colab optimized

---

## Phase 1: Foundation & Cleanup (Priority: CRITICAL)

### 1.1 Dependency Management Overhaul
**Problem**: Pillow version conflicts, outdated packages, conflicting requirements

**Solution**:
```python
# New unified requirements.txt with pinned versions
torch==2.5.0
pillow==11.0.0  # Match latest PyTorch requirements
diffusers==0.33.0  # Latest with Flux.1 support
transformers==4.48.0
```

**Actions**:
- [ ] Create single source of truth for all dependencies
- [ ] Use Poetry or pip-tools for deterministic builds
- [ ] Add dependency conflict checking to setup script
- [ ] Test on fresh Colab environment

### 1.2 Fix Colab Setup Script
**Problem**: Import errors, cached versions, fragile installation

**Solution**:
- Virtual environment isolation
- Proper Python package management
- Automatic error recovery
- Version pinning

---

## Phase 2: Model Upgrades (Priority: HIGH)

### 2.1 Image Generation Models

#### Current: Mixed SDXL/Flux
#### Target: Flux.1 Dev + SD3.5 Large

**Flux.1 Advantages**:
- Superior text rendering
- Better prompt adherence
- Perfect hand generation
- Complex composition accuracy

**Implementation**:
```python
# models/image_generation.py
SUPPORTED_MODELS = {
    "flux1-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "type": "flux",
        "resolution": 1024,
        "vram": "24GB",
        "speed": "medium"
    },
    "flux1-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "type": "flux",
        "resolution": 1024,
        "vram": "16GB",
        "speed": "fast"
    },
    "sd35-large": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "type": "sd3",
        "resolution": 1024,
        "vram": "20GB",
        "speed": "fast"
    }
}
```

### 2.2 LoRA Training Optimization

**Latest Techniques (2025)**:
- Kohya SS with Flux support
- 1000-2000 steps (vs 4000+ for SDXL)
- 20-30 images for best results
- Natural language captions (not tags)
- 1024x1024 training resolution

**Platforms Integration**:
- Fal.ai API (10x faster training)
- Local Kohya training (full control)
- Replicate.com fallback

### 2.3 Video Generation (NEW!)

**API Integration Priority**:
1. **CometAPI** (unified access to Sora 2, VEO3, Runway)
   - sora-2-hd, sora-2
   - veo3-pro, veo3-fast
   - runway-gen4

2. **Fallback Options**:
   - Haiper AI (fastest generation)
   - Kling AI (best control)

**Features to Add**:
```python
# New module: masuka/video/generator.py
class VideoGenerator:
    def generate(
        self,
        prompt: str,
        duration: int = 5,  # seconds
        resolution: str = "1080p",
        model: str = "veo3-pro",
        motion_intensity: float = 0.7
    ) -> str:
        """Generate video from text prompt"""
```

---

## Phase 3: Backend Modernization (Priority: HIGH)

### 3.1 FastAPI Structure

**Current Issues**:
- Generic error messages
- No request validation
- Weak CORS handling
- Missing rate limiting

**New Structure**:
```
masuka-backend/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings v2
│   │   ├── security.py         # JWT, rate limiting
│   │   ├── dependencies.py     # Dependency injection
│   │   └── exceptions.py       # Custom exceptions
│   ├── models/
│   │   ├── image.py           # SQLAlchemy models
│   │   ├── video.py
│   │   ├── dataset.py
│   │   └── training.py
│   ├── schemas/
│   │   ├── image.py           # Pydantic schemas
│   │   ├── video.py
│   │   └── requests.py
│   ├── api/
│   │   ├── v1/
│   │   │   ├── image_gen.py
│   │   │   ├── video_gen.py
│   │   │   ├── training.py
│   │   │   └── media.py
│   │   └── router.py
│   ├── services/
│   │   ├── image_service.py
│   │   ├── video_service.py
│   │   ├── training_service.py
│   │   └── storage_service.py
│   └── main.py
├── alembic/                   # Database migrations
├── tests/
└── requirements.txt
```

### 3.2 Error Handling & Logging

**Structured Logging**:
```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    logger.info("request_started",
                path=request.url.path,
                method=request.method)
    response = await call_next(request)
    logger.info("request_completed",
                status_code=response.status_code)
    return response
```

**Custom Exception Handlers**:
```python
class MasukaException(Exception):
    """Base exception with error codes"""

class ModelNotFoundError(MasukaException):
    code = "MODEL_NOT_FOUND"

class GenerationFailedError(MasukaException):
    code = "GENERATION_FAILED"
```

---

## Phase 4: Frontend Modernization (Priority: HIGH)

### 4.1 Rebrand to MASUKA

**Changes Required**:
- [ ] Update all "CharForge" → "MASUKA" in code
- [ ] New logo/branding
- [ ] Update page titles, meta tags
- [ ] New color scheme

### 4.2 Modern UI Framework

**Current**: Vue 3 + Tailwind (good foundation)

**Enhancements**:
- shadcn-vue components (already partially integrated)
- Proper dark/light theme
- Responsive design (mobile-first)
- Loading skeletons
- Optimistic updates

### 4.3 New Views

**Video Generation Tab** (NEW):
```vue
<!-- src/views/VideoGeneration.vue -->
<template>
  <div class="video-gen-container">
    <PromptInput
      v-model="prompt"
      placeholder="Describe the video you want to create..."
    />

    <SettingsPanel>
      <ModelSelector :models="videoModels" />
      <DurationSlider min="3" max="60" />
      <ResolutionSelect />
      <MotionIntensity />
    </SettingsPanel>

    <VideoPreview :src="generatedVideo" />
    <DownloadButton />
  </div>
</template>
```

**Dashboard Improvements**:
- Real-time generation progress
- Queue management
- Cost estimation
- Generation history
- Model comparison view

### 4.4 UX Enhancements

**File Upload**:
- Drag & drop anywhere
- Multiple file selection
- Progress bars per file
- Image preview grid
- Batch operations

**Error Display**:
- Toast notifications (already have vue-toastification)
- Error recovery suggestions
- Retry with exponential backoff

---

## Phase 5: New Features (Priority: MEDIUM)

### 5.1 Video Generation System

**Architecture**:
```
User Input → Queue System → API Router → Video Service → External API
                ↓
         Progress Updates (WebSocket)
                ↓
         Storage Service → CDN/Local
```

**Implementation Steps**:
1. Add CometAPI integration
2. Create video processing queue (Celery or Arq)
3. WebSocket progress updates
4. Video preview/player component
5. Download management

### 5.2 Advanced LoRA Training

**Features**:
- Automatic dataset curation
- Quality scoring (CLIP/aesthetic)
- Auto-captioning (BLIP-2, Florence-2)
- Training parameter optimization
- A/B testing different LoRAs

### 5.3 Batch Processing

**Capabilities**:
- Multiple prompt generation
- Parameter grid search
- Style mixing
- Upscaling pipeline
- Export to video (image sequence)

---

## Phase 6: Infrastructure (Priority: MEDIUM)

### 6.1 Colab Optimization

**New Setup Script Features**:
```python
# colab_setup_v2.py
class MasukaSetup:
    def __init__(self):
        self.check_gpu()
        self.create_venv()
        self.install_dependencies()
        self.download_models()
        self.start_services()

    def create_venv(self):
        """Isolated environment prevents conflicts"""
        subprocess.run([
            "python", "-m", "venv", "/content/masuka-env"
        ])

    def verify_installation(self):
        """Comprehensive health checks"""
        self.test_torch_cuda()
        self.test_models_loaded()
        self.test_api_endpoints()
        self.test_upload_flow()
```

**Features**:
- Progress bar for each step
- Automatic error recovery
- Model caching (persist across sessions)
- ngrok tunnel with custom domain
- Auto-restart on crash

### 6.2 Performance Optimization

**Model Loading**:
- Lazy loading (only load when needed)
- Model offloading (CPU ↔ GPU)
- Shared memory for multi-user

**Generation Speed**:
- xFormers memory-efficient attention
- torch.compile() for faster inference
- Batched generation
- Result caching

### 6.3 Storage Management

**Current**: Local filesystem
**Target**: Flexible storage backend

```python
# storage/backends.py
class StorageBackend(ABC):
    @abstractmethod
    def save(self, file: bytes, path: str) -> str: ...

class LocalStorage(StorageBackend): ...
class S3Storage(StorageBackend): ...
class GoogleCloudStorage(StorageBackend): ...
```

---

## Phase 7: Quality Assurance (Priority: HIGH)

### 7.1 Testing Strategy

**Backend Tests**:
```python
# tests/test_image_generation.py
def test_flux_generation():
    result = image_service.generate(
        prompt="a red apple on a table",
        model="flux1-dev"
    )
    assert result.width == 1024
    assert result.height == 1024
    assert "error" not in result
```

**Frontend Tests**:
- Component tests (Vitest)
- E2E tests (Playwright)
- Visual regression tests

**Integration Tests**:
- Full upload → process → download flow
- Training pipeline end-to-end
- API rate limiting

### 7.2 Error Resilience

**Retry Logic**:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def generate_with_retry(prompt: str):
    return await api_client.generate(prompt)
```

**Circuit Breakers**:
- Prevent cascade failures
- Fallback to alternative models
- Graceful degradation

### 7.3 Monitoring

**Metrics to Track**:
- Generation success rate
- Average generation time
- Error rates by type
- API quota usage
- User activity

**Tools**:
- Prometheus metrics
- Grafana dashboards
- Sentry error tracking

---

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Fix dependency conflicts
- [ ] Create new requirements.txt
- [ ] Rebuild Colab setup script
- [ ] Test on fresh environment

### Week 3-4: Core Features
- [ ] Integrate Flux.1 Dev
- [ ] Modernize backend structure
- [ ] Rebrand to MASUKA
- [ ] Fix all existing bugs

### Week 5-6: Video Generation
- [ ] CometAPI integration
- [ ] Video generation UI
- [ ] Queue system
- [ ] Progress tracking

### Week 7-8: Polish
- [ ] UI/UX improvements
- [ ] Testing suite
- [ ] Documentation
- [ ] Performance optimization

---

## Success Metrics

✅ **Zero dependency conflicts**
✅ **100% upload success rate**
✅ **< 5 second cold start time**
✅ **Video generation working**
✅ **All tests passing**
✅ **Clean, modern UI**
✅ **Comprehensive documentation**

---

## Technology Stack (Updated)

### Backend
- FastAPI 0.115.0
- Pydantic v2
- SQLAlchemy 2.0
- Alembic (migrations)
- Celery + Redis (task queue)
- Pillow 11.0.0 (compatible)

### AI/ML
- PyTorch 2.5.0
- Diffusers 0.33.0
- Transformers 4.48.0
- Flux.1 Dev/Schnell
- SD 3.5 Large
- CometAPI (video)

### Frontend
- Vue 3.5
- TypeScript 5.3
- Vite 7
- Tailwind CSS 3.4
- shadcn-vue
- Pinia (state)

### Infrastructure
- Google Colab (T4/A100 GPU)
- ngrok (tunnel)
- Optional: RunPod, Modal

---

## Next Steps

1. **Approve this plan** - Review and confirm approach
2. **Prioritize phases** - Which to tackle first?
3. **Begin implementation** - Start with Phase 1

Ready to build MASUKA - the bulletproof AI generation platform! 🚀
