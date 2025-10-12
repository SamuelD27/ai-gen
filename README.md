# MASUKA - AI Generation Platform

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-red.svg)

**Professional AI Image & Video Generation Platform**

[Features](#features) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Video Generation](#video-generation)

</div>

---

## 🎨 What is MASUKA?

MASUKA is a production-ready AI generation platform that combines cutting-edge image and video generation capabilities with an intuitive web interface. Perfect for artists, content creators, and AI enthusiasts.

### ✨ Key Features

- **🖼️ Image Generation**
  - Flux.1 Dev & Schnell (best-in-class quality)
  - Stable Diffusion 3.5 Large
  - Superior text rendering and composition
  - Perfect hand generation

- **🎬 Video Generation** *(New!)*
  - Sora 2 integration
  - Google VEO3 support
  - Runway Gen-4 API
  - Up to 60 seconds of high-quality video

- **🎯 LoRA Training**
  - Fast training (1000-2000 steps)
  - Automatic dataset curation
  - AI-powered captioning (BLIP-2, Florence-2)
  - 20-30 images for optimal results

- **🎨 Professional UI**
  - Modern, responsive design
  - Real-time progress tracking
  - Drag-and-drop uploads
  - Comprehensive media management

- **☁️ Cloud Ready**
  - Google Colab optimized
  - One-click deployment
  - ngrok tunneling
  - Persistent storage options

---

## 🚀 Quick Start (Google Colab)

### Option 1: One-Line Install

```python
!wget https://raw.githubusercontent.com/SamuelD27/ai-gen/main/masuka_colab_setup.py && python masuka_colab_setup.py
```

### Option 2: Colab Notebook

1. Open our Colab notebook: [MASUKA on Colab](./ai_gen_colab.ipynb)
2. Click "Run all"
3. Access your instance via the ngrok URL

### Option 3: Manual Setup

```bash
# Clone repository
git clone https://github.com/SamuelD27/ai-gen.git
cd ai-gen

# Install dependencies
pip install -r requirements-masuka.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start backend
cd charforge-gui/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd charforge-gui/frontend
npm install
npm run dev
```

---

## 📚 Features Deep Dive

### Image Generation

MASUKA uses the latest diffusion models for photorealistic image generation:

**Supported Models:**
- **Flux.1 Dev** - Best quality, superior text rendering, perfect hands
- **Flux.1 Schnell** - 4x faster generation
- **SD 3.5 Large** - Balanced performance

### Video Generation

Generate professional videos from text prompts:

**Supported APIs:**
- OpenAI Sora 2
- Google VEO3
- Runway Gen-4
- Haiper AI
- Kling AI

### LoRA Training

Train custom models with your own images:

1. **Upload Images** (20-30 recommended)
2. **Create Dataset** with automatic captioning
3. **Configure Training** (or use presets)
4. **Monitor Progress** in real-time
5. **Generate Images** with your LoRA

**Training Tips:**
- Use varied poses and angles
- Consistent lighting helps
- Natural language captions work best
- Training takes ~30-45 minutes on T4 GPU

---

## 🛠️ Configuration

### Environment Variables

```bash
# API Keys (required)
HF_TOKEN=your_huggingface_token
CIVITAI_API_KEY=your_civitai_key
GOOGLE_API_KEY=your_google_key
FAL_KEY=your_fal_key

# Video Generation (optional)
COMET_API_KEY=your_comet_api_key  # For Sora/VEO3/Runway
OPENAI_API_KEY=your_openai_key    # For Sora direct

# Storage (optional)
AWS_ACCESS_KEY=your_aws_key
AWS_SECRET_KEY=your_aws_secret
S3_BUCKET=your_bucket_name
```

---

## 📖 Documentation

- [Modernization Plan](./MASUKA_MODERNIZATION_PLAN.md)
- [Architecture](./ARCHITECTURE.md)
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- [Colab Quick Start](./COLAB_QUICKSTART.md)

---

## 🎯 Roadmap

- [x] Fix dependency conflicts
- [x] Rebrand to MASUKA
- [x] Modernize backend architecture
- [x] Production-ready Colab setup
- [ ] Complete video generation integration
- [ ] Advanced LoRA techniques (Kontext)
- [ ] Batch processing pipeline
- [ ] Model comparison view
- [ ] Mobile responsive UI
- [ ] Comprehensive test suite

See [MASUKA_MODERNIZATION_PLAN.md](./MASUKA_MODERNIZATION_PLAN.md) for full details.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) - Flux.1 models
- [Stability AI](https://stability.ai/) - Stable Diffusion
- [OpenAI](https://openai.com/) - Sora video generation
- [Google DeepMind](https://deepmind.google/) - VEO3
- [Runway](https://runwayml.com/) - Gen-4 video AI

---

## 💬 Support

- 🐛 Issues: [GitHub Issues](https://github.com/SamuelD27/ai-gen/issues)
- 📖 Docs: [Documentation](./docs/)

---

<div align="center">

**Made with ❤️ by the MASUKA team**

</div>
