# ⚙️ SDWorks Backend

The high-performance heart of SDWorks. A standalone Python server built with **FastAPI** and the Hugging Face **Diffusers** library, providing a robust and scalable API for Stable Diffusion.

## 🚀 Overview

The backend acts as the bridge between the UI and the underlying AI models. It is optimized for NVIDIA GPUs but includes seamless fallbacks for Apple Silicon and CPU-only environments.

## ✨ Key Features

- **⚡ FastAPI Architecture**: High-concurrency support with automatic OpenAPI documentation.
- **🎯 Diffusers Optimized**: Uses the latest optimizations (fp16 precision, attention slicing) for maximum speed.
- **🔗 A1111 Compatibility**: Implements a standard `/sdapi/v1/` interface, making it compatible with many existing tools.
- **🛠️ Flexible Model Support**: Easily switch between SD v1.5, v2.1, or custom safetensors models.
- **🐳 Containerized**: Docker-ready with NVIDIA GPU support out of the box.

## 🛠️ Technical Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI / Uvicorn
- **AI Engine**: PyTorch & Hugging Face Diffusers
- **Imaging**: PIL (Pillow)

## 📋 Requirements

### Hardware
- **NVIDIA GPU**: 4GB+ VRAM (Recommended: 8GB+)
- **System RAM**: 8GB+ (Recommended: 16GB+)
- **Storage**: ~10GB for models and environment

### Software
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Python**: 3.10+

## 🚀 Setup & Installation

### Option 1: Docker (Recommended)
```bash
cd backend
docker-compose up -d
```
The Docker setup handles all dependencies and environment configurations automatically.

### Option 2: Manual Installation
1. Navigate to the `backend` directory.
2. Install requirements: `pip install -r requirements.txt`.
3. Start the server: `python server.py`.

## 🔍 API Documentation

Once the server is running, you can explore the interactive API documentation at:
- **Swagger UI**: `http://localhost:7860/docs`
- **ReDoc**: `http://localhost:7860/redoc`

## 🧩 Model Management

Models are automatically downloaded from Hugging Face on first run. You can configure model paths and specific model IDs in `server.py` and `docker-compose.yml`.

---

*Part of the SDWorks ecosystem.*
