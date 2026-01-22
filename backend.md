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

## ⚡ CUDA Installation Guide

If you have an NVIDIA GPU, follow these steps to enable hardware acceleration:

### 1. Identify your GPU
Ensure your GPU supports CUDA (most NVIDIA cards from the last 10 years do). Check your model on the [NVIDIA website](https://developer.nvidia.com/cuda-gpus).

### 2. Install NVIDIA Drivers
Download and install the latest drivers for your specific card: [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx).

### 3. Install CUDA Toolkit
We recommend **CUDA 12.1** for the best compatibility with modern PyTorch:
1. Go to the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
2. Select **CUDA Toolkit 12.1.0**.
3. Choose your OS and follow the installation instructions. 

### 4. Install cuDNN (Optional but Recommended)
For extra performance:
1. Download cuDNN from [NVIDIA Developer](https://developer.nvidia.com/cudnn).
2. Copy the contents of the `bin`, `include`, and `lib` folders into the corresponding directories in your CUDA installation path (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`).

### 5. Verify Installation
Open a terminal and run:
```bash
nvidia-smi
```
This should display your GPU details and the installed CUDA version.

## 🚀 Setup & Installation

### Option 1: Docker (Recommended)
```bash
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

## 🆘 Troubleshooting

### Error: `could not select device driver "nvidia"`
If you see this error when running `docker-compose up`, it means Docker cannot find the NVIDIA driver on your host.

**The Fix: Install NVIDIA Container Toolkit**

For Ubuntu/Debian based systems, run the following commands:
```bash
# 1. Setup the package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install the toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker
sudo systemctl restart docker
```
*For other distributions, see the [official NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).*

### Running in CPU-Only Mode
If you don't have an NVIDIA GPU or can't get the driver working, you can force CPU mode:
1.  Open `docker-compose.yml`.
2.  **Remove** the `deploy:` section (lines 19-25) that reserves the GPU.
3.  Restart the container: `docker-compose up -d`.
*Note: Image generation will be significantly slower.*

### VRAM Out of Memory
If you have 4GB VRAM and experience crashes:
- Ensure no other GPU-heavy apps are running.
- Stick to 512x512 resolution.
- The backend automatically uses attention slicing to save memory.

---

*Part of the SDWorks ecosystem.*
