# SDWorks Backend

A standalone Python backend for Stable Diffusion using FastAPI and the `diffusers` library. No need for AUTOMATIC1111 or other complex setups!

## ✨ Features

- **Simple Setup**: Just Python and pip required
- **FastAPI Server**: Modern, fast API with automatic documentation
- **GPU Accelerated**: Automatic GPU detection and optimization
- **CPU Fallback**: Works on CPU (though slower)
- **CORS Enabled**: Works seamlessly with the frontend
- **AUTOMATIC1111 Compatible**: Uses the same API format

## 📋 System Requirements

### Minimum

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, will use CPU otherwise)
- **RAM**: 8GB system memory
- **Storage**: 10GB free space for model files

### Recommended

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system memory
- **CUDA**: CUDA 11.8 or 12.1 for GPU acceleration
- **Docker**: For containerized deployment (optional)

## 🐳 Docker Deployment (Recommended)

The easiest way to run the backend is with Docker! No manual Python setup required.

### Prerequisites

- **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop) for Windows
- **NVIDIA Container Toolkit**: For GPU support (see [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Quick Start with Docker Compose

```bash
# Navigate to backend directory
cd C:\Users\cms\Documents\Antigravity\SDWorks\backend

# Build and start the container
docker-compose up -d
```

That's it!

- **Frontend**: Open `http://localhost:8080`
- **Backend**: Running on `http://localhost:7860`

The first run downloads the Stable Diffusion model (~4GB).

**Note for Frontend**: The frontend is served via Nginx on port 8080. It will connect to the backend on port 7860 by default (CORS is enabled).

### Docker Commands

```bash
# Stop the container
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f

# Access container shell
docker exec -it sd-backend bash
```

### GPU Support

The Docker setup automatically uses your NVIDIA GPU. Ensure:

1. NVIDIA drivers are installed
2. NVIDIA Container Toolkit is installed
3. Docker Desktop has GPU support enabled

### Persistent Model Cache

Models are cached in a Docker volume (`huggingface-cache`) so you don't need to re-download on container restarts.

---

## 🚀 Manual Setup (Alternative)

If you prefer not to use Docker, you can run the backend directly with Python:

### 1. Install Python Dependencies

```bash
# Navigate to backend directory
cd C:\Users\cms\Documents\Antigravity\SDWorks\backend

# Install requirements
pip install -r requirements.txt
```

**Note**: First installation will take several minutes as it downloads PyTorch and other dependencies.

### 2. Start the Server

```bash
python server.py
```

**First Run**: The server will automatically download the Stable Diffusion model (~4GB) from HuggingFace. This is a one-time download and may take 5-15 minutes depending on your internet speed.

You'll see output like:

```
INFO:     Loading Stable Diffusion model...
INFO:     Using device: cuda with dtype: torch.float16
INFO:     Model loaded successfully on cuda
INFO:     Server ready!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860
```

### 3. Use with Frontend

The server runs on `http://localhost:7860` by default, which matches the frontend's default configuration. Just open the frontend and start generating!

## 🔧 Configuration

### Change Port

Edit `server.py` at the bottom:

```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8080,  # Change to your preferred port
    log_level="info"
)
```

### Use Different Model

Edit `server.py` line 56:

```python
model_id = "stabilityai/stable-diffusion-2-1"  # or any HuggingFace model
```

Popular models:

- `runwayml/stable-diffusion-v1-5` (default, fastest)
- `stabilityai/stable-diffusion-2-1` (better quality)
- `CompVis/stable-diffusion-v1-4` (original)

### CPU-Only Mode

The server automatically uses CPU if no GPU is detected. For explicit CPU mode:

```python
device = "cpu"  # Force CPU in load_model() function
```

## 📊 Performance

**With GPU (RTX 3060, 12GB VRAM)**:

- 512×512, 20 steps: ~2-3 seconds
- 768×768, 30 steps: ~5-7 seconds

**With CPU (i7-10700K)**:

- 512×512, 20 steps: ~60-90 seconds
- Not recommended for regular use

## 🐛 Troubleshooting

### Docker Issues

**Container won't start**:

- Check Docker is running: `docker ps`
- Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Check logs: `docker-compose logs stable-diffusion`

**GPU not detected in container**:

- Ensure NVIDIA drivers are installed on host
- Install NVIDIA Container Toolkit
- Verify Docker Desktop GPU support is enabled (Settings → Resources → WSL Integration)

**Model download fails**:

- Check internet connection
- Increase Docker memory limit (Settings → Resources)
- Clear volume and retry: `docker-compose down -v && docker-compose up -d`

### Manual Setup Issues

### "CUDA out of memory"

- Reduce image dimensions (try 512×512)
- Reduce batch size to 1
- Close other GPU-intensive applications
- Enable attention slicing (already automatic)

### "Model not found" or download errors

- Check your internet connection
- Try downloading manually: `python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"`
- Models are cached in: `C:\Users\<username>\.cache\huggingface\`

### Slow generation

- First generation is always slower (model loading)
- CPU mode is 20-30x slower than GPU
- Consider upgrading to GPU or using cloud services

### Import errors

```bash
# Reinstall dependencies
pip uninstall torch torchvision diffusers transformers -y
pip install -r requirements.txt
```

## 🔍 API Documentation

Once the server is running, visit:

- **Interactive Docs**: <http://localhost:7860/docs>
- **OpenAPI Schema**: <http://localhost:7860/openapi.json>

### Example API Call

```bash
curl -X POST "http://localhost:7860/sdapi/v1/txt2img" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "steps": 20,
    "cfg_scale": 7.0,
    "width": 512,
    "height": 512
  }'
```

## 💡 Tips

1. **First Run**: Be patient during model download
2. **GPU Memory**: Start with 512×512 images
3. **Quality vs Speed**: 20-30 steps is a good balance
4. **Prompting**: Be descriptive! "A majestic dragon..." works better than "dragon"
5. **Negative Prompts**: Use to avoid common issues like "blurry, low quality, distorted"

## 📦 Model Storage

Models are stored in HuggingFace cache:

- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`
- **Linux/Mac**: `~/.cache/huggingface/hub/`

To free up space, you can delete old models from this directory.

## 🆘 Getting Help

If you encounter issues:

1. Check the console logs for error messages
2. Verify GPU drivers are up to date (for NVIDIA: [nvidia.com/drivers](https://www.nvidia.com/drivers))
3. Ensure Python 3.10+ is installed: `python --version`
4. Check disk space for model downloads

## 🎉 You're Ready

Start the backend, open the frontend, and start creating amazing AI art!
