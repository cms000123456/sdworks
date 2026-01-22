# 🎨 SDWorks

A premium, modern, and standalone web interface for Stable Diffusion. Designed for artists and AI enthusiasts who want a beautiful and frictionless image generation experience without the complexity of traditional UIs.

![SDWorks Banner](info/assets/banner.png)

## ✨ Features

- **💎 Premium Glassmorphism UI**: A stunning, modern interface with real-time effects and smooth animations.
- **🚀 Standalone Python Backend**: Built with FastAPI and `diffusers`, no need for AUTOMATIC1111 or other heavy dependencies.
- **🪄 Magic Expand**: AI-powered prompt enhancement to turn simple ideas into detailed masterpieces.
- **🎨 LoRA Management**: Easy-to-use LoRA integration with search and weight controls.
- **📂 Prompt Library**: Save and load your favorite prompts and styles.
- **🐳 Docker Ready**: Full stack deployment with a single command.
- **📱 Responsive Design**: Works beautifully on desktops, tablets, and phones.

## 💻 Hardware Requirements

To run **SDWorks** locally, the following specifications are recommended:

### Minimum
- **GPU**: NVIDIA GPU with 4GB VRAM (or Apple Silicon Mac)
- **RAM**: 8GB System Memory
- **Storage**: 10GB free space for model weights
- **CPU Mode**: Supported as fallback (slower generation times)

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ System Memory
- **CUDA**: Version 11.8 or 12.1 installed

## 🚀 Quick Start

The easiest way to get started is using Docker.

### 1. Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU with drivers installed (recommended for speed)

### 2. Launch
```bash
git clone https://github.com/cms000123456/sdworks.git
cd sdworks
docker-compose up -d
```

Open `http://localhost:8080` in your browser. The first run will automatically download the necessary models (~4GB).

## 🛠️ Manual Installation

If you prefer to run it manually:

### Backend
```bash
cd backend
pip install -r requirements.txt
python server.py
```

### Frontend
Simply open `index.html` in any modern web browser. It connects to `http://localhost:7860` by default.

## 🧩 Components

- **Frontend**: Vanilla HTML5, CSS3, and JavaScript. No heavy frameworks, just pure performance.
- **Backend**: FastAPI, PyTorch, and Hugging Face `diffusers`.
- **Infrastructure**: Nginx and Docker for seamless serving.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Created with ❤️ for the AI Art Community.*
