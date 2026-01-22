# 🎨 SDWorks Frontend

A premium, high-performance web interface designed for Stable Diffusion. Built with modern aesthetics and a focus on user experience, SDWorks provides an intuitive way to interact with AI image generation.

## 🚀 Overview

The frontend is a standalone web application built using **Vanilla JavaScript**, **CSS3**, and **HTML5**. It is designed to be lightweight, responsive, and incredibly fast, avoiding the overhead of complex frameworks while maintaining a premium look and feel.

## ✨ Key Features

- **💎 Glassmorphism UI**: High-end aesthetics with real-time blur, glow effects, and smooth animations.
- **🪄 Magic Expand**: AI-powered prompt expansion using a dedicated backend endpoint.
- **🎨 LoRA Stacking**: A specialized modal for searching, adding, and weight-adjusting Multiple LoRAs.
- **📂 Prompt Helper**: A categorized library of subjects, styles, lighting, and moods to help you build better prompts.
- **🖼️ Img2Img Mode**: Full support for image-to-image generation with denoising strength controls.
- **📱 Responsive Layout**: Optimized for desktop, tablet, and mobile screens.

## 🛠️ Technical Stack

- **Logic**: Vanilla JavaScript (Class-based architecture in `app.js`).
- **Styling**: Modern CSS3 with Custom Properties (variables) and Glassmorphism techniques.
- **Typography**: Inter and Space Grotesk via Google Fonts.
- **Icons**: Font Awesome 6.

## ⚙️ Configuration

The frontend connects to the SDWorks backend via a configurable API endpoint.

1. **Default**: Connects to the same host via the Nginx proxy (e.g., `http://localhost:8080/sdapi/`).
2. **Manual Setting**: You can change the API endpoint in the "Settings" tab within the application.

## 📂 Architecture

The application logic resides in [app.js](app.js).
- **StableDiffusionApp**: The core class managing state, event listeners, and API communication.
- **DOM Elements**: Carefully managed to provide real-time updates without page reloads.
- **LocalStorage**: Used to persist user settings, API endpoints, and custom prompt styles.

## 🚀 How to Run

### Via Docker (Recommended)
The frontend is automatically served via Nginx when running the full stack:
```bash
docker-compose up -d
```
Access at: `http://localhost:8080`

### Manual Run
You can simply open `index.html` in any modern browser. You will need to ensure the backend is running and the "API Endpoint" in settings is correctly pointed to your backend (default `http://localhost:7860`).

---

*Part of the SDWorks ecosystem.*
