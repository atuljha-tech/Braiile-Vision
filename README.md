# 🔵 Braille Vision — AI Accessibility Scanner

> **Turning Touch Into Voice** — Real-time Braille OCR that reads physical Braille pages and speaks them aloud.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com)
[![Groq](https://img.shields.io/badge/Groq-AI-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Problem Statement
Over 253 million people worldwide live with visual impairment. Physical Braille books exist, but most people cannot read Braille. **Braille Vision** bridges this gap — point a phone camera at any Braille page and hear it read aloud instantly.

## ✨ Key Features
- 📷 **Real-time camera scanning** — live Braille detection at 8 FPS
- 🧠 **Dual AI pipeline** — CNN classifier + geometric dot decoder
- 🔊 **Instant TTS** — speaks detected text immediately
- 🤖 **Groq AI correction** — LLaMA 3.1 fixes OCR errors in context
- 📁 **Image upload** — process any Braille photo
- 🌐 **Browser-based** — no app install needed
- 📱 **Mobile responsive** — works on phones

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser UI (Flask)                 │
│  Camera Feed → Frame Capture → API → TTS Output     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              Detection Pipeline                      │
│  ┌─────────────┐    ┌──────────────┐                │
│  │ CV Detector │    │  CNN Model   │                │
│  │ (OpenCV)    │───▶│  (PyTorch)   │                │
│  │ Blob detect │    │  26 classes  │                │
│  │ Cell cluster│    │  100% acc    │                │
│  └─────────────┘    └──────┬───────┘                │
│                            │                        │
│                   ┌────────▼────────┐               │
│                   │  Groq LLaMA 3.1 │               │
│                   │  OCR Correction │               │
│                   └────────┬────────┘               │
│                            │                        │
│                   ┌────────▼────────┐               │
│                   │   TTS Engine    │               │
│                   │  (pyttsx3/say)  │               │
│                   └─────────────────┘               │
└─────────────────────────────────────────────────────┘
```

## 🤖 Model Details

### CNN Architecture
```
Input: 64×64 grayscale Braille cell image
  ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool2d(2)
  ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2)
  ↓
Flatten → Linear(16384→128) + ReLU
  ↓
Linear(128→26) → Softmax
  ↓
Output: Letter A-Z (26 classes)
```

### Dataset
- **1,560 synthetic images** (26 letters × 60 variants each)
- Augmentations: rotation ±8°, Gaussian blur, position jitter ±4px
- Generated with `synthetic_dataset/scripts/generate_dataset.py`
- Training: 5 epochs, Adam optimizer, CrossEntropyLoss
- **Final accuracy: 100% on test set**

### Detection Pipeline
1. **Blob detection** — adaptive threshold + Otsu + morphological ops
2. **Light-background pass** — CLAHE + morphological gradient for real embossed Braille
3. **Size filtering** — keep dots within 70% of median radius
4. **CNN fast path** — for small single-cell images (≤200×200px)
5. **Geometric decoder** — cluster dots into cells, decode bit mask
6. **Groq correction** — LLaMA 3.1 fixes context errors

## 🚀 Quick Start

```bash
git clone https://github.com/atuljha-tech/Braiile-Vision
cd Braiile-Vision
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
python3 app.py
# Open http://localhost:5050
```

## 📋 Requirements
```
Python 3.9+, Flask, PyTorch, OpenCV, Groq SDK, pyttsx3
```
Full list: `requirements.txt`

## 🔑 Environment Variables
```
GROQ_API_KEY=your_key_here   # Free at console.groq.com
```

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload` | POST | Process image file |
| `/api/process_frame` | POST | Process webcam frame |
| `/api/status` | GET | Scanner status |
| `/api/history` | GET | Recognition history |
| `/api/speak` | POST | TTS speak text |
| `/api/health` | GET | Health check |

## 🎮 Usage

### Camera Mode
1. Open http://localhost:5050
2. Click **Allow Camera**
3. Hold Braille page in front of camera
4. Text detected and spoken automatically

### Upload Mode
1. Click **Upload Image** or drag-drop
2. Supports JPEG, PNG
3. Works with photos of real Braille books

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `R` | Repeat last |
| `H` | Read history |
| `U` | Upload |

## 📁 Project Structure
```
Braiile-Vision/
├── app.py                          # Flask server
├── requirements.txt
├── .env.example
├── braille_ocr/
│   └── realtime/
│       ├── braille_detector.py     # Core CV + decode
│       ├── camera_loop.py          # Frame processing
│       ├── frame_analyzer.py       # Quality analysis
│       ├── session.py              # State management
│       └── tts_engine.py           # Text-to-speech
├── braille_ai/
│   ├── cnn_predictor.py            # CNN inference
│   └── ocr_corrector.py            # Groq correction
├── synthetic_dataset/
│   ├── models/braille_cnn.pth      # Trained weights
│   ├── generated/                  # 1,560 training images
│   └── scripts/
│       ├── generate_dataset.py     # Dataset generation
│       └── train_model.py          # Model training
├── templates/index.html            # Web UI
├── docs/sample_output.png          # Sample output screenshot
└── static/
    ├── css/style.css
    └── js/app.js
```

## 🏆 Hackathon Highlights
- ✅ Physical Braille recognition (real embossed dots)
- ✅ Real-world robustness (lighting, angle, blur handling)
- ✅ Complete ML pipeline (data → train → inference)
- ✅ Reproducible (all code + weights included)
- ✅ Accessibility-first design

## 📄 License
MIT — built for accessibility, free for all.
