# Braille Vision — AI Accessibility Scanner

> **Real-time Braille OCR with CNN recognition, live camera scanning, and Text-to-Speech.**  
> Built for accessibility. Powered by a **custom synthetic dataset**, PyTorch, and Groq LLaMA.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?style=flat-square)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-green?style=flat-square)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

**Repository:** [github.com/atuljha-tech/Braiile-Vision](https://github.com/atuljha-tech/Braiile-Vision)

---

## Dedication & synthetic dataset

This project is the result of hands-on work on **Braille accessibility**: building a pipeline that turns camera frames and uploads into readable, speakable text.

The CNN at its core was trained on a **synthetic Braille cell dataset** created for this project:

| | |
|---|---|
| **Images** | 1,560 (26 letters × 60 variants) |
| **Augmentation** | Position jitter, rotation, Gaussian blur |
| **Generator** | `synthetic_dataset/scripts/generate_dataset.py` |
| **Training** | `synthetic_dataset/scripts/train_model.py` |
| **Weights** | `synthetic_dataset/models/braille_cnn.pth` (~8 MB) |

That dataset and training run are what make single-cell recognition reliable; the geometric decoder and Groq correction extend that to real photos and noisy OCR.

---

## Screenshots

| Main UI | Scanning & results |
|---------|-------------------|
| ![Braille Vision home](public/ss1.png) | ![Braille Vision detection](public/ss2.png) |

- **GitHub (repo):** [ss1.png](public/ss1.png) · [ss2.png](public/ss2.png)  
- **Local (server running):** [http://localhost:5050/public/ss1.png](http://localhost:5050/public/ss1.png) · [http://localhost:5050/public/ss2.png](http://localhost:5050/public/ss2.png)

> Replace `public/ss1.png` and `public/ss2.png` with your own screenshots (see [Capture your own screenshots](#capture-your-own-screenshots) below).

---

## What it does

1. **Detects** Braille dots (adaptive CV + contour analysis)
2. **Classifies** cells with the trained CNN (A–Z)
3. **Corrects** noisy output with Groq LLaMA 3.1 (optional)
4. **Speaks** results via TTS

---

## Features

| Feature | Details |
|---------|---------|
| Live camera | Browser `getUserMedia` |
| Image upload | Drag-and-drop or file picker |
| CNN | PyTorch, synthetic dataset (1,560 images) |
| Groq AI | LLaMA 3.1-8b-instant correction |
| TTS | pyttsx3 (+ macOS `say` fallback) |
| UI | Dark/light theme, confidence, history |
| Keyboard | Space, R, H, C, U, D shortcuts |

---

## Run locally

### Prerequisites

- **Python 3.9+** (3.11 recommended)
- macOS / Linux / Windows
- Webcam optional (upload works without it)
- **Groq API key** optional ([console.groq.com](https://console.groq.com))

### Steps

```bash
# 1. Clone
git clone https://github.com/atuljha-tech/Braiile-Vision.git
cd Braiile-Vision

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment (optional)
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_...

# 5. Start server
python3 app.py
```

Open **http://localhost:5050** in Chrome or Edge (best camera support).

### Quick health check

```bash
curl http://localhost:5050/api/health
# → {"status":"ok","cnn":true,"groq":true|false,...}
```

### Capture your own screenshots

1. Start the server (`python3 app.py`).
2. Open http://localhost:5050.
3. **ss1.png** — full home view (header, camera/upload, status chips).  
   macOS: `Cmd + Shift + 4` → drag a region, then save as `public/ss1.png`.  
   Windows: `Win + Shift + S`.  
   Or browser DevTools → device toolbar → screenshot.
4. **ss2.png** — upload a Braille image or use the camera; capture when text and confidence are visible → save as `public/ss2.png`.
5. Commit and push:

```bash
git add public/ss1.png public/ss2.png
git commit -m "Update README screenshots"
git push
```

---

## Deploy

This app is a **single Flask service** (HTML + API + PyTorch). **Render** is the recommended host. **Vercel** does not run PyTorch/OpenCV well; use Vercel only if you add a separate static frontend that calls your Render API.

### A. Render (recommended — full app)

1. Push this repo to GitHub.
2. [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect **Braiile-Vision** (or this repo).
4. Settings:

| Field | Value |
|-------|--------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| **Plan** | Free (cold starts; upgrade if needed) |

5. **Environment variables** — paste in Render → **Environment**:

```env
PYTHON_VERSION=3.11.9
GROQ_API_KEY=gsk_your_groq_key_here
```

| Variable | Required | Notes |
|----------|----------|--------|
| `GROQ_API_KEY` | No | OCR correction; app runs without it |
| `PYTHON_VERSION` | Yes (Render) | Use `3.11.9` |
| `PORT` | Auto | Set by Render — do not override |

6. **Deploy**. Open `https://<your-service>.onrender.com`.

Optional: import [`render.yaml`](render.yaml) via **New → Blueprint** for one-click setup.

**Notes**

- First deploy may take several minutes (PyTorch install).
- Free tier sleeps after inactivity; first request can be slow.
- TTS on Render may be limited; browser speech still works for many flows.

---

### B. Vercel (frontend-only / API proxy — not the full backend)

Vercel’s serverless runtime is **not suitable** for this repo’s PyTorch + OpenCV stack. Typical setup:

1. Deploy the **Flask app on Render** (section A).
2. If you later add a React/Next landing page on Vercel, set:

```env
NEXT_PUBLIC_API_URL=https://<your-render-service>.onrender.com
```

(or `VITE_API_URL` for Vite) and point all `fetch('/api/...')` calls to that base URL.

**Do not** paste `GROQ_API_KEY` into Vercel for a static site — keep secrets on **Render** only.

| Vercel (static site) | Render (backend) |
|----------------------|------------------|
| `NEXT_PUBLIC_API_URL` | `GROQ_API_KEY` |
| (no secrets) | `PYTHON_VERSION=3.11.9` |

---

## Model & dataset

### Architecture

```
64×64 grayscale cell
  → Conv2d(1→32) + ReLU + MaxPool
  → Conv2d(32→64) + ReLU + MaxPool
  → Linear → 26 classes (A–Z)
```

### Reproduce training

```bash
python3 synthetic_dataset/scripts/generate_dataset.py
python3 synthetic_dataset/scripts/train_model.py
```

---

## Project structure

```
├── app.py                    # Flask server
├── Procfile                  # Render / Heroku start
├── render.yaml               # Render blueprint
├── public/ss1.png, ss2.png   # README screenshots
├── braille_ocr/realtime/     # Camera, TTS, detection
├── braille_ai/               # CNN + Groq correction
├── synthetic_dataset/        # Dataset, scripts, model
├── templates/ + static/      # Web UI
└── requirements.txt
```

---

## API (summary)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check |
| GET | `/api/status` | Scanner status |
| POST | `/api/upload` | Image upload |
| POST | `/api/process_frame` | Webcam frame (base64) |
| POST | `/api/speak` | TTS |

Full list: see earlier docs in repo or hit `/api/health` after deploy.

---

## Environment variables (local `.env`)

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=5050
```

`PORT` is optional locally (defaults to `5050`). On Render, use Render’s `PORT` only.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| R | Repeat last text |
| H | Read history |
| C | Copy last text |
| U | Upload |
| D | Demo |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with care for accessibility. Every person deserves to read.*
