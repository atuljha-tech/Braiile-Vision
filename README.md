# 🔵 Braille Vision — AI Accessibility Scanner

> **Real-time Braille OCR · PyTorch CNN · TensorFlow CNN · YOLOv8 · Groq LLaMA · Text-to-Speech**  
> Built for accessibility. Every model trained from scratch on a **hand-crafted synthetic dataset**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-8A2BE2?style=flat-square)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-2.3-green?style=flat-square)](https://flask.palletsprojects.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-red?style=flat-square)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

**Repository:** [github.com/atuljha-tech/Braiile-Vision](https://github.com/atuljha-tech/Braiile-Vision)

---

## 🇮🇳 Jai Hind — Dedicated to India & the Visually Impaired

> *"जय हिन्द"* — Dedicated to **India**, to every **visually impaired person** who deserves equal access to the written word, and to the spirit of **ScioBraille** — science in service of humanity.

The Braille text below reads:

> **"Jai Hind India ScioBraille Visually Impaired Great Project"**

This is real embossed Braille — the exact kind of input this system is built to read, decode, and speak aloud.

![Real Braille: Jai Hind India ScioBraille Visually Impaired Great Project](docs/sample_output.png)

---

## 🤖 Three AI Models, One Pipeline

This is not a wrapper around an existing library. Every model was **trained from scratch** on a custom synthetic dataset built specifically for this project.

| Model | Role | Framework | Weights |
|-------|------|-----------|---------|
| **PyTorch CNN** | Braille cell classification (A–Z) | PyTorch 2.1 | `braille_cnn.pth` (8 MB) |
| **TensorFlow CNN** | Braille cell classification (A–Z) — ensemble partner | TensorFlow 2.15 | `braille_cnn_tf.keras` (24 MB) |
| **YOLOv8n** | Braille dot detection — finds every dot in the image | Ultralytics | `braille_dots_yolov8n.pt` (18 MB) |

All three run together in a single inference pass:

```
Input image
    │
    ▼
YOLOv8n ──────────────────────────────► Dot bounding boxes
    +                                         │
OpenCV adaptive threshold ──────────► Merged dot list
                                              │
                                    Cell segmentation (geometry)
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                     ▼
                              PyTorch CNN          TensorFlow CNN
                              (A–Z, 26 cls)        (A–Z, 26 cls)
                                    └─────────┬──────────┘
                                         Ensemble vote
                                              │
                                    Groq LLaMA 3.1 correction
                                              │
                                         Final text + TTS
```

When both CNNs agree → confidence is boosted. When they disagree → the higher-confidence prediction wins.

---

## 📊 Synthetic Dataset

Every model was trained on the same dataset, generated from scratch:

| Property | Value |
|----------|-------|
| **Total images** | 1,560 |
| **Classes** | 26 (A–Z Braille cells) |
| **Variants per class** | 60 |
| **Augmentation** | Position jitter · Rotation ±15° · Gaussian blur · Brightness variation |
| **YOLO annotations** | 400 full-page images (320 train / 80 val) with per-dot bounding boxes |
| **Generator** | `synthetic_dataset/scripts/generate_dataset.py` |
| **YOLO generator** | `synthetic_dataset/scripts/generate_yolo_dataset.py` |

### Reproduce training

```bash
# 1. Generate cell images (1,560 PNG files)
python3 synthetic_dataset/scripts/generate_dataset.py

# 2. Train PyTorch CNN
python3 synthetic_dataset/scripts/train_model.py
# → synthetic_dataset/models/braille_cnn.pth

# 3. Train TensorFlow CNN
python3 synthetic_dataset/scripts/train_model_tf.py
# → synthetic_dataset/models/braille_cnn_tf.keras

# 4. Generate YOLO dot-detection dataset (400 page images + YOLO labels)
python3 synthetic_dataset/scripts/generate_yolo_dataset.py

# 5. Train YOLOv8n dot detector
python3 synthetic_dataset/scripts/train_yolo.py
# → synthetic_dataset/models/braille_dots_yolov8n.pt
```

---

## 📸 Screenshots

| Light theme — image upload | Dark theme — full UI |
|:---:|:---:|
| ![Light theme upload](public/ss1.png) | ![Dark theme UI](public/ss2.png) |
| Upload a Braille image → dots detected → text decoded → spoken aloud | Dark mode with status chips, confidence meter, and recognition history |

**Direct links:** [ss1.png](public/ss1.png) · [ss2.png](public/ss2.png)

---

## 🚀 Run Locally

### Prerequisites

- Python 3.9+ (3.11 recommended)
- macOS / Linux / Windows
- Webcam optional — upload works without it
- Groq API key optional → [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone
git clone https://github.com/atuljha-tech/Braiile-Vision.git
cd Braiile-Vision

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment (optional — enables Groq AI correction)
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_...

# 5. Start server
python3 app.py
```

Open **http://localhost:5050** in Chrome or Edge.

### Health check

```bash
curl http://localhost:5050/api/health
# → {"status":"ok","cnn_pytorch":true,"tensorflow":true,"yolo":true,"groq":true}
```

---

## 🌐 Deploy on Render

### Step 1 — Create Web Service

1. [dashboard.render.com](https://dashboard.render.com) → **New +** → **Web Service**
2. Connect repo: `atuljha-tech/Braiile-Vision`

### Step 2 — Configure

| Field | Value |
|-------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `bash build.sh` |
| **Start Command** | `gunicorn app:app -c gunicorn.conf.py` |
| **Instance type** | Free (or paid for no sleep) |

### Step 3 — Environment Variables

Paste these in **Environment → Add Environment Variable**:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.9` |
| `GROQ_API_KEY` | `gsk_xxxxxxxxxxxxxxxxxxxx` |
| `DISABLE_SERVER_TTS` | `1` |
| `OMP_NUM_THREADS` | `1` |

> Do **not** set `PORT` — Render injects it automatically.

### Step 4 — Deploy & verify

Click **Create Web Service** → wait 5–15 min for PyTorch/TF install → visit your URL.

Check: `https://<your-app>.onrender.com/api/health`

---

## ▲ Vercel Note

Vercel cannot run PyTorch/TensorFlow. Deploy the full backend on **Render**. Vercel is only relevant for a separate static frontend calling your Render API via `NEXT_PUBLIC_API_URL`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Status of all models (PyTorch, TF, YOLO, Groq) |
| `GET` | `/api/status` | Scanner status + session stats |
| `GET` | `/api/frame` | Latest annotated camera frame (base64 JPEG) |
| `GET` | `/api/history` | Last 30 recognised text segments |
| `GET` | `/api/voices` | Available TTS voices |
| `GET` | `/api/demo` | Run demo on bundled test image |
| `POST` | `/api/upload` | Upload image for OCR |
| `POST` | `/api/process_frame` | Browser webcam frame (base64) |
| `POST` | `/api/speak` | Speak text via TTS `{"text":"..."}` |
| `POST` | `/api/speak_history` | Re-read full session history |
| `POST` | `/api/pause` | Pause/resume `{"paused":true}` |
| `POST` | `/api/clear` | Clear session history |
| `POST` | `/api/tts_settings` | Update TTS rate/volume/voice |
| `POST` | `/api/camera/restart` | Restart OpenCV camera |

---

## 📁 Project Structure

```
Braiile-Vision/
├── app.py                          # Flask server — all API endpoints
├── requirements.txt                # All dependencies (PyTorch, TF, YOLO, Groq)
├── build.sh                        # Render build script
├── gunicorn.conf.py                # Gunicorn config
├── render.yaml                     # Render blueprint
│
├── braille_ai/
│   ├── cnn_predictor.py            # ✅ PyTorch CNN — cell classifier (A–Z)
│   ├── tf_predictor.py             # ✅ TensorFlow CNN — cell classifier (A–Z)
│   ├── yolo_dot_detector.py        # ✅ YOLOv8n — dot detector
│   ├── cell_classifier.py          # ✅ Ensemble: PyTorch + TensorFlow vote
│   ├── ocr_corrector.py            # ✅ Groq LLaMA 3.1 correction
│   ├── braille_decoder.py          # Braille bit-mask → letter map
│   └── dot_detector.py             # Classical CV fallback
│
├── braille_ocr/
│   ├── realtime/
│   │   ├── braille_detector.py     # Core pipeline: YOLO+CV dots → cells → text
│   │   ├── camera_loop.py          # OpenCV camera loop
│   │   ├── session.py              # Scan session + history
│   │   └── tts_engine.py           # pyttsx3 TTS
│   └── core/
│       ├── ocr.py                  # Full-page OCR
│       ├── segmentation.py         # Row/cell segmentation
│       └── transcription.py        # Braille → English
│
├── synthetic_dataset/
│   ├── generated/                  # 1,560 cell images (A–Z × 60 variants)
│   ├── models/
│   │   ├── braille_cnn.pth         # ✅ PyTorch weights (8 MB)
│   │   ├── braille_cnn_tf.keras    # ✅ TensorFlow weights (24 MB)
│   │   ├── braille_dots_yolov8n.pt # ✅ YOLOv8 weights (18 MB)
│   │   └── tf_class_labels.txt     # Class label map (A–Z)
│   ├── yolo/                       # YOLO dataset (400 page images + labels)
│   └── scripts/
│       ├── generate_dataset.py     # Cell image generator
│       ├── generate_yolo_dataset.py# YOLO page image + label generator
│       ├── train_model.py          # PyTorch trainer
│       ├── train_model_tf.py       # TensorFlow trainer
│       └── train_yolo.py           # YOLOv8 trainer
│
├── public/
│   ├── ss1.png                     # Screenshot: light theme
│   └── ss2.png                     # Screenshot: dark theme
│
├── templates/index.html            # Web UI (Jinja2)
└── static/
    ├── css/style.css               # Dark + light theme
    └── js/app.js                   # Camera, upload, TTS frontend
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume scanning |
| `R` | Repeat last detected text |
| `H` | Read full history aloud |
| `C` | Copy last text to clipboard |
| `U` | Open upload dialog |
| `D` | Run demo |

---

## 🔑 Environment Variables

```env
GROQ_API_KEY=gsk_your_key_here     # Groq LLaMA correction (optional)
PORT=5050                           # Local only — Render sets this automatically
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**🇮🇳 Jai Hind · जय हिन्द**

*Three AI models. One mission. Every person deserves to read.*

*Dedicated to the visually impaired community and the spirit of accessible technology.*

</div>
