# 😷 MaskSense AI — Face Mask Detection System

> Real-time face mask detection powered by MobileNetV2 + Haar Cascade, wrapped in a production-grade Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Detection** | Real-time webcam streaming with annotated overlays |
| **Upload Media** | Process still images and video files |
| **Analytics** | Pie/bar charts, compliance gauge, session statistics |
| **AI Model Info** | Architecture summary, training curves, model introspection |
| **Alert System** | Visual warnings when no-mask faces are detected |
| **Professional UI** | Dark futuristic glassmorphism design |

---

## 📁 Project Structure

```
project/
├── app.py                          ← Main Streamlit application
├── requirements.txt
├── README.md
├── model/
│   └── mask_model.h5               ← Keras MobileNetV2 model
└── haarcascade/
    └── haarcascade_frontalface_default.xml
```

---

## 🚀 Quick Start

### 1. Clone / download

```bash
git clone <your-repo-url>
cd face-mask-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Use `opencv-python` instead of `opencv-python-headless` if you need GUI windows.

### 3. Place model files

```
model/mask_model.h5
haarcascade/haarcascade_frontalface_default.xml
```

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output** | Binary (Mask / No Mask) |
| **Face Detector** | Haar Cascade FrontalFace |
| **Framework** | TensorFlow 2.x / Keras |

### Inference Pipeline

```
Webcam Frame
    ↓
Haar Cascade → Detect Faces → Extract ROI
    ↓
Resize to 224×224 → Normalize (÷255)
    ↓
MobileNetV2 → Softmax
    ↓
"Mask 😷" or "No Mask ❌" + Confidence %
```

---

## 🎛️ Sidebar Controls

- **Confidence Threshold** — Slider to filter low-confidence predictions (0.30–0.99)
- **Camera Index** — Select webcam (0 = default, 1 = secondary, etc.)
- **Navigation** — Switch between Dashboard, Live Detection, Upload, Analytics, Model, Settings

---

## 📊 Pages

### 🏠 Dashboard
- Session KPI cards (frames, faces, masks, FPS)
- Compliance ratio progress bars
- Live detection log feed

### 📷 Live Detection
- START / STOP / PAUSE / RECONNECT / SCREENSHOT controls
- Real-time annotated video feed
- Green box = Mask, Red box = No Mask
- Confidence gauge + live log

### 📂 Upload Media
- Image upload → annotated result + download
- Video upload → processed AVI + frame preview

### 📊 Analytics
- Interactive Plotly pie + bar charts
- Compliance rate gauge
- Session reset button

### 🧠 AI Model
- Model architecture summary
- Haar Cascade parameters
- Expandable training curves (illustrative)
- MobileNetV2 explanation

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| `Model not found` | Put `mask_model.h5` in `model/` folder |
| `Cascade not found` | Put XML in `haarcascade/` folder |
| `Camera index 0 failed` | Try camera index 1 or 2 in Settings |
| `TensorFlow import error` | `pip install tensorflow` |
| `Plotly charts not showing` | `pip install plotly` |
| Slow FPS | Reduce resolution or use a lighter model |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — Web UI framework
- **TensorFlow / Keras** — Deep learning inference
- **OpenCV** — Video capture & face detection
- **NumPy** — Numerical processing
- **Plotly** — Interactive charts
- **Pillow** — Image I/O

---

## 👤 Developer

Developed as a university final year AI project.  
Suitable for portfolio, hackathon demo, or production extension.

---

© 2024 MaskSense AI — All rights reserved
