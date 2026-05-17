"""
╔══════════════════════════════════════════════════════════════╗
║          FACE MASK DETECTION — AI SURVEILLANCE SYSTEM        ║
║          Powered by MobileNetV2 + Haar Cascade               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import time
import datetime
import io
import os
import base64
from collections import deque
# ──────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskSense AI · Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS (graceful fallback)
# ──────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "model/mask_model.h5"
CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"
IMG_SIZE = (224, 224)
HISTORY_MAXLEN = 200
CLASS_LABELS = ["Mask 😷", "No Mask ❌"]
MASK_COLOR = (0, 220, 100)      # green BGR
NO_MASK_COLOR = (0, 50, 255)    # red BGR

# ──────────────────────────────────────────────────────────────
# CSS — DARK FUTURISTIC GLASSMORPHISM THEME
# ──────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #050a18;
    --bg-secondary: #0b1428;
    --bg-card:      rgba(10, 20, 50, 0.75);
    --accent-cyan:  #00e5ff;
    --accent-green: #00ff88;
    --accent-red:   #ff2d55;
    --accent-gold:  #ffd700;
    --text-primary: #e0eaff;
    --text-dim:     #5a7aaa;
    --border-glow:  rgba(0, 229, 255, 0.3);
    --shadow-glow:  0 0 20px rgba(0, 229, 255, 0.15);
    --glass-blur:   blur(16px);
    --font-head:    'Rajdhani', sans-serif;
    --font-mono:    'Space Mono', monospace;
    --font-body:    'Inter', sans-serif;
    --radius:       12px;
    --radius-lg:    20px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text-primary);
}
.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(0,229,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(0,255,136,0.04) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(0,229,255,0.025) 39px,
            rgba(0,229,255,0.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(0,229,255,0.025) 39px,
            rgba(0,229,255,0.025) 40px
        );
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5, 10, 30, 0.95) !important;
    border-right: 1px solid var(--border-glow);
    backdrop-filter: var(--glass-blur);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: var(--font-head) !important;
    letter-spacing: 0.06em;
    color: var(--text-primary) !important;
}
h1 { font-size: 2.4rem !important; font-weight: 700 !important; }
h2 { font-size: 1.7rem !important; font-weight: 600 !important; }
h3 { font-size: 1.3rem !important; }

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    backdrop-filter: var(--glass-blur);
    box-shadow: var(--shadow-glow);
    transition: box-shadow 0.3s ease, transform 0.2s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    box-shadow: 0 0 32px rgba(0, 229, 255, 0.28);
    transform: translateY(-2px);
}

/* ── Metric card ── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.06), rgba(0,255,136,0.04));
    border: 1px solid rgba(0, 229, 255, 0.22);
    border-radius: var(--radius);
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: var(--glass-blur);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: -40%;
    left: -40%;
    width: 80%;
    height: 80%;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    animation: pulse-glow 3s ease-in-out infinite alternate;
}
@keyframes pulse-glow {
    from { opacity: 0.4; transform: scale(1); }
    to   { opacity: 1;   transform: scale(1.2); }
}
.metric-icon  { font-size: 2rem; margin-bottom: 0.3rem; }
.metric-value {
    font-family: var(--font-mono);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-shadow: 0 0 14px var(--accent-cyan);
}
.metric-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 0.2rem;
}

/* ── Status badge ── */
.badge-mask   { background: rgba(0,255,136,0.15); border: 1px solid var(--accent-green); color: var(--accent-green); border-radius: 20px; padding: 3px 12px; font-family: var(--font-mono); font-size: 0.85rem; }
.badge-nomask { background: rgba(255,45,85,0.15);  border: 1px solid var(--accent-red);   color: var(--accent-red);   border-radius: 20px; padding: 3px 12px; font-family: var(--font-mono); font-size: 0.85rem; }
.badge-idle   { background: rgba(90,122,170,0.15); border: 1px solid var(--text-dim);     color: var(--text-dim);     border-radius: 20px; padding: 3px 12px; font-family: var(--font-mono); font-size: 0.85rem; }

/* ── Alert banners ── */
.alert-warning {
    background: linear-gradient(90deg, rgba(255,45,85,0.18), rgba(255,100,0,0.10));
    border: 1px solid var(--accent-red);
    border-left: 4px solid var(--accent-red);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    color: #ff6b8a;
    font-family: var(--font-mono);
    animation: flash-border 1.2s ease-in-out infinite alternate;
}
.alert-success {
    background: linear-gradient(90deg, rgba(0,255,136,0.12), rgba(0,229,255,0.06));
    border: 1px solid var(--accent-green);
    border-left: 4px solid var(--accent-green);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    color: var(--accent-green);
    font-family: var(--font-mono);
}
@keyframes flash-border {
    from { border-left-color: var(--accent-red); }
    to   { border-left-color: rgba(255,45,85,0.3); }
}

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    background: linear-gradient(120deg, var(--accent-cyan) 0%, var(--accent-green) 60%, var(--accent-gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    text-shadow: none;
}
.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 0.92rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-green), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0%   { opacity: 0.4; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.4; }
}

/* ── Scan animation ── */
.scan-frame {
    position: relative;
    display: inline-block;
    border: 2px solid var(--accent-cyan);
    border-radius: 8px;
    box-shadow: 0 0 24px rgba(0,229,255,0.3), inset 0 0 24px rgba(0,229,255,0.05);
    overflow: hidden;
}
.scan-line {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    animation: scan-move 2s linear infinite;
    box-shadow: 0 0 12px var(--accent-cyan);
}
@keyframes scan-move {
    from { top: 0%;   }
    to   { top: 100%; }
}

/* ── Corner brackets ── */
.corner-tl, .corner-tr, .corner-bl, .corner-br {
    position: absolute;
    width: 18px;
    height: 18px;
    border-color: var(--accent-cyan);
    border-style: solid;
    z-index: 10;
}
.corner-tl { top: -1px;  left: -1px;  border-width: 3px 0 0 3px; }
.corner-tr { top: -1px;  right: -1px; border-width: 3px 3px 0 0; }
.corner-bl { bottom: -1px; left: -1px;  border-width: 0 0 3px 3px; }
.corner-br { bottom: -1px; right: -1px; border-width: 0 3px 3px 0; }

/* ── Log feed ── */
.log-feed {
    background: rgba(0, 5, 15, 0.8);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: var(--radius);
    padding: 0.8rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    height: 160px;
    overflow-y: auto;
    line-height: 1.7;
}
.log-mask   { color: var(--accent-green); }
.log-nomask { color: var(--accent-red); }
.log-info   { color: var(--accent-cyan); }
.log-ts     { color: #3a4a6a; margin-right: 6px; }

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
    box-shadow: 0 0 8px var(--accent-cyan);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(0,255,136,0.08)) !important;
    border: 1px solid rgba(0,229,255,0.4) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-head) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(0,255,136,0.18)) !important;
    box-shadow: 0 0 18px rgba(0,229,255,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,5,20,0.6);
    border-radius: 10px;
    border: 1px solid var(--border-glow);
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-dim) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.18), rgba(0,255,136,0.10)) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.2) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}

/* ── Selectbox / widgets ── */
.stSelectbox > div, .stNumberInput > div {
    background: rgba(0,20,50,0.7) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(0,229,255,0.15);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
}
.footer .tech-pills span {
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px;
    display: inline-block;
    font-size: 0.72rem;
    color: var(--accent-cyan);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(0,20,50,0.5) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    color: var(--accent-cyan) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.5); }

/* ── Loading spinner ── */
.loading-ring {
    display: inline-block;
    width: 32px; height: 32px;
    border: 3px solid rgba(0,229,255,0.15);
    border-top-color: var(--accent-cyan);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Gauge container ── */
.gauge-wrap { text-align: center; padding: 0.5rem; }
.gauge-val  { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700; }

/* ── Sidebar logo ── */
.sb-logo {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.sb-logo .icon-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 72px; height: 72px;
    border: 2px solid var(--accent-cyan);
    border-radius: 50%;
    font-size: 2.4rem;
    box-shadow: 0 0 24px rgba(0,229,255,0.4);
    animation: breathe 4s ease-in-out infinite alternate;
}
@keyframes breathe {
    from { box-shadow: 0 0 12px rgba(0,229,255,0.3); }
    to   { box-shadow: 0 0 36px rgba(0,229,255,0.7); }
}
.sb-logo .app-name {
    font-family: var(--font-head);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--accent-cyan);
    margin-top: 0.5rem;
    display: block;
    text-shadow: 0 0 10px rgba(0,229,255,0.5);
}
.sb-logo .app-tagline {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Detection frame overlay ── */
.no-mask-warn {
    border: 3px solid var(--accent-red) !important;
    box-shadow: 0 0 30px rgba(255,45,85,0.5) !important;
}

/* ── Separator ── */
.neon-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.4), transparent);
    margin: 1.2rem 0;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green  { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); animation: blink 1.4s ease-in-out infinite; }
.dot-red    { background: var(--accent-red);   box-shadow: 0 0 8px var(--accent-red);   animation: blink 0.7s ease-in-out infinite; }
.dot-idle   { background: var(--text-dim); }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
</style>
"""

# ──────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "cam_running": False,
        "cam_paused": False,
        "total_frames": 0,
        "total_faces": 0,
        "mask_count": 0,
        "no_mask_count": 0,
        "fps": 0.0,
        "last_label": "—",
        "last_conf": 0.0,
        "log_entries": deque(maxlen=HISTORY_MAXLEN),
        "screenshot": None,
        "session_start": datetime.datetime.now(),
        "conf_threshold": 0.60,
        "cam_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ──────────────────────────────────────────────────────────────
# MODEL / CASCADE LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    if not TF_AVAILABLE:
        return None, "TensorFlow not installed."
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = keras.models.load_model(path, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_cascade(path: str):
    if not os.path.exists(path):
        return None, f"Cascade file not found: {path}"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None, "Failed to load Haar Cascade."
    return cascade, None

# ──────────────────────────────────────────────────────────────
# FACE DETECTION + PREDICTION
# ──────────────────────────────────────────────────────────────
def detect_and_predict(frame, cascade, model, conf_threshold=0.5):
    """
    Detect faces, run model, return annotated frame + results list.
    results: list of dicts {label, conf, bbox}
    """
    results = []
    if cascade is None or model is None:
        return frame, results

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract + preprocess face ROI
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_res = cv2.resize(face_rgb, IMG_SIZE)
        face_arr = np.expand_dims(face_res / 255.0, axis=0).astype(np.float32)

        # Predict
        pred = np.array(model.predict(face_arr, verbose=0)).flatten()

        # Binary output: [mask_prob, no_mask_prob] OR single sigmoid
        if pred.size == 1:
            no_mask_prob = float(pred[0])
            mask_prob = 1.0 - no_mask_prob
        else:
            mask_prob = float(pred[0])
            no_mask_prob = float(pred[1])

        if mask_prob >= no_mask_prob:
            label = "Mask 😷"
            conf  = mask_prob
            color = MASK_COLOR
        else:
            label = "No Mask ❌"
            conf  = no_mask_prob
            color = NO_MASK_COLOR

        results.append({"label": label, "conf": conf, "bbox": (x, y, w, h)})

        # Draw rectangle
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        # Label background pill
        label_text = f"{label}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, results

def preprocess_image_for_prediction(img_bgr, cascade, model, conf_threshold=0.5):
    """Run detection on a still image."""
    annotated, results = detect_and_predict(img_bgr.copy(), cascade, model, conf_threshold)
    return annotated, results

# ──────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ──────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def add_log(entry: str, kind: str = "info"):
    css_cls = {"mask": "log-mask", "nomask": "log-nomask", "info": "log-info"}.get(kind, "log-info")
    html = f'<span class="log-ts">[{ts()}]</span><span class="{css_cls}">{entry}</span>'
    st.session_state.log_entries.appendleft(html)

def render_log_feed():
    entries_html = "<br>".join(list(st.session_state.log_entries)[:40])
    st.markdown(f'<div class="log-feed">{entries_html}</div>', unsafe_allow_html=True)

def render_metric(icon, value, label):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def confidence_color(conf: float) -> str:
    if conf >= 0.85: return "#00ff88"
    if conf >= 0.65: return "#ffd700"
    return "#ff2d55"

def render_confidence_gauge(conf: float, label: str):
    color = confidence_color(conf)
    pct = int(conf * 100)
    st.markdown(f"""
    <div class="gauge-wrap">
        <div class="gauge-val" style="color:{color};text-shadow:0 0 12px {color};">{pct}%</div>
        <div style="font-size:0.75rem;color:#5a7aaa;letter-spacing:0.12em;text-transform:uppercase;">Confidence</div>
        <div style="font-size:0.9rem;margin-top:4px;">{label}</div>
    </div>""", unsafe_allow_html=True)
    st.progress(conf)

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sb-logo">
            <div class="icon-ring">😷</div>
            <span class="app-name">MASKSENSE</span>
            <div class="app-tagline">AI · Detection System</div>
        </div>
        <div class="neon-sep"></div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("### ⬡  NAVIGATION")
        page = st.radio(
            label="",
            options=["🏠  Dashboard", "📷  Live Detection", "📂  Upload Media",
                     "📊  Analytics", "🧠  AI Model", "⚙️  Settings"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="neon-sep"></div>', unsafe_allow_html=True)

        # Camera settings
        st.markdown("### ⚙  CAMERA SETTINGS")
        st.session_state.conf_threshold = st.slider(
            "Confidence Threshold", 0.30, 0.99, st.session_state.conf_threshold, 0.01,
            help="Minimum confidence to display prediction"
        )
        st.session_state.cam_index = st.number_input(
            "Camera Index", min_value=0, max_value=10,
            value=st.session_state.cam_index, step=1,
            help="0 = default webcam"
        )

        st.markdown('<div class="neon-sep"></div>', unsafe_allow_html=True)

        # Status
        st.markdown("### ◉  STATUS")
        if st.session_state.cam_running and not st.session_state.cam_paused:
            st.markdown('<span class="status-dot dot-green"></span>**LIVE**', unsafe_allow_html=True)
        elif st.session_state.cam_paused:
            st.markdown('<span class="status-dot dot-red"></span>**PAUSED**', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot dot-idle"></span>**IDLE**', unsafe_allow_html=True)

        uptime = datetime.datetime.now() - st.session_state.session_start
        st.caption(f"Uptime: {str(uptime).split('.')[0]}")

        st.markdown('<div class="neon-sep"></div>', unsafe_allow_html=True)

        # Model info
        with st.expander("🧠 Model Info"):
            st.markdown("""
            **Architecture:** MobileNetV2  
            **Task:** Binary Classification  
            **Input:** 224×224 RGB  
            **Classes:** Mask · No Mask  
            **Backend:** TensorFlow/Keras  
            """)

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            MaskSense AI uses deep learning to detect
            face masks in real time. Built for public
            safety monitoring and AI portfolio demos.

            ---
            🔗 [GitHub](#) · [LinkedIn](#)
            """)

    return page.split("  ")[-1].strip()

# ──────────────────────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────────────────────

def page_dashboard():
    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">MASKSENSE AI</div>
        <div class="hero-subtitle">Real-Time Face Mask Detection · Powered by MobileNetV2</div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric("👁", st.session_state.total_frames, "Frames Processed")
    with c2: render_metric("👤", st.session_state.total_faces,  "Faces Detected")
    with c3: render_metric("😷", st.session_state.mask_count,   "Masks Found")
    with c4: render_metric("❌", st.session_state.no_mask_count,"No Mask")
    with c5:
        fps_val = f"{st.session_state.fps:.1f}"
        render_metric("⚡", fps_val, "FPS")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🖥  Quick Start Guide")
        st.markdown("""
        <div style="line-height:2.2;font-family:var(--font-mono);font-size:0.85rem;color:#8aaccc;">
        <b style="color:#00e5ff;">01</b> &nbsp; Open <b>Live Detection</b> tab<br>
        <b style="color:#00e5ff;">02</b> &nbsp; Click <b>▶ START CAMERA</b><br>
        <b style="color:#00e5ff;">03</b> &nbsp; Position face in frame<br>
        <b style="color:#00e5ff;">04</b> &nbsp; AI detects mask status in real-time<br>
        <b style="color:#00e5ff;">05</b> &nbsp; Use <b>Upload Media</b> for offline images<br>
        <b style="color:#00e5ff;">06</b> &nbsp; View <b>Analytics</b> for session statistics
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📡  Detection Ratio")
        total = st.session_state.mask_count + st.session_state.no_mask_count
        mask_pct = (st.session_state.mask_count / total * 100) if total else 0
        no_mask_pct = 100 - mask_pct if total else 0

        st.markdown(f"**😷 With Mask**")
        st.progress(mask_pct / 100)
        st.caption(f"{mask_pct:.1f}%  ({st.session_state.mask_count} detections)")

        st.markdown(f"**❌ No Mask**")
        st.progress(no_mask_pct / 100)
        st.caption(f"{no_mask_pct:.1f}%  ({st.session_state.no_mask_count} detections)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Log feed
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋  Detection Log")
    render_log_feed()
    st.markdown('</div>', unsafe_allow_html=True)


def page_live():
    st.markdown("## 📷  Live Webcam Detection")

    model, model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err  = load_cascade(CASCADE_PATH)

    if model_err:
        st.error(f"⚠️ Model: {model_err}")
    if cas_err:
        st.error(f"⚠️ Cascade: {cas_err}")
    if model_err or cas_err:
        st.info("💡 Ensure `model/mask_model.h5` and `haarcascade/haarcascade_frontalface_default.xml` are in the project folder.")
        return

    # ── Controls ──
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
    with ctrl1:
        if not st.session_state.cam_running:
            if st.button("▶  START CAMERA"):
                st.session_state.cam_running = True
                st.session_state.cam_paused  = False
                add_log("Camera started", "info")
                st.rerun()
    with ctrl2:
        if st.session_state.cam_running:
            if st.button("⏹  STOP CAMERA"):
                st.session_state.cam_running = False
                st.session_state.cam_paused  = False
                add_log("Camera stopped", "info")
                st.rerun()
    with ctrl3:
        if st.session_state.cam_running:
            label = "▶  RESUME" if st.session_state.cam_paused else "⏸  PAUSE"
            if st.button(label):
                st.session_state.cam_paused = not st.session_state.cam_paused
                add_log("Detection paused" if st.session_state.cam_paused else "Detection resumed", "info")
                st.rerun()
    with ctrl4:
        if st.session_state.screenshot is not None:
            buf = io.BytesIO()
            if PIL_AVAILABLE:
                Image.fromarray(cv2.cvtColor(st.session_state.screenshot, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
            st.download_button("📸  DOWNLOAD", buf.getvalue(), "screenshot.png", "image/png")
    with ctrl5:
        if st.session_state.cam_running:
            if st.button("🔄  RECONNECT"):
                add_log("Reconnect requested", "info")
                st.rerun()

    st.markdown('<div class="neon-sep"></div>', unsafe_allow_html=True)

    if not st.session_state.cam_running:
        st.markdown("""
        <div style="text-align:center;padding:4rem;border:1px dashed rgba(0,229,255,0.2);border-radius:16px;color:#3a4a6a;">
            <div style="font-size:4rem;margin-bottom:1rem;">📷</div>
            <div style="font-family:var(--font-mono);font-size:0.9rem;letter-spacing:0.1em;">
                CAMERA OFFLINE — Press START CAMERA to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Two-column layout ──
    feed_col, info_col = st.columns([3, 2])

    frame_placeholder = feed_col.empty()
    alert_placeholder  = feed_col.empty()

    with info_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        status_ph   = st.empty()
        gauge_ph    = st.empty()
        metrics_ph  = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📋 Live Log**")
        log_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Main capture loop ──
    cap = cv2.VideoCapture(st.session_state.cam_index)
    if not cap.isOpened():
        st.error(f"❌ Cannot open camera index {st.session_state.cam_index}. Try a different index in Settings.")
        st.session_state.cam_running = False
        return

    add_log(f"Camera {st.session_state.cam_index} opened", "info")
    fps_counter = deque(maxlen=30)
    no_mask_detected = False

    while st.session_state.cam_running:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame capture failed — trying to reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(st.session_state.cam_index)
            if not cap.isOpened():
                st.error("❌ Camera reconnect failed.")
                break
            continue

        st.session_state.total_frames += 1

        if not st.session_state.cam_paused:
            annotated, results = detect_and_predict(
                frame, cascade, model, st.session_state.conf_threshold
            )
            st.session_state.total_faces += len(results)
            no_mask_detected = False

            for r in results:
                if "No Mask" in r["label"]:
                    st.session_state.no_mask_count += 1
                    add_log(f"No Mask detected — {r['conf']*100:.1f}%", "nomask")
                    no_mask_detected = True
                else:
                    st.session_state.mask_count += 1
                    add_log(f"Mask detected — {r['conf']*100:.1f}%", "mask")

                st.session_state.last_label = r["label"]
                st.session_state.last_conf  = r["conf"]
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "PAUSED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (100, 100, 200), 3)

        # Screenshot storage
        st.session_state.screenshot = annotated.copy()

        # FPS
        t1 = time.perf_counter()
        fps_counter.append(1.0 / max(t1 - t0, 1e-6))
        st.session_state.fps = np.mean(fps_counter)

        # Display frame
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Alert
        if no_mask_detected:
            alert_placeholder.markdown("""
            <div class="alert-warning">🚨 WARNING — NO MASK DETECTED  |  Please wear a face mask!</div>
            """, unsafe_allow_html=True)
        else:
            alert_placeholder.markdown(
                '<div class="alert-success">✅ COMPLIANT — All detected faces wearing masks</div>'
                if results else "", unsafe_allow_html=True
            )

        # Info panel
        lbl = st.session_state.last_label
        conf = st.session_state.last_conf
        if lbl == "—":
            status_ph.markdown('<span class="badge-idle">● SCANNING...</span>', unsafe_allow_html=True)
        elif "Mask 😷" in lbl:
            status_ph.markdown('<span class="badge-mask">● MASK DETECTED</span>', unsafe_allow_html=True)
        else:
            status_ph.markdown('<span class="badge-nomask">● NO MASK DETECTED</span>', unsafe_allow_html=True)

        with gauge_ph:
            if conf > 0:
                render_confidence_gauge(conf, lbl)

        metrics_ph.markdown(f"""
        <div style="font-family:var(--font-mono);font-size:0.8rem;color:#5a7aaa;line-height:2;">
        FPS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#00e5ff;">{st.session_state.fps:.1f}</b><br>
        Frames &nbsp;&nbsp;&nbsp;: <b style="color:#00e5ff;">{st.session_state.total_frames}</b><br>
        Faces &nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#00e5ff;">{st.session_state.total_faces}</b><br>
        Masks &nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#00ff88;">{st.session_state.mask_count}</b><br>
        No Mask &nbsp;: <b style="color:#ff2d55;">{st.session_state.no_mask_count}</b>
        </div>
        """, unsafe_allow_html=True)

        log_ph.markdown(
            '<div class="log-feed">' +
            "<br>".join(list(st.session_state.log_entries)[:15]) +
            '</div>',
            unsafe_allow_html=True
        )

        time.sleep(0.03)   # ~33 FPS cap

    cap.release()
    add_log("Camera released", "info")


def page_upload():
    st.markdown("## 📂  Upload Media")

    model, model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err  = load_cascade(CASCADE_PATH)

    if model_err or cas_err:
        st.error("Model or Cascade not loaded. Check file paths.")
        return

    tab_img, tab_vid = st.tabs(["🖼  Image", "🎬  Video"])

    # ── Image tab ──
    with tab_img:
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_uploader"
        )
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Impossible de lire cette image. Essaie avec une image JPG ou PNG standard.")
                return

            with st.spinner("🔍 Running detection..."):
                annotated, results = preprocess_image_for_prediction(
                    img_bgr, cascade, model, st.session_state.conf_threshold
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_b:
                st.markdown("**Detected**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Results
            if results:
                st.markdown('<div class="neon-sep"></div>', unsafe_allow_html=True)
                st.markdown(f"**🔍 Found {len(results)} face(s):**")
                for i, r in enumerate(results, 1):
                    badge = "badge-mask" if "Mask 😷" in r["label"] else "badge-nomask"
                    st.markdown(
                        f'Face {i}: <span class="{badge}">{r["label"]}</span> '
                        f'— confidence <b>{r["conf"]*100:.1f}%</b>',
                        unsafe_allow_html=True
                    )
                    if "No Mask" in r["label"]:
                        st.markdown('<div class="alert-warning">⚠️ No mask detected on this face!</div>', unsafe_allow_html=True)
            else:
                st.info("No faces detected in the image.")

            # Download
            is_success, buffer = cv2.imencode(".png", annotated)
            if is_success:
                st.download_button("📥 Download Annotated Image", buffer.tobytes(),
                                   "detected.png", "image/png")

    # ── Video tab ──
    with tab_vid:
        st.info("Upload a video for frame-by-frame detection and download the processed output.")
        vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")
        if vid_file:
            tmp_path = "/tmp/uploaded_video.mp4"
            out_path = "/tmp/output_video.avi"

            with open(tmp_path, "wb") as f:
                f.write(vid_file.read())

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps_vid, (w, h))

            progress_bar = st.progress(0)
            status_text  = st.empty()
            preview_ph   = st.empty()
            frame_idx    = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_f, _ = detect_and_predict(frame, cascade, model, st.session_state.conf_threshold)
                out.write(annotated_f)
                frame_idx += 1
                pct = frame_idx / max(total_frames, 1)
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(f"Processing frame **{frame_idx}** / {total_frames}")
                if frame_idx % 15 == 0:
                    preview_ph.image(cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB),
                                     caption="Preview", use_container_width=True)

            cap.release()
            out.release()

            with open(out_path, "rb") as f:
                st.download_button("📥 Download Processed Video", f.read(),
                                   "processed_video.avi", "video/avi")
            st.success("✅ Video processing complete!")


def page_analytics():
    st.markdown("## 📊  Analytics Dashboard")

    total = st.session_state.mask_count + st.session_state.no_mask_count

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1: render_metric("👤", st.session_state.total_faces,  "Total Faces")
    with k2: render_metric("😷", st.session_state.mask_count,   "With Mask")
    with k3: render_metric("❌", st.session_state.no_mask_count, "No Mask")
    mask_rate = (st.session_state.mask_count / total * 100) if total else 0
    with k4: render_metric("📈", f"{mask_rate:.1f}%", "Compliance Rate")

    st.markdown("<br>", unsafe_allow_html=True)

    if not PLOTLY_AVAILABLE:
        st.warning("Install `plotly` for interactive charts: `pip install plotly`")
        # Fallback text charts
        st.markdown(f"**Mask:** {st.session_state.mask_count}  |  **No Mask:** {st.session_state.no_mask_count}")
        return

    if total == 0:
        st.info("No detections yet. Run live detection or upload an image to see analytics.")
        return

    col_pie, col_bar = st.columns(2)

    # Pie chart
    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Mask 😷", "No Mask ❌"],
            values=[st.session_state.mask_count, st.session_state.no_mask_count],
            hole=0.55,
            marker=dict(colors=["#00ff88", "#ff2d55"],
                        line=dict(color="#050a18", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Space Mono", color="white"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaff"),
            title=dict(text="Detection Distribution", font=dict(family="Rajdhani", size=18, color="#00e5ff")),
            showlegend=True,
            legend=dict(font=dict(color="#e0eaff")),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart
    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=["Mask 😷", "No Mask ❌"],
            y=[st.session_state.mask_count, st.session_state.no_mask_count],
            marker=dict(
                color=["rgba(0,255,136,0.7)", "rgba(255,45,85,0.7)"],
                line=dict(color=["#00ff88", "#ff2d55"], width=2)
            ),
            text=[st.session_state.mask_count, st.session_state.no_mask_count],
            textposition="outside",
            textfont=dict(family="Space Mono", color="white"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaff"),
            title=dict(text="Count Comparison", font=dict(family="Rajdhani", size=18, color="#00e5ff")),
            xaxis=dict(gridcolor="rgba(0,229,255,0.08)", zeroline=False),
            yaxis=dict(gridcolor="rgba(0,229,255,0.08)", zeroline=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Compliance gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mask_rate,
        delta=dict(reference=80, valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#e0eaff")),
            bar=dict(color="#00e5ff"),
            bgcolor="rgba(0,20,50,0.5)",
            borderwidth=2,
            bordercolor="rgba(0,229,255,0.3)",
            steps=[
                dict(range=[0,  60], color="rgba(255,45,85,0.15)"),
                dict(range=[60, 80], color="rgba(255,215,0,0.12)"),
                dict(range=[80,100], color="rgba(0,255,136,0.12)"),
            ],
            threshold=dict(line=dict(color="#ffd700", width=3), thickness=0.8, value=80),
        ),
        number=dict(suffix="%", font=dict(family="Space Mono", color="#00e5ff", size=36)),
        title=dict(text="Compliance Rate", font=dict(family="Rajdhani", size=20, color="#00e5ff")),
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0eaff"),
        height=300,
        margin=dict(l=30, r=30, t=40, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Reset button
    if st.button("🔄  Reset Statistics"):
        for k in ["total_frames","total_faces","mask_count","no_mask_count","fps"]:
            st.session_state[k] = 0
        st.session_state.log_entries.clear()
        st.session_state.last_label = "—"
        st.session_state.last_conf  = 0.0
        st.success("Statistics reset.")
        st.rerun()


def page_model():
    st.markdown("## 🧠  AI Model Information")

    model, model_err = load_keras_model(MODEL_PATH)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏗  Architecture")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Base Model** | MobileNetV2 |
        | **Task** | Binary Classification |
        | **Input Shape** | 224 × 224 × 3 |
        | **Output** | Softmax / Sigmoid |
        | **Classes** | 2 (Mask / No Mask) |
        | **Framework** | TensorFlow 2.x / Keras |
        | **Optimizer** | Adam |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔬  Face Detection")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Method** | Haar Cascade |
        | **Classifier** | FrontalFace Default |
        | **Scale Factor** | 1.1 |
        | **Min Neighbours** | 5 |
        | **Min Size** | 60 × 60 |
        | **Library** | OpenCV |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Model summary
    if model and TF_AVAILABLE:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋  Model Summary")
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        summary_str = buf.getvalue()
        st.code(summary_str, language="text")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"Load model at `{MODEL_PATH}` to see architecture summary.")

    # MobileNetV2 explanation
    with st.expander("📖 How MobileNetV2 Works"):
        st.markdown("""
        **MobileNetV2** is a lightweight convolutional neural network designed for mobile and embedded vision applications.

        **Key Features:**
        - **Inverted Residuals:** Expands channel dimensions before applying depthwise convolutions, then projects back. This allows learning richer features with fewer parameters.
        - **Linear Bottlenecks:** Removes ReLU from bottleneck layers to prevent information loss.
        - **Depthwise Separable Convolutions:** Factorises standard convolutions into depthwise + pointwise, drastically reducing computation.
        - **~3.4M Parameters** (base) — highly efficient for real-time inference.

        **Transfer Learning in this project:**
        The base MobileNetV2 weights (pre-trained on ImageNet) are frozen. A custom classification head (Dense layers) is added and trained on mask / no-mask images.

        **Detection Pipeline:**
        ```
        Frame → Haar Cascade → Face ROI → Resize 224×224 → /255 normalise → MobileNetV2 → Softmax → Label
        ```
        """)

    # Training info placeholder
    with st.expander("📈 Training Details"):
        if PLOTLY_AVAILABLE:
            # Simulated curves for demo
            epochs = list(range(1, 21))
            acc  = [0.62 + 0.018*i - 0.0003*i*i for i in epochs]
            val  = [0.58 + 0.016*i - 0.0004*i*i for i in epochs]
            loss = [0.72 - 0.028*i + 0.0005*i*i for i in epochs]
            vloss= [0.76 - 0.025*i + 0.0006*i*i for i in epochs]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=acc,  name="Train Acc",  line=dict(color="#00e5ff",  width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=val,  name="Val Acc",    line=dict(color="#00ff88",  width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=loss, name="Train Loss", line=dict(color="#ff2d55",  width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=epochs, y=vloss,name="Val Loss",   line=dict(color="#ffd700",  width=2, dash="dot")))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0eaff"),
                title=dict(text="Training Curves (Illustrative)", font=dict(family="Rajdhani", size=16, color="#00e5ff")),
                xaxis=dict(title="Epoch", gridcolor="rgba(0,229,255,0.08)"),
                yaxis=dict(title="Value",  gridcolor="rgba(0,229,255,0.08)"),
                legend=dict(font=dict(color="#e0eaff")),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: Curves are illustrative. Replace with actual training history.")


def page_settings():
    st.markdown("## ⚙️  Settings")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎚  Detection Thresholds")
    new_thresh = st.slider("Confidence Threshold", 0.30, 0.99,
                           st.session_state.conf_threshold, 0.01)
    st.session_state.conf_threshold = new_thresh
    st.caption(f"Current: {new_thresh:.2f} — Predictions below this threshold are suppressed.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📷  Camera")
    new_cam = st.number_input("Camera Index", 0, 10, st.session_state.cam_index, 1)
    st.session_state.cam_index = new_cam
    st.caption("0 = default webcam, 1 = secondary camera, etc.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁  File Paths")
    st.code(f"Model   : {os.path.abspath(MODEL_PATH)}\nCascade : {os.path.abspath(CASCADE_PATH)}")
    model_ok   = "✅ Found" if os.path.exists(MODEL_PATH) else "❌ Missing"
    cascade_ok = "✅ Found" if os.path.exists(CASCADE_PATH) else "❌ Missing"
    st.markdown(f"**Model file:** {model_ok}")
    st.markdown(f"**Cascade file:** {cascade_ok}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="footer">
        <div style="font-size:1.1rem;font-family:var(--font-head);letter-spacing:0.15em;
                    color:rgba(0,229,255,0.7);margin-bottom:0.6rem;">
            MASKSENSE AI &nbsp;·&nbsp; Face Mask Detection System
        </div>
        <div style="margin-bottom:0.8rem;color:#3a4a6a;">
            Developed for AI & Deep Learning Portfolio · University Final Year Project
        </div>
        <div class="tech-pills">
            <span>Python</span><span>Streamlit</span><span>TensorFlow</span>
            <span>Keras</span><span>OpenCV</span><span>MobileNetV2</span>
            <span>Plotly</span><span>NumPy</span>
        </div>
        <div style="margin-top:1rem;color:#2a3a5a;">
            © 2024 MaskSense AI · All rights reserved
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Live Detection":
        page_live()
    elif page == "Upload Media":
        page_upload()
    elif page == "Analytics":
        page_analytics()
    elif page == "AI Model":
        page_model()
    elif page == "Settings":
        page_settings()

    render_footer()


if __name__ == "__main__":
    main()
