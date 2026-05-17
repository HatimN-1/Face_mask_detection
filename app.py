"""
╔══════════════════════════════════════════════════════════════╗
║             MASKSENSE AI — FACE MASK DETECTION SYSTEM        ║
║             Streamlit UI · MobileNetV2 · OpenCV              ║
╚══════════════════════════════════════════════════════════════╝

Sidebar avec icônes PNG personnalisées (base64) + navigation via query params.
"""

import base64
import datetime
import io
import os
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskSense AI · Face Mask Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ──────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
MODEL_PATH   = "model/mask_model.h5"
CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"
LOGO_PATH    = "assets/logo158.png"

IMG_SIZE       = (224, 224)
HISTORY_MAXLEN = 200

MASK_COLOR    = (0, 220, 120)    # BGR
NO_MASK_COLOR = (45, 45, 255)    # BGR

# Mapping slug → page name (used in query params)
SLUG_TO_PAGE = {
    "dashboard": "Dashboard",
    "live":      "Live Detection",
    "upload":    "Upload Media",
    "analytics": "Analytics",
    "model":     "AI Model",
    "settings":  "Settings",
}
PAGE_TO_SLUG = {v: k for k, v in SLUG_TO_PAGE.items()}

# Nav items with icon path + fallback emoji
PAGE_OPTIONS = {
    "Dashboard":      {"icon": "assets/dashboard_icon.png",    "emoji": "⌂"},
    "Live Detection": {"icon": "assets/model_small_icon.png",  "emoji": "▣"},
    "Upload Media":   {"icon": "assets/upload_icon.png",       "emoji": "▭"},
    "Analytics":      {"icon": "assets/analytics_icon.png",    "emoji": "▥"},
    "AI Model":       {"icon": "assets/ai_model_icon.png",     "emoji": "◌"},
    "Settings":       {"icon": "",                             "emoji": "⚙"},
}


# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def img_to_base64(path: str) -> str:
    """Convert a local image to base64 for inline HTML display."""
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return ""


def logo_html(width: int = 180) -> str:
    b64 = img_to_base64(LOGO_PATH)
    if b64:
        return (
            f'<img class="ms-logo-img" '
            f'src="data:image/png;base64,{b64}" '
            f'style="width:{width}px;" />'
        )
    return (
        f'<div class="logo-fallback" '
        f'style="width:{width}px;height:{int(width*0.68)}px;">M</div>'
    )


def now_ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def add_log(entry: str, kind: str = "info") -> None:
    css_cls = {"mask": "log-mask", "nomask": "log-nomask", "info": "log-info"}.get(kind, "log-info")
    html = (
        f'<span class="log-ts">[{now_ts()}]</span>'
        f'<span class="{css_cls}">{entry}</span>'
    )
    st.session_state.log_entries.appendleft(html)


def safe_percent(part: int, total: int) -> float:
    return (part / total * 100) if total else 0.0


# ──────────────────────────────────────────────────────────────
# CSS — PROFESSIONAL DARK NEON THEME
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #060b1c;
    --bg2: #091327;
    --panel: rgba(7,16,34,0.92);
    --panel2: rgba(8,28,44,0.82);
    --line: rgba(0,230,255,0.20);
    --line2: rgba(255,255,255,0.12);
    --cyan: #18e7ff;
    --cyan2: #00bcd4;
    --green: #19ff9b;
    --red: #ff3d2e;
    --text: #e7eeff;
    --muted: #7f93b7;
    --muted2: #5b6d91;
    --font-head: 'Rajdhani', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── App background ── */
.stApp {
    background:
        radial-gradient(circle at 52% 13%, rgba(0,238,255,.12), transparent 18%),
        radial-gradient(circle at 15% 100%, rgba(0,255,160,.05), transparent 22%),
        linear-gradient(rgba(0,229,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,.025) 1px, transparent 1px),
        var(--bg);
    background-size: auto, auto, 50px 50px, 50px 50px, auto;
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { background: transparent !important; }

/* Main content spacing */
.block-container {
    padding-top: 1.8rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
    max-width: 1550px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b1d 0%, #030716 100%) !important;
    border-right: 1px solid rgba(0,229,255,.16);
    box-shadow: 18px 0 50px rgba(0,0,0,.20);
    min-width: 280px !important;
    width: 280px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
    padding-left: .6rem;
    padding-right: .6rem;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: var(--font-head) !important; letter-spacing: .06em; }
p, div, span { font-family: var(--font-body); }

/* ── Sidebar logo / title ── */
.sidebar-logo-box {
    text-align: center;
    padding: 1.25rem 0 1.05rem;
}

.ms-logo-img {
    display: block;
    margin: 0 auto;
    border-radius: 17px;
    filter: drop-shadow(0 0 24px rgba(24,231,255,.45));
}

.logo-fallback {
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(4,13,28,.8);
    border: 1px solid rgba(24,231,255,.35);
    border-radius: 18px;
    font-family: var(--font-head);
    font-size: 64px;
    font-weight: 700;
    color: var(--cyan);
    filter: drop-shadow(0 0 24px rgba(24,231,255,.45));
}

.sidebar-title {
    margin-top: .9rem;
    font-family: var(--font-head);
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: .18em;
    color: #f4f8ff;
    text-shadow: 0 0 16px rgba(24,231,255,.55);
}

.sidebar-subtitle {
    margin-top: .3rem;
    font-family: var(--font-mono);
    font-size: .72rem;
    letter-spacing: .16em;
    color: #dbe5ff;
}

.sidebar-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(24,231,255,.25), transparent);
    margin: 1.1rem 0;
}

.sidebar-section-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: .08em;
    margin: .3rem 0 .6rem .3rem;
    color: #dbe5ff;
}

/* ── Custom Nav items ── */
.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 14px;
    border-radius: 12px;
    border: 1px solid transparent;
    cursor: pointer;
    text-decoration: none !important;
    background: transparent;
    transition: all .22s ease;
    color: #cdd8f0 !important;
}

.nav-item:hover {
    background: rgba(255,255,255,.045);
    border-color: rgba(255,255,255,.08);
    color: #e7eeff !important;
    text-decoration: none !important;
}

.nav-item.nav-active {
    background: linear-gradient(135deg, rgba(24,231,255,.16), rgba(25,255,155,.055));
    border-color: rgba(24,231,255,.32);
    box-shadow: inset 0 0 22px rgba(24,231,255,.06), 0 0 18px rgba(24,231,255,.08);
    color: #e7eeff !important;
}

.nav-icon-wrap {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.nav-icon-img {
    width: 26px;
    height: 26px;
    object-fit: contain;
    filter: brightness(0) invert(1) drop-shadow(0 0 5px rgba(24,231,255,.5));
    opacity: .85;
}

.nav-item.nav-active .nav-icon-img {
    filter: brightness(0) saturate(100%) invert(79%) sepia(100%) saturate(400%) hue-rotate(155deg) brightness(105%) drop-shadow(0 0 6px rgba(24,231,255,.8));
    opacity: 1;
}

.nav-emoji-icon {
    font-size: 1.3rem;
    line-height: 1;
    color: #cdd8f0;
}

.nav-item.nav-active .nav-emoji-icon {
    color: var(--cyan) !important;
    text-shadow: 0 0 10px rgba(24,231,255,.7);
}

.nav-label {
    font-family: var(--font-body);
    font-size: .97rem;
    font-weight: 500;
    flex: 1;
}

.nav-arrow {
    font-size: 1.3rem;
    color: rgba(200,210,235,.55);
    margin-left: auto;
}

.nav-item.nav-active .nav-arrow {
    color: rgba(200,210,235,.9);
}

/* ── Sidebar bottom cards (Model Info / About) ── */
.side-action-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: .95rem 1rem;
    margin: .45rem 0;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(10,26,46,.82), rgba(4,11,25,.84));
    box-shadow: 0 0 0 1px rgba(0,0,0,.22), inset 0 0 25px rgba(24,231,255,.025);
    cursor: pointer;
    text-decoration: none !important;
    color: var(--text) !important;
    transition: all .22s ease;
}

.side-action-card:hover {
    border-color: rgba(24,231,255,.3);
    box-shadow: 0 0 20px rgba(24,231,255,.08);
}

.side-action-icon {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(11,28,46,.88);
    border: 1px solid rgba(24,231,255,.13);
    box-shadow: 0 0 20px rgba(24,231,255,.08);
    font-size: 1.25rem;
    flex-shrink: 0;
}

.side-action-icon img {
    width: 24px;
    height: 24px;
    object-fit: contain;
    filter: brightness(0) invert(1);
    opacity: .85;
}

.side-action-main { flex: 1; }

.side-action-title {
    font-family: var(--font-head);
    font-size: 1.15rem;
    letter-spacing: .1em;
    color: var(--cyan);
    font-weight: 650;
    line-height: 1.1;
}

.side-action-desc {
    margin-top: .3rem;
    font-size: .82rem;
    line-height: 1.5;
    color: rgba(220,232,255,.70);
}

.side-action-arrow {
    font-size: 1.6rem;
    color: rgba(220,232,255,.75);
}

/* ── Hero section ── */
.hero-section {
    text-align: center;
    padding: 1.2rem 0 1.8rem;
}

.hero-logo-wrap img, .hero-logo-wrap .logo-fallback {
    max-width: 290px;
    border-radius: 20px;
    filter: drop-shadow(0 0 36px rgba(24,231,255,.34));
}

.hero-title {
    margin-top: 1.35rem;
    font-family: var(--font-head);
    font-size: 3.8rem;
    line-height: 1;
    font-weight: 700;
    letter-spacing: .18em;
    background: linear-gradient(90deg, #0ff, #19ff9b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 1.35rem;
    font-family: var(--font-mono);
    color: #5e80bd;
    letter-spacing: .28em;
    font-size: 1rem;
}

.hero-divider {
    height: 1px;
    width: 54%;
    margin: 2.1rem auto 0;
    background: linear-gradient(90deg, transparent, rgba(24,231,255,.48), rgba(25,255,155,.36), transparent);
}

/* ── Cards ── */
.ms-card, .metric-card, .feature-card {
    background: linear-gradient(135deg, rgba(6,30,43,.88), rgba(4,13,27,.92));
    border: 1px solid rgba(0,229,255,.22);
    border-radius: 15px;
    box-shadow: inset 0 0 28px rgba(24,231,255,.025), 0 10px 30px rgba(0,0,0,.16);
}

.metric-card {
    min-height: 168px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: .22s ease;
    overflow: hidden;
    position: relative;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(24,231,255,.42);
    box-shadow: 0 0 30px rgba(24,231,255,.12);
}

.metric-icon { font-size: 2.15rem; line-height: 1; color: #f1f6ff; margin-bottom: .72rem; }
.metric-icon.red { color: var(--red); text-shadow: 0 0 12px rgba(255,61,46,.55); font-size: 2.6rem; }
.metric-value {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 2.55rem;
    line-height: 1;
    color: var(--cyan);
    text-shadow: 0 0 18px rgba(24,231,255,.75);
}
.metric-label {
    margin-top: 1rem;
    font-family: var(--font-mono);
    color: #6986b7;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .13em;
}

.ms-card { padding: 1.35rem 1.45rem; margin-bottom: 1rem; }
.ms-card-title {
    font-family: var(--font-head);
    color: var(--cyan);
    font-size: 1.65rem;
    font-weight: 650;
    letter-spacing: .11em;
    margin-bottom: .55rem;
}
.ms-muted { color: var(--muted); line-height: 1.8; }

/* Feature card */
.feature-card {
    min-height: 176px;
    padding: 1.55rem 1.65rem;
    display: flex;
    align-items: center;
    gap: 1.6rem;
    margin: 1.25rem 0;
}

.feature-icon {
    width: 76px; height: 76px;
    border-radius: 50%;
    background: rgba(13,30,48,.85);
    border: 1px solid rgba(24,231,255,.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; color: #f3f6ff;
}

.feature-body { flex: 1; }
.feature-title {
    font-family: var(--font-head); font-size: 1.72rem;
    letter-spacing: .15em; color: var(--cyan); font-weight: 650;
}
.feature-desc { margin-top: .75rem; color: rgba(235,240,255,.82); font-size: 1.02rem; line-height: 1.65; }
.feature-arrow { font-size: 3rem; color: rgba(235,240,255,.85); }

/* Alerts */
.alert-warning {
    padding: 1rem 1.2rem;
    background: rgba(255,61,46,.13);
    border: 1px solid rgba(255,61,46,.55);
    border-left: 5px solid var(--red);
    border-radius: 12px;
    color: #ff8c83;
    font-family: var(--font-mono);
}
.alert-success {
    padding: 1rem 1.2rem;
    background: rgba(25,255,155,.08);
    border: 1px solid rgba(25,255,155,.35);
    border-left: 5px solid var(--green);
    border-radius: 12px;
    color: #78ffc5;
    font-family: var(--font-mono);
}

/* Badges */
.badge-mask, .badge-nomask, .badge-idle {
    border-radius: 999px; padding: .28rem .78rem;
    font-family: var(--font-mono); font-size: .86rem;
}
.badge-mask   { background: rgba(25,255,155,.11); border: 1px solid rgba(25,255,155,.45); color: var(--green); }
.badge-nomask { background: rgba(255,61,46,.12);  border: 1px solid rgba(255,61,46,.52);  color: #ff837b; }
.badge-idle   { background: rgba(127,147,183,.10); border: 1px solid rgba(127,147,183,.32); color: #a9b8d3; }

/* Log */
.log-feed {
    background: rgba(1,7,18,.68);
    border: 1px solid rgba(24,231,255,.13);
    border-radius: 13px;
    padding: .9rem 1rem;
    font-family: var(--font-mono);
    font-size: .78rem;
    color: var(--muted);
    height: 170px;
    overflow-y: auto;
    line-height: 1.75;
}
.log-ts     { color: #445a7e; margin-right: .35rem; }
.log-mask   { color: var(--green); }
.log-nomask { color: #ff766d; }
.log-info   { color: var(--cyan); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, rgba(24,231,255,.13), rgba(25,255,155,.07)) !important;
    border: 1px solid rgba(24,231,255,.35) !important;
    color: var(--cyan) !important;
    border-radius: 11px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    min-height: 44px !important;
    transition: all .2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: rgba(24,231,255,.7) !important;
    box-shadow: 0 0 22px rgba(24,231,255,.16) !important;
    transform: translateY(-1px);
}

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(8,19,38,.72) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 12px !important;
}

/* Footer */
.footer {
    margin-top: 2.2rem;
    padding: 1.8rem 0;
    text-align: center;
    border-top: 1px solid rgba(24,231,255,.13);
    color: #526b93;
    font-family: var(--font-mono);
    font-size: .8rem;
}
.tech-pills span {
    display: inline-block;
    margin: .25rem;
    padding: .24rem .72rem;
    border-radius: 999px;
    border: 1px solid rgba(24,231,255,.18);
    background: rgba(24,231,255,.055);
    color: #74eaff;
}

/* Mobile */
@media (max-width: 900px) {
    [data-testid="stSidebar"] { min-width: 260px !important; width: 260px !important; }
    .hero-title { font-size: 2.5rem; }
    .hero-divider { width: 90%; }
    .feature-card { flex-direction: column; align-items: flex-start; }
}
</style>
"""


# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
def init_session() -> None:
    defaults = {
        "page":          "Dashboard",
        "total_frames":  0,
        "total_faces":   0,
        "mask_count":    0,
        "no_mask_count": 0,
        "fps":           0.0,
        "last_label":    "—",
        "last_conf":     0.0,
        "log_entries":   deque(maxlen=HISTORY_MAXLEN),
        "session_start": datetime.datetime.now(),
        "conf_threshold": 0.60,
        "cam_index":     0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ──────────────────────────────────────────────────────────────
# QUERY PARAMS → page routing
# ──────────────────────────────────────────────────────────────
_qp = st.query_params
if "page" in _qp and _qp["page"] in SLUG_TO_PAGE:
    st.session_state.page = SLUG_TO_PAGE[_qp["page"]]


# ──────────────────────────────────────────────────────────────
# MODEL / CASCADE LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed."
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = keras.models.load_model(path, compile=False)
        return model, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_cascade(path: str):
    if not os.path.exists(path):
        return None, f"Cascade file not found: {path}"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None, "Failed to load Haar Cascade."
    return cascade, None


# ──────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────
def detect_and_predict(frame, cascade, model, conf_threshold=0.5):
    results = []
    if cascade is None or model is None:
        return frame, results

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
    )

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        face_rgb     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        face_arr     = np.expand_dims(face_resized / 255.0, axis=0).astype(np.float32)
        pred         = np.array(model.predict(face_arr, verbose=0)).flatten()

        if pred.size == 1:
            no_mask_prob = float(pred[0])
            mask_prob    = 1.0 - no_mask_prob
        else:
            mask_prob    = float(pred[0])
            no_mask_prob = float(pred[1])

        if mask_prob >= no_mask_prob:
            label = "Mask";    conf = mask_prob;    color = MASK_COLOR
        else:
            label = "No Mask"; conf = no_mask_prob; color = NO_MASK_COLOR

        if conf < conf_threshold:
            label = "Low Confidence"

        results.append({
            "label": label, "conf": conf, "bbox": (x, y, w, h),
            "mask_prob": mask_prob, "no_mask_prob": no_mask_prob,
        })

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        l, t = 22, 3
        cv2.line(frame, (x, y),     (x+l, y),     color, t)
        cv2.line(frame, (x, y),     (x, y+l),     color, t)
        cv2.line(frame, (x+w, y),   (x+w-l, y),   color, t)
        cv2.line(frame, (x+w, y),   (x+w, y+l),   color, t)
        cv2.line(frame, (x, y+h),   (x+l, y+h),   color, t)
        cv2.line(frame, (x, y+h),   (x, y+h-l),   color, t)
        cv2.line(frame, (x+w, y+h), (x+w-l, y+h), color, t)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, t)

        text = f"{label} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        y_text = max(28, y - 8)
        cv2.rectangle(frame, (x, y_text-th-12), (x+tw+14, y_text+4), color, -1)
        cv2.putText(frame, text, (x+7, y_text-4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    return frame, results


def update_stats_from_results(results) -> None:
    st.session_state.total_faces += len(results)
    for r in results:
        if r["label"] == "No Mask":
            st.session_state.no_mask_count += 1
            add_log(f"No Mask detected — {r['conf']*100:.1f}%", "nomask")
        elif r["label"] == "Mask":
            st.session_state.mask_count += 1
            add_log(f"Mask detected — {r['conf']*100:.1f}%", "mask")
        st.session_state.last_label = r["label"]
        st.session_state.last_conf  = r["conf"]


# ──────────────────────────────────────────────────────────────
# COMPONENT RENDERERS
# ──────────────────────────────────────────────────────────────
def render_metric(icon: str, value: str, label: str, red: bool = False) -> None:
    icon_class = "metric-icon red" if red else "metric-icon"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="{icon_class}">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_log_feed(max_items: int = 40) -> None:
    entries_html = "<br>".join(list(st.session_state.log_entries)[:max_items])
    if not entries_html:
        entries_html = (
            '<span class="log-ts">[--:--:--]</span>'
            '<span class="log-info">Waiting for detections...</span>'
        )
    st.markdown(f'<div class="log-feed">{entries_html}</div>', unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-body">
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            <div class="feature-arrow">›</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# SIDEBAR — custom HTML nav with PNG icons
# ──────────────────────────────────────────────────────────────
def _nav_icon_html(icon_path: str, emoji: str) -> str:
    """Return the <img> tag (base64) or an emoji span for the nav icon."""
    b64 = img_to_base64(icon_path)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" class="nav-icon-img" alt="">'
    return f'<span class="nav-emoji-icon">{emoji}</span>'


def render_sidebar() -> str:
    with st.sidebar:
        # ── Logo + title ──────────────────────────────────────
        st.markdown(
            f"""
            <div class="sidebar-logo-box">
                {logo_html(115)}
                <div class="sidebar-title">MASKSENSE</div>
                <div class="sidebar-subtitle">AI DETECTION SYSTEM</div>
            </div>
            <div class="sidebar-sep"></div>
            <div class="sidebar-section-title">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )

        # ── Nav items (pure HTML <a> links → query params) ────
        nav_html = '<div class="sidebar-nav">'
        for page_name, data in PAGE_OPTIONS.items():
            slug       = PAGE_TO_SLUG[page_name]
            is_active  = st.session_state.page == page_name
            active_cls = "nav-active" if is_active else ""
            icon_html  = _nav_icon_html(data["icon"], data["emoji"])
            arrow      = '<span class="nav-arrow">›</span>' if page_name not in ("Dashboard",) else ""

            nav_html += f"""
            <a href="?page={slug}" class="nav-item {active_cls}">
                <div class="nav-icon-wrap">{icon_html}</div>
                <span class="nav-label">{page_name}</span>
                {arrow}
            </a>"""
        nav_html += "</div>"
        st.markdown(nav_html, unsafe_allow_html=True)

        # ── Separator ─────────────────────────────────────────
        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # ── Model Info / About cards ──────────────────────────
        # Build small icon for Model Info (reuse ai_model_icon)
        model_icon_b64 = img_to_base64("assets/ai_model_icon.png")
        if model_icon_b64:
            mi_icon = f'<img src="data:image/png;base64,{model_icon_b64}" style="width:22px;height:22px;object-fit:contain;filter:brightness(0) invert(1);">'
        else:
            mi_icon = "♟"

        st.markdown(
            f"""
            <div class="side-action-card">
                <div class="side-action-icon">{mi_icon}</div>
                <div class="side-action-main">
                    <div class="side-action-title">MODEL INFO</div>
                    <div class="side-action-desc">View details about the AI model,
                    training data, and performance.</div>
                </div>
                <div class="side-action-arrow">›</div>
            </div>
            <div class="side-action-card">
                <div class="side-action-icon" style="font-family:Georgia,serif;font-style:italic;font-size:1.3rem;color:#18e7ff;">i</div>
                <div class="side-action-main">
                    <div class="side-action-title">ABOUT</div>
                    <div class="side-action-desc">Learn more about MaskSense AI
                    and the system.</div>
                </div>
                <div class="side-action-arrow">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Separator ─────────────────────────────────────────
        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # ── Quick controls ────────────────────────────────────
        st.session_state.conf_threshold = st.slider(
            "Confidence Threshold",
            0.30, 0.99,
            float(st.session_state.conf_threshold),
            0.01,
        )
        st.session_state.cam_index = st.number_input(
            "Camera Index",
            min_value=0, max_value=10,
            value=int(st.session_state.cam_index),
            step=1,
        )

        uptime = datetime.datetime.now() - st.session_state.session_start
        st.caption(f"Uptime: {str(uptime).split('.')[0]}")

    return st.session_state.page


# ──────────────────────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown(
        f"""
        <div class="hero-section">
            <div class="hero-logo-wrap">{logo_html(290)}</div>
            <div class="hero-title">MASKSENSE AI</div>
            <div class="hero-subtitle">FACE MASK DETECTION SYSTEM</div>
            <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric("▥", str(st.session_state.total_frames), "Frames Processed")
    with c2: render_metric("○", str(st.session_state.total_faces),  "Faces Detected")
    with c3: render_metric("◎", str(st.session_state.mask_count),   "Masks Found")
    with c4: render_metric("×", str(st.session_state.no_mask_count),"No Mask", red=True)
    with c5: render_metric("□", f"{st.session_state.fps:.1f}",      "FPS")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            <div class="ms-card">
                <div class="ms-card-title">QUICK START</div>
                <div class="ms-muted">
                    <b style="color:#18e7ff;">01</b> &nbsp; Go to <b>Live Detection</b><br>
                    <b style="color:#18e7ff;">02</b> &nbsp; Take a camera photo or upload media<br>
                    <b style="color:#18e7ff;">03</b> &nbsp; The model detects face + mask status<br>
                    <b style="color:#18e7ff;">04</b> &nbsp; Open Analytics to read compliance rate
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ms-card"><div class="ms-card-title">DETECTION LOG</div>', unsafe_allow_html=True)
        render_log_feed()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        render_feature_card("▣", "MODEL INFO",
            "View AI model details, architecture (MobileNetV2), training data, and performance metrics.")
        render_feature_card("▥", "ANALYTICS",
            "Real-time compliance charts, detection distribution, and session statistics.")
        render_feature_card("↗", "UPLOAD MEDIA",
            "Analyse images or videos by uploading files directly to the detection pipeline.")


def page_live_detection():
    st.markdown("## Live Detection")

    model,   model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err   = load_cascade(CASCADE_PATH)

    if model_err or cas_err:
        st.error("Model or Cascade not loaded.")
        if model_err: st.warning(model_err)
        if cas_err:   st.warning(cas_err)
        return

    st.markdown(
        """
        <div class="ms-card">
            <div class="ms-card-title">CAMERA SNAPSHOT DETECTION</div>
            <div class="ms-muted">
                Take a browser camera snapshot — the model will detect faces
                and classify mask status in real time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    img_file = st.camera_input("Take a photo")

    if img_file is None:
        st.markdown(
            """
            <div class="ms-card" style="text-align:center;padding:4rem 1rem;">
                <div style="font-size:4rem;">▣</div>
                <div class="ms-card-title">CAMERA READY</div>
                <div class="ms-muted">Take a photo to begin AI detection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    bytes_data = img_file.getvalue()
    img_bgr    = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Cannot read the image.")
        return

    start              = time.perf_counter()
    annotated, results = detect_and_predict(
        img_bgr.copy(), cascade, model, st.session_state.conf_threshold
    )
    elapsed                    = max(time.perf_counter() - start, 1e-6)
    st.session_state.total_frames += 1
    st.session_state.fps          = 1.0 / elapsed
    update_stats_from_results(results)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Detection Result",
            use_container_width=True,
        )

    with col2:
        total    = len(results)
        masks    = sum(1 for r in results if r["label"] == "Mask")
        no_masks = sum(1 for r in results if r["label"] == "No Mask")

        a, b, c = st.columns(3)
        with a: render_metric("○", str(total),    "Faces")
        with b: render_metric("◎", str(masks),    "With Mask")
        with c: render_metric("×", str(no_masks), "No Mask", red=True)

        st.markdown('<div class="ms-card"><div class="ms-card-title">RESULTS</div>', unsafe_allow_html=True)
        if not results:
            st.warning("No face detected.")
        else:
            for i, r in enumerate(results, 1):
                badge = "badge-mask" if r["label"] == "Mask" else "badge-nomask"
                st.markdown(
                    f'Face {i}: <span class="{badge}">{r["label"]}</span> '
                    f'— <b>{r["conf"]*100:.1f}%</b>',
                    unsafe_allow_html=True,
                )
            if no_masks > 0:
                st.markdown(
                    '<div class="alert-warning">NO MASK DETECTED — Please wear a face mask.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-success">COMPLIANT — All detected faces wearing masks.</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def page_upload_media():
    st.markdown("## Upload Media")

    model,   model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err   = load_cascade(CASCADE_PATH)

    if model_err or cas_err:
        st.error("Model or Cascade not loaded.")
        if model_err: st.warning(model_err)
        if cas_err:   st.warning(cas_err)
        return

    tab_img, tab_vid = st.tabs(["Image", "Video"])

    with tab_img:
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                st.error("Cannot read this image.")
                return

            start              = time.perf_counter()
            annotated, results = detect_and_predict(
                img_bgr.copy(), cascade, model, st.session_state.conf_threshold
            )
            elapsed                        = max(time.perf_counter() - start, 1e-6)
            st.session_state.total_frames += 1
            st.session_state.fps           = 1.0 / elapsed
            update_stats_from_results(results)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Original")
                st.image(cv2.cvtColor(img_bgr,     cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_b:
                st.markdown("### Detected")
                st.image(cv2.cvtColor(annotated,   cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown(
                '<div class="ms-card"><div class="ms-card-title">DETECTION RESULTS</div>',
                unsafe_allow_html=True,
            )
            if results:
                for i, r in enumerate(results, 1):
                    badge = "badge-mask" if r["label"] == "Mask" else "badge-nomask"
                    st.markdown(
                        f'Face {i}: <span class="{badge}">{r["label"]}</span> '
                        f'— confidence <b>{r["conf"]*100:.1f}%</b>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No faces detected in the image.")
            st.markdown("</div>", unsafe_allow_html=True)

            ok, buffer = cv2.imencode(".png", annotated)
            if ok:
                st.download_button(
                    "Download Annotated Image",
                    buffer.tobytes(),
                    file_name="masksense_detected.png",
                    mime="image/png",
                )

    with tab_vid:
        st.info("Video processing is available. Keep videos short on Streamlit Cloud to avoid timeout.")
        vid_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if vid_file:
            tmp_path = "/tmp/masksense_input_video.mp4"
            out_path = "/tmp/masksense_output_video.avi"

            with open(tmp_path, "wb") as f:
                f.write(vid_file.read())

            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                fps_vid,
                (width, height),
            )

            progress  = st.progress(0)
            status    = st.empty()
            preview   = st.empty()
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, results = detect_and_predict(
                    frame.copy(), cascade, model, st.session_state.conf_threshold
                )
                out.write(annotated)
                frame_idx += 1
                st.session_state.total_frames += 1
                update_stats_from_results(results)
                progress.progress(min(frame_idx / total_frames, 1.0))
                status.markdown(f"Processing frame **{frame_idx}** / {total_frames}")
                if frame_idx % 20 == 0:
                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Preview",
                        use_container_width=True,
                    )

            cap.release()
            out.release()

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    f.read(),
                    file_name="masksense_processed_video.avi",
                    mime="video/avi",
                )
            st.success("Video processing complete.")


def page_analytics():
    st.markdown("## Analytics")

    total      = st.session_state.mask_count + st.session_state.no_mask_count
    compliance = safe_percent(st.session_state.mask_count, total)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric("○",  str(st.session_state.total_faces),   "Total Faces")
    with c2: render_metric("◎",  str(st.session_state.mask_count),    "With Mask")
    with c3: render_metric("×",  str(st.session_state.no_mask_count), "No Mask", red=True)
    with c4: render_metric("↗",  f"{compliance:.1f}%",                "Compliance Rate")

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.info("No detections yet. Use Live Detection or Upload Media.")
        return

    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly to see charts: pip install plotly")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=["With Mask", "No Mask"],
                values=[st.session_state.mask_count, st.session_state.no_mask_count],
                hole=0.56,
                marker=dict(colors=["#19ff9b", "#ff3d2e"]),
                textinfo="label+percent",
            )
        ])
        fig_pie.update_layout(
            title="Detection Distribution",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e7eeff"), margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["With Mask", "No Mask"],
                y=[st.session_state.mask_count, st.session_state.no_mask_count],
                text=[st.session_state.mask_count, st.session_state.no_mask_count],
                textposition="outside",
                marker=dict(color=["#19ff9b", "#ff3d2e"]),
            )
        ])
        fig_bar.update_layout(
            title="Count Comparison",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e7eeff"),
            xaxis=dict(gridcolor="rgba(24,231,255,.08)"),
            yaxis=dict(gridcolor="rgba(24,231,255,.08)"),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compliance,
        number=dict(suffix="%", font=dict(color="#18e7ff", size=42)),
        title=dict(text="Compliance Rate", font=dict(color="#18e7ff")),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#e7eeff")),
            bar=dict(color="#18e7ff"),
            bgcolor="rgba(8,28,44,.6)",
            borderwidth=1,
            bordercolor="rgba(24,231,255,.25)",
            steps=[
                dict(range=[0, 60],  color="rgba(255,61,46,.18)"),
                dict(range=[60, 80], color="rgba(255,215,0,.16)"),
                dict(range=[80,100], color="rgba(25,255,155,.16)"),
            ],
        ),
    ))
    fig_gauge.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e7eeff"),
        margin=dict(l=20, r=20, t=35, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if st.button("Reset Statistics"):
        for key in ["total_frames", "total_faces", "mask_count", "no_mask_count"]:
            st.session_state[key] = 0
        st.session_state.fps        = 0.0
        st.session_state.last_label = "—"
        st.session_state.last_conf  = 0.0
        st.session_state.log_entries.clear()
        st.rerun()


def page_model_info():
    st.markdown("## AI Model")

    model, model_err = load_keras_model(MODEL_PATH)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="ms-card">
                <div class="ms-card-title">ARCHITECTURE</div>
                <div class="ms-muted">
                    <b>Base Model:</b> MobileNetV2<br>
                    <b>Task:</b> Binary Classification<br>
                    <b>Input Shape:</b> 224 × 224 × 3<br>
                    <b>Classes:</b> Mask / No Mask<br>
                    <b>Framework:</b> TensorFlow / Keras
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="ms-card">
                <div class="ms-card-title">FACE DETECTION</div>
                <div class="ms-muted">
                    <b>Method:</b> Haar Cascade<br>
                    <b>Classifier:</b> FrontalFace Default<br>
                    <b>Library:</b> OpenCV<br>
                    <b>ROI:</b> Face crop → Resize → Normalize → Prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if model is not None and TF_AVAILABLE:
        st.markdown(
            '<div class="ms-card"><div class="ms-card-title">MODEL SUMMARY</div>',
            unsafe_allow_html=True,
        )
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        st.code(buf.getvalue(), language="text")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"Load model at `{MODEL_PATH}` to see model summary.")
        if model_err:
            st.warning(model_err)

    with st.expander("How the detection pipeline works"):
        st.markdown(
            """
            **Pipeline:**
            ```
            Image / Frame
                ↓
            Haar Cascade detects face boxes
                ↓
            Each face is cropped and resized to 224×224
                ↓
            MobileNetV2 classifies Mask / No Mask
                ↓
            Result is drawn on the image
            ```
            """
        )


def page_settings():
    st.markdown("## Settings")

    st.markdown(
        '<div class="ms-card"><div class="ms-card-title">DETECTION THRESHOLD</div>',
        unsafe_allow_html=True,
    )
    st.session_state.conf_threshold = st.slider(
        "Confidence Threshold", 0.30, 0.99,
        float(st.session_state.conf_threshold), 0.01,
        key="settings_conf_threshold",
    )
    st.caption(f"Current threshold: {st.session_state.conf_threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="ms-card"><div class="ms-card-title">CAMERA</div>',
        unsafe_allow_html=True,
    )
    st.session_state.cam_index = st.number_input(
        "Camera Index", min_value=0, max_value=10,
        value=int(st.session_state.cam_index), step=1,
        key="settings_cam_index",
    )
    st.caption("0 = default camera, 1 = secondary camera.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="ms-card"><div class="ms-card-title">PROJECT FILES</div>',
        unsafe_allow_html=True,
    )
    model_ok   = "✔ Found" if os.path.exists(MODEL_PATH)   else "✘ Missing"
    cascade_ok = "✔ Found" if os.path.exists(CASCADE_PATH) else "✘ Missing"
    logo_ok    = "✔ Found" if os.path.exists(LOGO_PATH)    else "✘ Missing"
    st.code(
        f"MODEL_PATH   = {MODEL_PATH}   [{model_ok}]\n"
        f"CASCADE_PATH = {CASCADE_PATH} [{cascade_ok}]\n"
        f"LOGO_PATH    = {LOGO_PATH}    [{logo_ok}]",
        language="text",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="footer">
            <div style="font-family:var(--font-head);font-size:1.2rem;letter-spacing:.16em;color:#18e7ff;">
                MASKSENSE AI · Face Mask Detection System
            </div>
            <div class="tech-pills" style="margin-top:.75rem;">
                <span>Python</span><span>Streamlit</span><span>TensorFlow</span>
                <span>Keras</span><span>OpenCV</span><span>MobileNetV2</span><span>Plotly</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Live Detection":
        page_live_detection()
    elif page == "Upload Media":
        page_upload_media()
    elif page == "Analytics":
        page_analytics()
    elif page == "AI Model":
        page_model_info()
    elif page == "Settings":
        page_settings()

    render_footer()


if __name__ == "__main__":
    main()