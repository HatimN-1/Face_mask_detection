"""
╔══════════════════════════════════════════════════════════════╗
║             MASKSENSE AI — FACE MASK DETECTION SYSTEM        ║
║             Streamlit UI · MobileNetV2 · OpenCV              ║
╚══════════════════════════════════════════════════════════════╝
Sidebar : icône PNG (base64) à CÔTÉ du st.button — pas dedans.
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
# PAGE CONFIG
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
MASK_COLOR     = (0, 220, 120)
NO_MASK_COLOR  = (45, 45, 255)

# Nav items: label → icon file + fallback emoji
PAGE_OPTIONS = {
    "Dashboard":      {"icon": "assets/dashboard_icon.png",   "emoji": "⌂"},
    "Live Detection": {"icon": "assets/model_small_icon.png", "emoji": "▣"},
    "Upload Media":   {"icon": "assets/upload_icon.png",      "emoji": "▭"},
    "Analytics":      {"icon": "assets/analytics_icon.png",   "emoji": "▥"},
    "AI Model":       {"icon": "assets/ai_model_icon.png",    "emoji": "◌"},
    "Settings":       {"icon": "",                            "emoji": "⚙"},
}


# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def img_to_base64(path: str) -> str:
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
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
    css = {"mask": "log-mask", "nomask": "log-nomask", "info": "log-info"}.get(kind, "log-info")
    st.session_state.log_entries.appendleft(
        f'<span class="log-ts">[{now_ts()}]</span><span class="{css}">{entry}</span>'
    )


def safe_percent(part: int, total: int) -> float:
    return (part / total * 100) if total else 0.0


# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:#060b1c; --bg2:#091327;
    --panel:rgba(7,16,34,.92); --panel2:rgba(8,28,44,.82);
    --line:rgba(0,230,255,.20); --line2:rgba(255,255,255,.12);
    --cyan:#18e7ff; --cyan2:#00bcd4; --green:#19ff9b; --red:#ff3d2e;
    --text:#e7eeff; --muted:#7f93b7; --muted2:#5b6d91;
    --font-head:'Rajdhani',sans-serif;
    --font-body:'Inter',sans-serif;
    --font-mono:'Space Mono',monospace;
}

.stApp {
    background:
        radial-gradient(circle at 52% 13%,rgba(0,238,255,.12),transparent 18%),
        radial-gradient(circle at 15% 100%,rgba(0,255,160,.05),transparent 22%),
        linear-gradient(rgba(0,229,255,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,229,255,.025) 1px,transparent 1px),
        var(--bg);
    background-size:auto,auto,50px 50px,50px 50px,auto;
    color:var(--text);
}
html,body,[class*="css"]{font-family:var(--font-body);color:var(--text);}
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{background:transparent!important;}
.block-container{padding-top:1.8rem!important;padding-left:2rem!important;padding-right:2rem!important;max-width:1550px!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#060b1d 0%,#030716 100%)!important;
    border-right:1px solid rgba(0,229,255,.16);
    box-shadow:18px 0 50px rgba(0,0,0,.20);
    min-width:270px!important; width:270px!important;
}
[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem;padding-left:.4rem;padding-right:.4rem;}
[data-testid="stSidebar"] *{color:var(--text)!important;}

h1,h2,h3{font-family:var(--font-head)!important;letter-spacing:.06em;}

/* ── Sidebar logo ── */
.sidebar-logo-box{text-align:center;padding:1.25rem 0 1.05rem;}
.ms-logo-img{display:block;margin:0 auto;border-radius:17px;filter:drop-shadow(0 0 24px rgba(24,231,255,.45));}
.logo-fallback{margin:0 auto;display:flex;align-items:center;justify-content:center;
    background:rgba(4,13,28,.8);border:1px solid rgba(24,231,255,.35);border-radius:18px;
    font-family:var(--font-head);font-size:64px;font-weight:700;color:var(--cyan);
    filter:drop-shadow(0 0 24px rgba(24,231,255,.45));}
.sidebar-title{margin-top:.9rem;font-family:var(--font-head);font-size:1.75rem;font-weight:700;
    letter-spacing:.18em;color:#f4f8ff;text-shadow:0 0 16px rgba(24,231,255,.55);}
.sidebar-subtitle{margin-top:.3rem;font-family:var(--font-mono);font-size:.72rem;letter-spacing:.16em;color:#dbe5ff;}
.sidebar-sep{height:1px;background:linear-gradient(90deg,transparent,rgba(24,231,255,.25),transparent);margin:1.1rem 0;}
.sidebar-section-title{font-family:var(--font-head);font-size:1.1rem;font-weight:700;
    letter-spacing:.08em;margin:.3rem 0 .5rem .2rem;color:#dbe5ff;}

/* ── Nav icon cell ── */
.nav-icon-cell{
    display:flex; align-items:center; justify-content:center;
    height:40px; padding-top:1px;
}
.nav-icon-cell img{
    width:24px; height:24px; object-fit:contain;
    filter:brightness(0) invert(1);
    opacity:.75;
    transition:all .22s ease;
}
.nav-icon-cell.active img{
    filter:brightness(0) saturate(100%) invert(79%) sepia(100%)
           saturate(400%) hue-rotate(155deg) brightness(105%)
           drop-shadow(0 0 5px rgba(24,231,255,.9));
    opacity:1;
}
.nav-icon-cell .nav-emoji{font-size:1.25rem;line-height:1;opacity:.75;}
.nav-icon-cell.active .nav-emoji{color:var(--cyan)!important;opacity:1;text-shadow:0 0 8px rgba(24,231,255,.7);}

/* ── Nav buttons — override default Streamlit button in sidebar ── */
[data-testid="stSidebar"] .stButton>button{
    background:transparent!important;
    border:1px solid transparent!important;
    color:#cdd8f0!important;
    text-align:left!important;
    justify-content:flex-start!important;
    font-family:var(--font-body)!important;
    font-size:.97rem!important;
    font-weight:500!important;
    letter-spacing:0!important;
    min-height:40px!important;
    height:40px!important;
    padding:6px 10px!important;
    border-radius:10px!important;
    width:100%!important;
    transition:all .22s ease!important;
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,255,255,.05)!important;
    border-color:rgba(255,255,255,.10)!important;
    color:#e7eeff!important;
    transform:none!important;
    box-shadow:none!important;
}
/* Active nav button — injected via .nav-active-btn wrapper */
.nav-active-btn .stButton>button{
    background:linear-gradient(135deg,rgba(24,231,255,.15),rgba(25,255,155,.05))!important;
    border-color:rgba(24,231,255,.32)!important;
    color:#e7eeff!important;
    box-shadow:inset 0 0 20px rgba(24,231,255,.06)!important;
}

/* ── Bottom action cards ── */
.side-action-card{
    display:flex; align-items:center; gap:.9rem;
    padding:.85rem 1rem; margin:.4rem 0;
    border:1px solid rgba(255,255,255,.13); border-radius:13px;
    background:linear-gradient(135deg,rgba(10,26,46,.82),rgba(4,11,25,.84));
    box-shadow:0 0 0 1px rgba(0,0,0,.22),inset 0 0 25px rgba(24,231,255,.025);
    transition:all .22s ease;
}
.side-action-card:hover{border-color:rgba(24,231,255,.28);box-shadow:0 0 18px rgba(24,231,255,.08);}
.side-action-icon{
    width:40px; height:40px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    background:rgba(11,28,46,.88); border:1px solid rgba(24,231,255,.13);
    font-size:1.1rem;
}
.side-action-icon img{width:20px;height:20px;object-fit:contain;filter:brightness(0) invert(1);opacity:.85;}
.side-action-main{flex:1;}
.side-action-title{font-family:var(--font-head);font-size:1.1rem;letter-spacing:.1em;color:var(--cyan);font-weight:650;}
.side-action-desc{margin-top:.25rem;font-size:.8rem;line-height:1.5;color:rgba(220,232,255,.68);}
.side-action-arrow{font-size:1.5rem;color:rgba(220,232,255,.7);}

/* ── Hero ── */
.hero-section{text-align:center;padding:1.2rem 0 1.8rem;}
.hero-logo-wrap img,.hero-logo-wrap .logo-fallback{max-width:290px;border-radius:20px;filter:drop-shadow(0 0 36px rgba(24,231,255,.34));}
.hero-title{margin-top:1.35rem;font-family:var(--font-head);font-size:3.8rem;line-height:1;font-weight:700;
    letter-spacing:.18em;background:linear-gradient(90deg,#0ff,#19ff9b);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero-subtitle{margin-top:1.35rem;font-family:var(--font-mono);color:#5e80bd;letter-spacing:.28em;font-size:1rem;}
.hero-divider{height:1px;width:54%;margin:2.1rem auto 0;
    background:linear-gradient(90deg,transparent,rgba(24,231,255,.48),rgba(25,255,155,.36),transparent);}

/* ── Cards ── */
.ms-card,.metric-card,.feature-card{
    background:linear-gradient(135deg,rgba(6,30,43,.88),rgba(4,13,27,.92));
    border:1px solid rgba(0,229,255,.22); border-radius:15px;
    box-shadow:inset 0 0 28px rgba(24,231,255,.025),0 10px 30px rgba(0,0,0,.16);
}
.metric-card{min-height:168px;display:flex;flex-direction:column;align-items:center;justify-content:center;transition:.22s ease;overflow:hidden;position:relative;}
.metric-card:hover{transform:translateY(-3px);border-color:rgba(24,231,255,.42);box-shadow:0 0 30px rgba(24,231,255,.12);}
.metric-icon{font-size:2.15rem;line-height:1;color:#f1f6ff;margin-bottom:.72rem;}
.metric-icon.red{color:var(--red);text-shadow:0 0 12px rgba(255,61,46,.55);font-size:2.6rem;}
.metric-value{font-family:var(--font-mono);font-weight:700;font-size:2.55rem;line-height:1;color:var(--cyan);text-shadow:0 0 18px rgba(24,231,255,.75);}
.metric-label{margin-top:1rem;font-family:var(--font-mono);color:#6986b7;font-size:.78rem;text-transform:uppercase;letter-spacing:.13em;}
.ms-card{padding:1.35rem 1.45rem;margin-bottom:1rem;}
.ms-card-title{font-family:var(--font-head);color:var(--cyan);font-size:1.65rem;font-weight:650;letter-spacing:.11em;margin-bottom:.55rem;}
.ms-muted{color:var(--muted);line-height:1.8;}
.feature-card{min-height:176px;padding:1.55rem 1.65rem;display:flex;align-items:center;gap:1.6rem;margin:1.25rem 0;}
.feature-icon{width:76px;height:76px;border-radius:50%;background:rgba(13,30,48,.85);border:1px solid rgba(24,231,255,.15);display:flex;align-items:center;justify-content:center;font-size:2rem;color:#f3f6ff;}
.feature-body{flex:1;}
.feature-title{font-family:var(--font-head);font-size:1.72rem;letter-spacing:.15em;color:var(--cyan);font-weight:650;}
.feature-desc{margin-top:.75rem;color:rgba(235,240,255,.82);font-size:1.02rem;line-height:1.65;}
.feature-arrow{font-size:3rem;color:rgba(235,240,255,.85);}

/* Alerts */
.alert-warning{padding:1rem 1.2rem;background:rgba(255,61,46,.13);border:1px solid rgba(255,61,46,.55);border-left:5px solid var(--red);border-radius:12px;color:#ff8c83;font-family:var(--font-mono);}
.alert-success{padding:1rem 1.2rem;background:rgba(25,255,155,.08);border:1px solid rgba(25,255,155,.35);border-left:5px solid var(--green);border-radius:12px;color:#78ffc5;font-family:var(--font-mono);}
.badge-mask,.badge-nomask,.badge-idle{border-radius:999px;padding:.28rem .78rem;font-family:var(--font-mono);font-size:.86rem;}
.badge-mask{background:rgba(25,255,155,.11);border:1px solid rgba(25,255,155,.45);color:var(--green);}
.badge-nomask{background:rgba(255,61,46,.12);border:1px solid rgba(255,61,46,.52);color:#ff837b;}
.badge-idle{background:rgba(127,147,183,.10);border:1px solid rgba(127,147,183,.32);color:#a9b8d3;}

/* Log */
.log-feed{background:rgba(1,7,18,.68);border:1px solid rgba(24,231,255,.13);border-radius:13px;padding:.9rem 1rem;font-family:var(--font-mono);font-size:.78rem;color:var(--muted);height:170px;overflow-y:auto;line-height:1.75;}
.log-ts{color:#445a7e;margin-right:.35rem;}
.log-mask{color:var(--green);} .log-nomask{color:#ff766d;} .log-info{color:var(--cyan);}

/* Main content buttons (non-sidebar) */
.block-container .stButton>button,.stDownloadButton>button{
    background:linear-gradient(135deg,rgba(24,231,255,.13),rgba(25,255,155,.07))!important;
    border:1px solid rgba(24,231,255,.35)!important; color:var(--cyan)!important;
    border-radius:11px!important; font-family:var(--font-head)!important;
    font-weight:700!important; letter-spacing:.08em!important;
    min-height:44px!important; transition:all .2s ease!important;
}
.block-container .stButton>button:hover{border-color:rgba(24,231,255,.7)!important;box-shadow:0 0 22px rgba(24,231,255,.16)!important;transform:translateY(-1px);}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--cyan),var(--green))!important;}
.streamlit-expanderHeader{background:rgba(8,19,38,.72)!important;border:1px solid rgba(255,255,255,.10)!important;border-radius:12px!important;}
.footer{margin-top:2.2rem;padding:1.8rem 0;text-align:center;border-top:1px solid rgba(24,231,255,.13);color:#526b93;font-family:var(--font-mono);font-size:.8rem;}
.tech-pills span{display:inline-block;margin:.25rem;padding:.24rem .72rem;border-radius:999px;border:1px solid rgba(24,231,255,.18);background:rgba(24,231,255,.055);color:#74eaff;}

@media(max-width:900px){
    [data-testid="stSidebar"]{min-width:240px!important;width:240px!important;}
    .hero-title{font-size:2.5rem;} .hero-divider{width:90%;}
    .feature-card{flex-direction:column;align-items:flex-start;}
}
</style>
"""


# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
def init_session() -> None:
    defaults = {
        "page":           "Dashboard",
        "total_frames":   0,
        "total_faces":    0,
        "mask_count":     0,
        "no_mask_count":  0,
        "fps":            0.0,
        "last_label":     "—",
        "last_conf":      0.0,
        "log_entries":    deque(maxlen=HISTORY_MAXLEN),
        "session_start":  datetime.datetime.now(),
        "conf_threshold": 0.60,
        "cam_index":      0,
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
        return None, "TensorFlow is not installed."
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        return keras.models.load_model(path, compile=False), None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_cascade(path: str):
    if not os.path.exists(path):
        return None, f"Cascade file not found: {path}"
    c = cv2.CascadeClassifier(path)
    return (None, "Failed to load Haar Cascade.") if c.empty() else (c, None)


# ──────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────
def detect_and_predict(frame, cascade, model, conf_threshold=0.5):
    results = []
    if cascade is None or model is None:
        return frame, results

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_rgb  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_arr  = np.expand_dims(cv2.resize(face_rgb, IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        pred      = np.array(model.predict(face_arr, verbose=0)).flatten()

        if pred.size == 1:
            no_mask_prob = float(pred[0]); mask_prob = 1.0 - no_mask_prob
        else:
            mask_prob = float(pred[0]); no_mask_prob = float(pred[1])

        if mask_prob >= no_mask_prob:
            label, conf, color = "Mask",    mask_prob,    MASK_COLOR
        else:
            label, conf, color = "No Mask", no_mask_prob, NO_MASK_COLOR

        if conf < conf_threshold:
            label = "Low Confidence"

        results.append({"label": label, "conf": conf, "bbox": (x,y,w,h),
                         "mask_prob": mask_prob, "no_mask_prob": no_mask_prob})

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        l, t = 22, 3
        for pt1, pt2 in [
            ((x,y),(x+l,y)),((x,y),(x,y+l)),
            ((x+w,y),(x+w-l,y)),((x+w,y),(x+w,y+l)),
            ((x,y+h),(x+l,y+h)),((x,y+h),(x,y+h-l)),
            ((x+w,y+h),(x+w-l,y+h)),((x+w,y+h),(x+w,y+h-l)),
        ]:
            cv2.line(frame, pt1, pt2, color, t)

        text        = f"{label} {conf*100:.1f}%"
        (tw,th), _  = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        y_text      = max(28, y-8)
        cv2.rectangle(frame, (x,y_text-th-12), (x+tw+14,y_text+4), color, -1)
        cv2.putText(frame, text, (x+7,y_text-4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 2)

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
    cls = "metric-icon red" if red else "metric-icon"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="{cls}">{icon}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_log_feed(max_items: int = 40) -> None:
    entries = "<br>".join(list(st.session_state.log_entries)[:max_items])
    if not entries:
        entries = '<span class="log-ts">[--:--:--]</span><span class="log-info">Waiting for detections...</span>'
    st.markdown(f'<div class="log-feed">{entries}</div>', unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-body">'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-desc">{desc}</div>'
        f'</div><div class="feature-arrow">›</div></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# SIDEBAR — icon (markdown) + button (st.button) côte à côte
# ──────────────────────────────────────────────────────────────
def _icon_html(icon_path: str, emoji: str, active: bool) -> str:
    """Returns HTML for the nav icon cell."""
    active_cls = "nav-icon-cell active" if active else "nav-icon-cell"
    b64 = img_to_base64(icon_path)
    if b64:
        inner = f'<img src="data:image/png;base64,{b64}" alt="">'
    else:
        inner = f'<span class="nav-emoji">{emoji}</span>'
    return f'<div class="{active_cls}">{inner}</div>'


def render_sidebar() -> str:
    with st.sidebar:

        # ── Logo + branding ──────────────────────────────────
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

        # ── Nav rows : icon column + button column ────────────
        for page_name, data in PAGE_OPTIONS.items():
            is_active = st.session_state.page == page_name

            # Wrap button in active class when needed
            if is_active:
                st.markdown('<div class="nav-active-btn">', unsafe_allow_html=True)

            col_icon, col_btn = st.columns([1, 4])

            with col_icon:
                st.markdown(
                    _icon_html(data["icon"], data["emoji"], is_active),
                    unsafe_allow_html=True,
                )

            with col_btn:
                if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.page = page_name
                    st.rerun()

            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Separator ─────────────────────────────────────────
        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # ── Model Info / About cards ──────────────────────────
        ai_b64 = img_to_base64("assets/ai_model_icon.png")
        mi_icon = (
            f'<img src="data:image/png;base64,{ai_b64}">'
            if ai_b64 else "♟"
        )
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
                <div class="side-action-icon"
                     style="font-family:Georgia,serif;font-style:italic;
                            font-size:1.2rem;color:#18e7ff;">i</div>
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

        # ── Quick controls ────────────────────────────────────
        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        st.session_state.conf_threshold = st.slider(
            "Confidence Threshold", 0.30, 0.99,
            float(st.session_state.conf_threshold), 0.01,
        )
        st.session_state.cam_index = st.number_input(
            "Camera Index", min_value=0, max_value=10,
            value=int(st.session_state.cam_index), step=1,
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

    c1,c2,c3,c4,c5 = st.columns(5)
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
            """, unsafe_allow_html=True,
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
    model, model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err = load_cascade(CASCADE_PATH)

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
        """, unsafe_allow_html=True,
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
            """, unsafe_allow_html=True,
        )
        return

    img_bgr = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Cannot read the image.")
        return

    start = time.perf_counter()
    annotated, results = detect_and_predict(img_bgr.copy(), cascade, model, st.session_state.conf_threshold)
    elapsed = max(time.perf_counter() - start, 1e-6)
    st.session_state.total_frames += 1
    st.session_state.fps = 1.0 / elapsed
    update_stats_from_results(results)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
    with col2:
        total, masks, no_masks = len(results), sum(1 for r in results if r["label"]=="Mask"), sum(1 for r in results if r["label"]=="No Mask")
        a, b, c = st.columns(3)
        with a: render_metric("○", str(total),    "Faces")
        with b: render_metric("◎", str(masks),    "With Mask")
        with c: render_metric("×", str(no_masks), "No Mask", red=True)

        st.markdown('<div class="ms-card"><div class="ms-card-title">RESULTS</div>', unsafe_allow_html=True)
        if not results:
            st.warning("No face detected.")
        else:
            for i, r in enumerate(results, 1):
                badge = "badge-mask" if r["label"]=="Mask" else "badge-nomask"
                st.markdown(f'Face {i}: <span class="{badge}">{r["label"]}</span> — <b>{r["conf"]*100:.1f}%</b>', unsafe_allow_html=True)
            if no_masks > 0:
                st.markdown('<div class="alert-warning">NO MASK DETECTED — Please wear a face mask.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success">COMPLIANT — All detected faces wearing masks.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_upload_media():
    st.markdown("## Upload Media")
    model, model_err = load_keras_model(MODEL_PATH)
    cascade, cas_err = load_cascade(CASCADE_PATH)

    if model_err or cas_err:
        st.error("Model or Cascade not loaded.")
        if model_err: st.warning(model_err)
        if cas_err:   st.warning(cas_err)
        return

    tab_img, tab_vid = st.tabs(["Image", "Video"])

    with tab_img:
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"], key="image_uploader")
        if uploaded:
            img_bgr = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Cannot read this image."); return

            start = time.perf_counter()
            annotated, results = detect_and_predict(img_bgr.copy(), cascade, model, st.session_state.conf_threshold)
            st.session_state.total_frames += 1
            st.session_state.fps = 1.0 / max(time.perf_counter()-start, 1e-6)
            update_stats_from_results(results)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Original")
                st.image(cv2.cvtColor(img_bgr,   cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_b:
                st.markdown("### Detected")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown('<div class="ms-card"><div class="ms-card-title">DETECTION RESULTS</div>', unsafe_allow_html=True)
            if results:
                for i, r in enumerate(results, 1):
                    badge = "badge-mask" if r["label"]=="Mask" else "badge-nomask"
                    st.markdown(f'Face {i}: <span class="{badge}">{r["label"]}</span> — confidence <b>{r["conf"]*100:.1f}%</b>', unsafe_allow_html=True)
            else:
                st.info("No faces detected in the image.")
            st.markdown("</div>", unsafe_allow_html=True)

            ok, buf = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("Download Annotated Image", buf.tobytes(), "masksense_detected.png", "image/png")

    with tab_vid:
        st.info("Video processing available. Keep videos short on Streamlit Cloud.")
        vid_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"], key="video_uploader")
        if vid_file:
            tmp = "/tmp/masksense_in.mp4"; out_p = "/tmp/masksense_out.avi"
            with open(tmp, "wb") as f: f.write(vid_file.read())
            cap  = cv2.VideoCapture(tmp)
            n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_ = cap.get(cv2.CAP_PROP_FPS) or 25
            W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out  = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"XVID"), fps_, (W, H))
            prog = st.progress(0); stat = st.empty(); prev = st.empty(); idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                ann, res = detect_and_predict(frame.copy(), cascade, model, st.session_state.conf_threshold)
                out.write(ann); idx += 1; st.session_state.total_frames += 1
                update_stats_from_results(res)
                prog.progress(min(idx/n, 1.0)); stat.markdown(f"Frame **{idx}** / {n}")
                if idx % 20 == 0:
                    prev.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)
            cap.release(); out.release()
            with open(out_p, "rb") as f:
                st.download_button("Download Processed Video", f.read(), "masksense_processed.avi", "video/avi")
            st.success("Video processing complete.")


def page_analytics():
    st.markdown("## Analytics")
    total = st.session_state.mask_count + st.session_state.no_mask_count
    compliance = safe_percent(st.session_state.mask_count, total)

    c1,c2,c3,c4 = st.columns(4)
    with c1: render_metric("○",  str(st.session_state.total_faces),   "Total Faces")
    with c2: render_metric("◎",  str(st.session_state.mask_count),    "With Mask")
    with c3: render_metric("×",  str(st.session_state.no_mask_count), "No Mask", red=True)
    with c4: render_metric("↗",  f"{compliance:.1f}%",                "Compliance Rate")

    st.markdown("<br>", unsafe_allow_html=True)
    if total == 0:
        st.info("No detections yet. Use Live Detection or Upload Media."); return
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly: pip install plotly"); return

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=["With Mask","No Mask"],
            values=[st.session_state.mask_count, st.session_state.no_mask_count],
            hole=0.56, marker=dict(colors=["#19ff9b","#ff3d2e"]), textinfo="label+percent",
        )])
        fig.update_layout(title="Detection Distribution", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eeff"), margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=["With Mask","No Mask"],
            y=[st.session_state.mask_count, st.session_state.no_mask_count],
            text=[st.session_state.mask_count, st.session_state.no_mask_count],
            textposition="outside", marker=dict(color=["#19ff9b","#ff3d2e"]),
        )])
        fig2.update_layout(title="Count Comparison", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eeff"),
            xaxis=dict(gridcolor="rgba(24,231,255,.08)"), yaxis=dict(gridcolor="rgba(24,231,255,.08)"),
            margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number", value=compliance,
        number=dict(suffix="%", font=dict(color="#18e7ff", size=42)),
        title=dict(text="Compliance Rate", font=dict(color="#18e7ff")),
        gauge=dict(axis=dict(range=[0,100], tickfont=dict(color="#e7eeff")),
            bar=dict(color="#18e7ff"), bgcolor="rgba(8,28,44,.6)",
            borderwidth=1, bordercolor="rgba(24,231,255,.25)",
            steps=[dict(range=[0,60],color="rgba(255,61,46,.18)"),
                   dict(range=[60,80],color="rgba(255,215,0,.16)"),
                   dict(range=[80,100],color="rgba(25,255,155,.16)")]),
    ))
    fig3.update_layout(height=310, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e7eeff"), margin=dict(l=20,r=20,t=35,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    if st.button("Reset Statistics"):
        for k in ["total_frames","total_faces","mask_count","no_mask_count"]:
            st.session_state[k] = 0
        st.session_state.fps=0.0; st.session_state.last_label="—"
        st.session_state.last_conf=0.0; st.session_state.log_entries.clear()
        st.rerun()


def page_model_info():
    st.markdown("## AI Model")
    model, model_err = load_keras_model(MODEL_PATH)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="ms-card"><div class="ms-card-title">ARCHITECTURE</div>'
            '<div class="ms-muted"><b>Base Model:</b> MobileNetV2<br>'
            '<b>Task:</b> Binary Classification<br><b>Input Shape:</b> 224×224×3<br>'
            '<b>Classes:</b> Mask / No Mask<br><b>Framework:</b> TensorFlow / Keras</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="ms-card"><div class="ms-card-title">FACE DETECTION</div>'
            '<div class="ms-muted"><b>Method:</b> Haar Cascade<br>'
            '<b>Classifier:</b> FrontalFace Default<br><b>Library:</b> OpenCV<br>'
            '<b>ROI:</b> Face crop → Resize → Normalize → Prediction</div></div>',
            unsafe_allow_html=True)

    if model is not None and TF_AVAILABLE:
        st.markdown('<div class="ms-card"><div class="ms-card-title">MODEL SUMMARY</div>', unsafe_allow_html=True)
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x+"\n"))
        st.code(buf.getvalue(), language="text")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"Load model at `{MODEL_PATH}` to see model summary.")
        if model_err: st.warning(model_err)

    with st.expander("How the detection pipeline works"):
        st.markdown("""
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
```""")


def page_settings():
    st.markdown("## Settings")
    st.markdown('<div class="ms-card"><div class="ms-card-title">DETECTION THRESHOLD</div>', unsafe_allow_html=True)
    st.session_state.conf_threshold = st.slider(
        "Confidence Threshold", 0.30, 0.99, float(st.session_state.conf_threshold), 0.01, key="s_thresh")
    st.caption(f"Current threshold: {st.session_state.conf_threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ms-card"><div class="ms-card-title">CAMERA</div>', unsafe_allow_html=True)
    st.session_state.cam_index = st.number_input(
        "Camera Index", min_value=0, max_value=10, value=int(st.session_state.cam_index), step=1, key="s_cam")
    st.caption("0 = default camera, 1 = secondary camera.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ms-card"><div class="ms-card-title">PROJECT FILES</div>', unsafe_allow_html=True)
    st.code(
        f"MODEL_PATH   = {MODEL_PATH}   [{'✔ Found' if os.path.exists(MODEL_PATH)   else '✘ Missing'}]\n"
        f"CASCADE_PATH = {CASCADE_PATH} [{'✔ Found' if os.path.exists(CASCADE_PATH) else '✘ Missing'}]\n"
        f"LOGO_PATH    = {LOGO_PATH}    [{'✔ Found' if os.path.exists(LOGO_PATH)    else '✘ Missing'}]",
        language="text")
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

    if   page == "Dashboard":      page_dashboard()
    elif page == "Live Detection":  page_live_detection()
    elif page == "Upload Media":    page_upload_media()
    elif page == "Analytics":       page_analytics()
    elif page == "AI Model":        page_model_info()
    elif page == "Settings":        page_settings()

    render_footer()


if __name__ == "__main__":
    main()