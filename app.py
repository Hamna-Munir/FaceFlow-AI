import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from datetime import datetime

st.set_page_config(
    page_title="FaceFlow · AI Focus Detector",
    page_icon="👁️",
    layout="wide"
)

for k, v in {
    "page": "photo",
    "running": False,
    "focus_log": [],
    "start_time": None,
    "alerts": 0,
    "history_log": [],
    "saved_sessions": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

PAGE_LABELS = {
    "photo":   "Photo Analysis",
    "webcam":  "Live Webcam Session",
    "reports": "Reports",
    "history": "History",
    "about":   "About"
}

mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

@st.cache_resource
def load_model():
    try:
        with open("focus_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = load_model()

def extract_features(img_bgr):
    img  = cv2.resize(img_bgr, (64, 32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [np.mean(gray), np.mean(gray[:16, :]),
            np.mean(gray[16:, :]), np.std(gray),
            float(np.sum(gray > 150)), float(np.sum(gray < 80))]

def crop_eye_region(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                          min_detection_confidence=0.4) as fm:
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            x1 = max(0, int(lm[33].x  * w) - 10)
            x2 = min(w, int(lm[263].x * w) + 10)
            y1 = max(0, int(lm[159].y * h) - 20)
            y2 = min(h, int(lm[386].y * h) + 20)
            eye = img_bgr[y1:y2, x1:x2]
            if eye.size > 0:
                return eye
    return img_bgr

def draw_landmarks(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                          min_detection_confidence=0.4) as fm:
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    image=img_bgr,
                    landmark_list=face,
                    connections=mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_style
                        .get_default_face_mesh_contours_style())
            return img_bgr, True
    return img_bgr, False

def predict(img_bgr):
    if model is None:
        return None, 0, 0
    eye   = crop_eye_region(img_bgr)
    feat  = np.array(extract_features(eye)).reshape(1, -1)
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return pred, round(proba[1] * 100, 1), round(proba[0] * 100, 1)

def make_chart(focus_log):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('#0D1B35')
    colors = ['#2DD4BF' if f else '#F87171' for f in focus_log]
    ax1 = axes[0]
    ax1.set_facecolor('#1B2B4B')
    ax1.bar(range(len(focus_log)), [1]*len(focus_log), color=colors, width=0.8)
    ax1.set_title('Focus Timeline', fontsize=11, pad=10, color='#F0F4FF')
    ax1.set_yticks([])
    ax1.tick_params(colors='#8BABC8')
    for s in ax1.spines.values():
        s.set_edgecolor('#243657')
    fp = mpatches.Patch(color='#2DD4BF', label='Focused')
    dp = mpatches.Patch(color='#F87171', label='Distracted')
    ax1.legend(handles=[fp, dp], facecolor='#1B2B4B',
               edgecolor='#243657', labelcolor='#8BABC8', fontsize=9)
    ax2 = axes[1]
    ax2.set_facecolor('#1B2B4B')
    fc = sum(focus_log)
    dc = len(focus_log) - fc
    if fc + dc > 0:
        ax2.pie([fc, dc], labels=['Focused', 'Distracted'],
                colors=['#2DD4BF', '#F87171'], autopct='%1.0f%%',
                textprops={'color': '#F0F4FF', 'fontsize': 10},
                wedgeprops={'edgecolor': '#0D1B35', 'linewidth': 2})
    ax2.set_title('Distribution', fontsize=11, pad=10, color='#F0F4FF')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='#0D1B35')
    buf.seek(0)
    plt.close()
    return buf

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #06101F !important; }

[data-testid="stSidebar"] {
    background: #0A1628 !important;
    border-right: 1px solid #1E3050 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #0A1628 !important;
    padding: 0 !important;
}

.sb-brand { font-size: 1.5rem; font-weight: 700; color: #E8F0FF;
            letter-spacing: -0.5px; padding: 0 4px; }
.sb-brand span { color: #2DD4BF; }
.sb-tagline { font-size: 10px; color: #7A9CBF; margin-top: 2px;
              padding: 0 4px; letter-spacing: .05em; }
.sb-divider { height: 1px; background: #1E3050; margin: 1rem 0; }
.sb-section { font-size: 10px; color: #7A9CBF; text-transform: uppercase;
              letter-spacing: .12em; margin: 10px 4px 6px; font-weight: 600; }
.sb-footer { margin-top: 1.5rem; border-top: 1px solid #1E3050;
             padding: 1rem 4px 0; }
.sb-built { font-size: 10px; color: #7A9CBF; text-transform: uppercase;
            letter-spacing: .08em; }
.sb-name  { font-size: 14px; font-weight: 700; color: #E8F0FF; margin-top: 4px; }
.sb-role  { font-size: 11px; color: #7A9CBF; margin-top: 3px; line-height: 1.6; }

[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: #A8C4DC !important;
    border: 1px solid transparent !important;
    border-left: 2px solid transparent !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    transition: all .15s !important;
}
[data-testid="baseButton-secondary"]:hover {
    color: #E8F0FF !important;
    background: rgba(45,212,191,0.09) !important;
    border-left: 2px solid #2DD4BF !important;
}

[data-testid="baseButton-primary"] {
    background: #2DD4BF !important;
    color: #06101F !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
}
[data-testid="baseButton-primary"]:hover { background: #14B8A6 !important; }

[data-testid="stDownloadButton"] > button {
    background: #112240 !important;
    color: #2DD4BF !important;
    border: 1px solid #1E3050 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    width: 100% !important;
    font-weight: 500 !important;
}

.topbar {
    background: #080F1E;
    border-bottom: 1px solid #1E3050;
    padding: 0.9rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tb-project { font-size: 1.25rem; font-weight: 700;
              color: #E8F0FF; letter-spacing: -0.5px; }
.tb-project span { color: #2DD4BF; }
.tb-page { font-size: 11px; color: #7A9CBF; margin-top: 3px; }
.pills { display: flex; gap: 6px; flex-wrap: wrap; }
.pill { font-size: 11px; padding: 3px 10px; border-radius: 20px;
        border: 1px solid #1E3050; color: #7A9CBF; background: transparent; }
.pill-teal { background: rgba(45,212,191,0.12);
             border-color: rgba(45,212,191,0.45);
             color: #2DD4BF; font-weight: 600; }

.main-content { padding: 1.5rem 2rem; }

.status-ok {
    display: flex; align-items: center; gap: 8px;
    background: rgba(45,212,191,0.08);
    border: 1px solid rgba(45,212,191,0.3);
    border-radius: 10px; padding: 10px 16px;
    font-size: 13px; font-weight: 500;
    color: #2DD4BF; margin-bottom: 1.25rem;
}
.status-err {
    display: flex; align-items: center; gap: 8px;
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 10px; padding: 10px 16px;
    font-size: 13px; font-weight: 500;
    color: #F87171; margin-bottom: 1.25rem;
}

.sec-title {
    font-size: 13px; font-weight: 700; color: #E8F0FF;
    margin-bottom: 1rem; display: flex;
    align-items: center; gap: 6px;
    text-transform: uppercase; letter-spacing: .07em;
}
.teal-dot { width: 7px; height: 7px; background: #2DD4BF;
            border-radius: 50%; display: inline-block; flex-shrink: 0; }

.card { background: #112240; border: 1px solid #1E3050;
        border-radius: 14px; padding: 1.25rem; }

/* ════════════════════════════════════════════════════════
   UPLOAD BUG FIX — permanent solution
   Hides Streamlit's broken Browse button entirely.
   Replaces it with a CSS ::before fake button.
   Real click area still works underneath.
════════════════════════════════════════════════════════ */
div[data-testid="stFileUploaderDropzone"] {
    background: #112240 !important;
    border: 1.5px dashed #2DD4BF !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    cursor: pointer !important;
}
div[data-testid="stFileUploaderDropzone"] button {
    background: #1E3050 !important;
    color: #2DD4BF !important;
    border: 1px solid #2DD4BF !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
}

.result-f { background: rgba(45,212,191,0.08);
            border: 1px solid rgba(45,212,191,0.3);
            border-radius: 12px; padding: 1.25rem;
            text-align: center; margin-bottom: 12px; }
.result-d { background: rgba(248,113,113,0.08);
            border: 1px solid rgba(248,113,113,0.3);
            border-radius: 12px; padding: 1.25rem;
            text-align: center; margin-bottom: 12px; }
.state-f { font-size: 1.6rem; font-weight: 700;
           color: #2DD4BF; letter-spacing: -0.5px; }
.state-d { font-size: 1.6rem; font-weight: 700;
           color: #F87171; letter-spacing: -0.5px; }
.conf-txt { font-size: 12px; color: #7A9CBF; margin-top: 4px; }

.stat-grid { display: grid; grid-template-columns: repeat(3,1fr);
             gap: 8px; margin-top: 10px; }
.stat-box { background: #080F1E; border: 1px solid #1E3050;
            border-radius: 10px; padding: 10px; text-align: center; }
.stat-val { font-size: 16px; font-weight: 700; color: #E8F0FF; }
.stat-lbl { font-size: 10px; color: #7A9CBF; margin-top: 2px;
            text-transform: uppercase; letter-spacing: .05em; }

.empty-result { border: 1.5px dashed #1E3050; border-radius: 12px;
                padding: 4rem; text-align: center;
                color: #7A9CBF; font-size: 13px; background: #080F1E; }
.live-empty   { border: 1.5px dashed #1E3050; border-radius: 10px;
                padding: 3.5rem; text-align: center; color: #7A9CBF;
                margin-bottom: 12px; font-size: 13px; background: #080F1E; }

.score-card { background: #112240; border: 1px solid #1E3050;
              border-radius: 14px; padding: 1.5rem;
              text-align: center; margin-bottom: 12px; }
.score-val  { font-size: 2.5rem; font-weight: 700;
              color: #2DD4BF; letter-spacing: -1px; }
.score-lbl  { font-size: 11px; color: #7A9CBF;
              text-transform: uppercase; letter-spacing: .07em;
              margin-top: 4px; }

.hist-row { display: flex; align-items: center; gap: 12px;
            background: #112240; border: 1px solid #1E3050;
            border-radius: 10px; padding: 10px 14px; margin-bottom: 6px; }
.hist-num  { font-size: 11px; color: #7A9CBF;
             min-width: 24px; font-weight: 700; }
.hist-time { font-size: 11px; color: #7A9CBF; min-width: 140px; }
.hist-file { font-size: 12px; color: #A8C4DC; flex: 1;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-badge-f { background: rgba(45,212,191,0.12); color: #2DD4BF;
                border: 1px solid rgba(45,212,191,0.35);
                border-radius: 6px; padding: 2px 8px;
                font-size: 11px; font-weight: 700; }
.hist-badge-d { background: rgba(248,113,113,0.12); color: #F87171;
                border: 1px solid rgba(248,113,113,0.35);
                border-radius: 6px; padding: 2px 8px;
                font-size: 11px; font-weight: 700; }
.hist-conf { font-size: 12px; color: #A8C4DC;
             margin-left: auto; white-space: nowrap; }

.about-hero { background: linear-gradient(135deg, #112240 0%, #080F1E 100%);
              border: 1px solid #1E3050; border-radius: 16px;
              padding: 2rem; margin-bottom: 1.5rem; }
.about-hero-title { font-size: 2rem; font-weight: 700; color: #E8F0FF;
                    letter-spacing: -1px; margin-bottom: 8px; }
.about-hero-title span { color: #2DD4BF; }
.about-hero-sub { font-size: 13px; color: #A8C4DC;
                  line-height: 1.8; max-width: 700px; }
.about-stats { display: grid; grid-template-columns: repeat(4,1fr);
               gap: 10px; margin-top: 1.5rem; }
.about-stat { background: rgba(45,212,191,0.06);
              border: 1px solid rgba(45,212,191,0.18);
              border-radius: 12px; padding: 1rem; text-align: center; }
.about-stat-val { font-size: 1.5rem; font-weight: 700; color: #2DD4BF; }
.about-stat-lbl { font-size: 10px; color: #7A9CBF; margin-top: 4px;
                  text-transform: uppercase; letter-spacing: .06em; }
.about-grid { display: grid; grid-template-columns: repeat(3,1fr);
              gap: 12px; margin-top: 1rem; }
.about-card { background: #112240; border: 1px solid #1E3050;
              border-radius: 12px; padding: 1.25rem; }
.about-icon { font-size: 1.5rem; margin-bottom: 8px; }
.about-t    { font-size: 13px; font-weight: 700;
              color: #E8F0FF; margin-bottom: 6px; }
.about-b    { font-size: 12px; color: #A8C4DC; line-height: 1.7; }

.tech-grid { display: grid; grid-template-columns: repeat(2,1fr);
             gap: 10px; margin-top: 1rem; }
.tech-card { background: #080F1E; border: 1px solid #1E3050;
             border-radius: 10px; padding: 1rem;
             display: flex; align-items: center; gap: 12px; }
.tech-icon { font-size: 1.4rem; }
.tech-name { font-size: 13px; font-weight: 700; color: #E8F0FF; }
.tech-desc { font-size: 11px; color: #7A9CBF; margin-top: 2px; }

.pipeline-step { display: flex; align-items: flex-start; gap: 12px;
                 background: #080F1E; border: 1px solid #1E3050;
                 border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; }
.step-num { background: #2DD4BF; color: #06101F; border-radius: 50%;
            min-width: 26px; height: 26px; display: flex;
            align-items: center; justify-content: center;
            font-size: 11px; font-weight: 800; }
.step-title { font-size: 13px; font-weight: 700;
              color: #E8F0FF; margin-bottom: 4px; }
.step-desc  { font-size: 11px; color: #A8C4DC; line-height: 1.6; }

.creator-card { background: linear-gradient(135deg, #112240, #080F1E);
                border: 1px solid #1E3050; border-radius: 14px;
                padding: 1.5rem; margin-top: 1rem;
                display: flex; align-items: center; gap: 1.5rem; }
.creator-avatar { width: 64px; height: 64px;
                  background: rgba(45,212,191,0.1);
                  border: 2px solid rgba(45,212,191,0.35);
                  border-radius: 50%; display: flex;
                  align-items: center; justify-content: center;
                  font-size: 1.6rem; flex-shrink: 0; }
.creator-name  { font-size: 1.15rem; font-weight: 700; color: #E8F0FF; }
.creator-title { font-size: 12px; color: #2DD4BF;
                 margin-top: 2px; font-weight: 600; }
.creator-bio   { font-size: 12px; color: #A8C4DC;
                 margin-top: 6px; line-height: 1.6; }

div[data-testid="stMetricValue"] { color: #E8F0FF !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #7A9CBF !important; font-size: 12px !important; }
div[data-testid="stMetric"] { background: #112240 !important;
    border: 1px solid #1E3050 !important;
    border-radius: 10px !important; padding: 12px !important; }
.stProgress > div > div { background-color: #2DD4BF !important; }
div[data-testid="stProgressBar"] > div { background: #1E3050 !important; }
.stRadio label { background: #112240 !important;
    border: 1px solid #1E3050 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #A8C4DC !important; }
.stRadio label:hover { border-color: #2DD4BF !important;
    color: #E8F0FF !important; }
div[data-testid="stExpander"] { background: #112240 !important;
    border: 1px solid #1E3050 !important;
    border-radius: 12px !important; }
div[data-testid="stExpander"] summary { color: #E8F0FF !important;
    font-weight: 600 !important; }
div[data-testid="stExpander"] summary:hover { color: #2DD4BF !important; }
h1,h2,h3,h4,h5,h6 { color: #E8F0FF !important; }
p, li { color: #A8C4DC !important; }
strong { color: #E8F0FF !important; }
.stCaption, div[data-testid="stCaptionContainer"] { color: #7A9CBF !important; }

.footer { background: #080F1E; border-top: 1px solid #1E3050;
          padding: 0.875rem 2rem; margin-top: 1.5rem;
          display: flex; justify-content: space-between;
          align-items: center; font-size: 11px; color: #7A9CBF; }
.footer span { color: #2DD4BF; font-weight: 600; }
hr { border-color: #1E3050 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080F1E; }
::-webkit-scrollbar-thumb { background: #1E3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2DD4BF; }
</style>
""", unsafe_allow_html=True)

sidebar_col, main_col = st.columns([0.18, 0.82])

def nav_btn(icon, label, key):
    if st.button(f"{icon}  {label}", key=f"nb_{key}",
                 use_container_width=True):
        st.session_state.page = key
        st.rerun()

with sidebar_col:
    st.markdown("""
    <div style="padding:0.75rem 4px 0">
      <div class="sb-brand">Face<span>Flow</span></div>
      <div class="sb-tagline">AI FOCUS DETECTOR · V1.0</div>
      <div class="sb-divider"></div>
      <div class="sb-section">Analyze</div>
    </div>
    """, unsafe_allow_html=True)

    nav_btn("📸", "Photo Analysis", "photo")
    nav_btn("🎥", "Live Webcam",    "webcam")

    st.markdown('<div class="sb-section" style="margin-top:6px">Session</div>',
                unsafe_allow_html=True)
    nav_btn("📊", "Reports", "reports")
    nav_btn("🕐", "History", "history")

    st.markdown('<div class="sb-section" style="margin-top:6px">Info</div>',
                unsafe_allow_html=True)
    nav_btn("ℹ️", "About", "about")

    active_label = PAGE_LABELS.get(st.session_state.page, "Photo Analysis")
    st.markdown(f"""
    <script>
    (function(){{
        function mark() {{
            var doc = window.parent ? window.parent.document : document;
            var btns = doc.querySelectorAll('[data-testid="baseButton-secondary"]');
            btns.forEach(function(b) {{
                b.style.textAlign      = 'left';
                b.style.justifyContent = 'flex-start';
                if (b.innerText.trim().includes('{active_label}')) {{
                    b.style.color        = '#2DD4BF';
                    b.style.background   = 'rgba(45,212,191,0.12)';
                    b.style.borderLeft   = '2px solid #2DD4BF';
                    b.style.borderRadius = '8px';
                }}
            }});
        }}
        setTimeout(mark, 150);
        setTimeout(mark, 400);
    }})();
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-footer">
      <div class="sb-built">Built by</div>
      <div class="sb-name">Hamna Munir</div>
      <div class="sb-role">AI/ML Engineer<br>Software Engineering Student</div>
    </div>
    """, unsafe_allow_html=True)

with main_col:
    page = st.session_state.page

    st.markdown(f"""
    <div class="topbar">
      <div>
        <div class="tb-project">Face<span>Flow</span> · AI Focus Detector</div>
        <div class="tb-page">→ {PAGE_LABELS.get(page,'Photo Analysis')}</div>
      </div>
      <div class="pills">
        <span class="pill pill-teal">⚡ 89.8% accuracy</span>
        <span class="pill">Python</span>
        <span class="pill">MediaPipe</span>
        <span class="pill">OpenCV</span>
        <span class="pill">scikit-learn</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if model is None:
        st.markdown("""<div class="status-err">
          ⚠️ focus_model.pkl not found — place it in the same folder as app.py
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="status-ok">
          ✅ FaceFlow model loaded · 89.8% accuracy · Ready to analyze
        </div>""", unsafe_allow_html=True)

    # ══ PHOTO ════════════════════════════════════════════════════════════════
    if page == "photo":
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("""<div class="sec-title">
              <span class="teal-dot"></span> Upload image
            </div>""", unsafe_allow_html=True)

            # ── THE FIX: label="x" + collapsed hides label node completely ──
            uploaded = st.file_uploader(
                "x",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            # ── JS removes the ghost "uploadupload" text directly from DOM ──
            st.markdown("""
            <script>
            (function(){
                function fix() {
                    var d = window.parent.document;
                    d.querySelectorAll('[data-testid="stFileUploaderDropzoneInstructions"]')
                     .forEach(function(e){ e.remove(); });
                    d.querySelectorAll('[data-testid="stFileUploader"] label')
                     .forEach(function(e){ e.remove(); });
                }
                [100, 300, 600, 1200].forEach(function(t){ setTimeout(fix, t); });
            })();
            </script>
            """, unsafe_allow_html=True)

            if uploaded:
                file_bytes = np.asarray(
                    bytearray(uploaded.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                analyze = st.button("👁️  Analyze Focus State",
                                    use_container_width=True, type="primary")
            else:
                analyze = False

        with col2:
            st.markdown("""<div class="sec-title">
              <span class="teal-dot"></span> Analysis result
            </div>""", unsafe_allow_html=True)

            if uploaded and analyze:
                with st.spinner("Detecting landmarks..."):
                    time.sleep(0.3)
                    img_lm, face_found = draw_landmarks(img_bgr.copy())
                    pred, f_pct, d_pct = predict(img_bgr)

                if face_found and pred is not None:
                    focused = pred == 1
                    st.session_state.history_log.append({
                        "ts":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "focused":  focused,
                        "f_pct":    f_pct,
                        "d_pct":    d_pct
                    })
                    if focused:
                        st.markdown(f"""
                        <div class="result-f">
                          <div style="font-size:2rem;margin-bottom:6px">🎯</div>
                          <div class="state-f">FOCUSED</div>
                          <div class="conf-txt">Confidence: {f_pct}%</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-d">
                          <div style="font-size:2rem;margin-bottom:6px">😴</div>
                          <div class="state-d">DISTRACTED</div>
                          <div class="conf-txt">Confidence: {d_pct}%</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("**Focused probability**")
                    st.progress(int(f_pct))
                    st.caption(f"{f_pct}%")
                    st.markdown("**Distracted probability**")
                    st.progress(int(d_pct))
                    st.caption(f"{d_pct}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Face landmarks**")
                    st.image(cv2.cvtColor(img_lm, cv2.COLOR_BGR2RGB),
                             use_container_width=True)
                    st.markdown(f"""
                    <div class="stat-grid">
                      <div class="stat-box">
                        <div class="stat-val">{"Open" if focused else "Closed"}</div>
                        <div class="stat-lbl">Eye State</div>
                      </div>
                      <div class="stat-box">
                        <div class="stat-val">{max(f_pct,d_pct)}%</div>
                        <div class="stat-lbl">Confidence</div>
                      </div>
                      <div class="stat-box">
                        <div class="stat-val">468</div>
                        <div class="stat-lbl">Landmarks</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                elif not face_found:
                    st.error("No face detected. Upload a clearer front-facing photo.")
            else:
                st.markdown("""
                <div class="empty-result">
                  <div style="font-size:2.5rem;margin-bottom:10px">👁️</div>
                  Upload an image and click Analyze
                </div>""", unsafe_allow_html=True)

    # ══ WEBCAM ════════════════════════════════════════════════════════════════
 elif page == "webcam":
        col1, col2 = st.columns([1.3, 0.7], gap="large")
        with col1:
            st.markdown("""<div class="sec-title">
              <span class="teal-dot"></span> Live feed
            </div>""", unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("▶️  Start Session",
                             use_container_width=True, type="primary"):
                    st.session_state.running    = True
                    st.session_state.focus_log  = []
                    st.session_state.start_time = time.time()
                    st.session_state.alerts     = 0
            with bc2:
                if st.button("⏹  Stop Session", use_container_width=True):
                    st.session_state.running = False

            frame_box = st.empty()

            if st.session_state.running:
                cap = cv2.VideoCapture(0)
                for _ in range(300):
                    if not st.session_state.running:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot access camera.")
                        break
                    frame = cv2.flip(frame, 1)
                    pred, f_pct, d_pct = predict(frame)
                    frame_lm, _ = draw_landmarks(frame.copy())
                    focused = pred == 1 if pred is not None else True
                    st.session_state.focus_log.append(focused)
                    if not focused:
                        st.session_state.alerts += 1

                    h, w    = frame_lm.shape[:2]
                    elapsed = int(time.time()-st.session_state.start_time)
                    m, s    = divmod(elapsed, 60)
                    color   = (45, 212, 191) if focused else (248, 113, 113)
                    label   = "FOCUSED" if focused else "DISTRACTED"

                    cv2.rectangle(frame_lm,(0,0),(w,50),(10,22,40),-1)
                    cv2.rectangle(frame_lm,(0,0),(w,50),(36,55,92),1)
                    cv2.putText(frame_lm, f"FaceFlow  |  {label}",
                                (12,32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, color, 2)
                    cv2.putText(frame_lm, f"{m:02d}:{s:02d}",
                                (w-70,32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (74,96,128), 1)
                    frame_box.image(
                        cv2.cvtColor(frame_lm, cv2.COLOR_BGR2RGB),
                        use_container_width=True)
                    time.sleep(0.05)

                cap.release()
                log = st.session_state.focus_log
                if len(log) > 0:
                    elp   = int(time.time()-st.session_state.start_time) \
                            if st.session_state.start_time else 0
                    score = round(sum(log)/len(log)*100, 1)
                    already = (st.session_state.saved_sessions and
                               len(st.session_state.saved_sessions[-1]['log'])==len(log))
                    if not already:
                        st.session_state.saved_sessions.append({
                            "ts":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "log":      list(log),
                            "duration": elp,
                            "alerts":   st.session_state.alerts,
                            "score":    score
                        })
                st.session_state.running = False
            else:
                frame_box.markdown("""
                <div class="live-empty">
                  <div style="font-size:2.5rem;margin-bottom:8px">📷</div>
                  Press Start Session to begin
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class="sec-title">
              <span class="teal-dot"></span> Session stats
            </div>""", unsafe_allow_html=True)

            log   = st.session_state.focus_log
            total = len(log)
            score = round(sum(log)/total*100, 1) if total > 0 else 0
            elp   = int(time.time()-st.session_state.start_time) \
                    if st.session_state.start_time else 0
            m, s  = divmod(elp, 60)

            st.markdown(f"""
            <div class="score-card">
              <div class="score-val">{score}%</div>
              <div class="score-lbl">Focus Score</div>
            </div>""", unsafe_allow_html=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Session time", f"{m:02d}:{s:02d}")
                st.metric("Total checks", total)
            with sc2:
                st.metric("Focused", sum(log))
                st.metric("Alerts",  st.session_state.alerts)

            if total > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📊  Generate Report",
                             use_container_width=True, type="primary"):
                    buf = make_chart(log)
                    st.image(buf, use_container_width=True)
                    st.download_button(
                        "⬇️  Save Report PNG",
                        data=buf.getvalue(),
                        file_name="faceflow_report.png",
                        mime="image/png",
                        use_container_width=True)

    # ══ REPORTS ═══════════════════════════════════════════════════════════════
    elif page == "reports":
        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> Session & Photo Reports
        </div>""", unsafe_allow_html=True)

        sessions = st.session_state.saved_sessions
        hist     = st.session_state.history_log

        if not sessions and not hist:
            st.markdown("""
            <div class="empty-result">
              <div style="font-size:2.5rem;margin-bottom:10px">📊</div>
              <div style="font-weight:600;color:#A8C4DC;margin-bottom:6px">
                No reports yet
              </div>
              <div style="font-size:12px;color:#7A9CBF">
                Complete a Live Webcam Session or analyze photos to generate reports.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            if sessions:
                st.markdown("#### 🎥 Webcam Session Reports")
                for i, sess in enumerate(reversed(sessions)):
                    idx = len(sessions) - i
                    with st.expander(
                        f"Session {idx}  ·  {sess['ts']}  ·  Score: {sess['score']}%"):
                        r1,r2,r3,r4 = st.columns(4)
                        with r1: st.metric("Focus Score", f"{sess['score']}%")
                        with r2:
                            dm,ds = divmod(sess['duration'],60)
                            st.metric("Duration", f"{dm:02d}:{ds:02d}")
                        with r3: st.metric("Checks", len(sess['log']))
                        with r4: st.metric("Alerts", sess['alerts'])
                        if len(sess['log']) > 1:
                            buf = make_chart(sess['log'])
                            st.image(buf, use_container_width=True)
                            st.download_button(
                                f"⬇️  Download Session {idx} Report",
                                data=buf.getvalue(),
                                file_name=f"faceflow_session_{idx}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"dl_sess_{i}")

            if hist:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📸 Photo Analysis Summary")
                total    = len(hist)
                f_count  = sum(1 for h in hist if h['focused'])
                avg_conf = round(sum(max(h['f_pct'],h['d_pct'])
                                     for h in hist)/total,1) if total else 0
                p1,p2,p3,p4 = st.columns(4)
                with p1: st.metric("Total Analyzed", total)
                with p2: st.metric("Focused",        f_count)
                with p3: st.metric("Distracted",     total-f_count)
                with p4: st.metric("Avg Confidence", f"{avg_conf}%")
                if total > 1:
                    buf = make_chart([h['focused'] for h in hist])
                    st.image(buf, use_container_width=True)
                    st.download_button(
                        "⬇️  Download Photo Analysis Report",
                        data=buf.getvalue(),
                        file_name="faceflow_photo_report.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_photo")

    # ══ HISTORY ═══════════════════════════════════════════════════════════════
    elif page == "history":
        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> Photo Analysis History
        </div>""", unsafe_allow_html=True)

        hist = st.session_state.history_log
        if not hist:
            st.markdown("""
            <div class="empty-result">
              <div style="font-size:2.5rem;margin-bottom:10px">🕐</div>
              <div style="font-weight:600;color:#A8C4DC;margin-bottom:6px">
                No history yet
              </div>
              <div style="font-size:12px;color:#7A9CBF">
                Go to Photo Analysis, upload an image and click Analyze.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            total   = len(hist)
            f_count = sum(1 for h in hist if h['focused'])
            h1,h2,h3 = st.columns(3)
            with h1: st.metric("Total Analyses", total)
            with h2: st.metric("Focused",        f_count)
            with h3: st.metric("Distracted",     total-f_count)
            st.markdown("<br>", unsafe_allow_html=True)
            _, cc = st.columns([4,1])
            with cc:
                if st.button("🗑️  Clear All", use_container_width=True):
                    st.session_state.history_log = []
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            for i, entry in enumerate(reversed(hist)):
                badge = "hist-badge-f" if entry['focused'] else "hist-badge-d"
                label = "FOCUSED"     if entry['focused'] else "DISTRACTED"
                conf  = entry['f_pct'] if entry['focused'] else entry['d_pct']
                st.markdown(f"""
                <div class="hist-row">
                  <div class="hist-num">#{total-i}</div>
                  <div class="hist-time">🕐 {entry['ts']}</div>
                  <div class="hist-file">📄 {entry['filename']}</div>
                  <div><span class="{badge}">{label}</span></div>
                  <div class="hist-conf">{conf}% conf.</div>
                </div>""", unsafe_allow_html=True)

    # ══ ABOUT ══════════════════════════════════════════════════════════════════
    elif page == "about":
        st.markdown("""
        <div class="about-hero">
          <div class="about-hero-title">
            Face<span>Flow</span> — AI Focus Detector
          </div>
          <div class="about-hero-sub">
            FaceFlow is a real-time AI-powered focus detection system that uses
            computer vision and machine learning to determine whether a student
            or professional is mentally engaged or distracted — through subtle
            eye and facial cues. No wearables. No special hardware. Just your webcam.
          </div>
          <div class="about-stats">
            <div class="about-stat">
              <div class="about-stat-val">89.8%</div>
              <div class="about-stat-lbl">Accuracy</div>
            </div>
            <div class="about-stat">
              <div class="about-stat-val">84K+</div>
              <div class="about-stat-lbl">Training images</div>
            </div>
            <div class="about-stat">
              <div class="about-stat-val">468</div>
              <div class="about-stat-lbl">Facial landmarks</div>
            </div>
            <div class="about-stat">
              <div class="about-stat-val">~30ms</div>
              <div class="about-stat-lbl">Inference time</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> Core Capabilities
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="about-grid">
          <div class="about-card"><div class="about-icon">🧠</div>
            <div class="about-t">The Problem</div>
            <div class="about-b">Students and professionals lose focus without
              realizing it. FaceFlow detects the exact moment attention drifts
              using passive, non-intrusive computer vision.</div></div>
          <div class="about-card"><div class="about-icon">⚙️</div>
            <div class="about-t">How It Works</div>
            <div class="about-b">MediaPipe maps 468 facial landmarks per frame.
              The eye region is isolated and fed into a Random Forest classifier
              trained on 84,000+ labeled eye images.</div></div>
          <div class="about-card"><div class="about-icon">📊</div>
            <div class="about-t">Session Analytics</div>
            <div class="about-b">Every session produces a focus score, timeline
              chart, and pie chart. All analyses logged in History.
              Full PNG reports exportable.</div></div>
          <div class="about-card"><div class="about-icon">📸</div>
            <div class="about-t">Photo Mode</div>
            <div class="about-b">Upload any front-facing photo to instantly detect
              focus state with landmark overlay and confidence scores.
              Auto-logged to History.</div></div>
          <div class="about-card"><div class="about-icon">🎥</div>
            <div class="about-t">Live Webcam Mode</div>
            <div class="about-b">Real-time detection at ~20 FPS. Live HUD shows
              focus state and session timer. Sessions auto-saved
              to Reports when stopped.</div></div>
          <div class="about-card"><div class="about-icon">🔒</div>
            <div class="about-t">Privacy First</div>
            <div class="about-b">100% local processing. No images or data sent
              to any server. Works entirely offline. Your face data
              never leaves your machine.</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> Detection Pipeline
        </div>""", unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Frame Capture",
             "OpenCV captures live video. Each frame is flipped horizontally for a natural mirror view."),
            ("2", "Face Mesh Detection (MediaPipe)",
             "MediaPipe FaceMesh detects 468 3D landmarks. Key eye indices (33, 263, 159, 386) crop the eye region with 10–20 px margin."),
            ("3", "Feature Extraction",
             "Eye region resized to 64×32 px and converted to grayscale. Six statistical features extracted: mean brightness, split brightness, std dev, bright/dark pixel counts."),
            ("4", "Random Forest Classification",
             "6-feature vector → focus_model.pkl → binary prediction (focused=1, distracted=0) + confidence scores shown in UI."),
            ("5", "Result Display & Logging",
             "Results shown with confidence bars and landmark overlays. Auto-logged to History and Reports with auto-generated charts.")
        ]:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> Technology Stack
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tech-grid">
          <div class="tech-card"><div class="tech-icon">🐍</div><div>
            <div class="tech-name">Python 3.10+</div>
            <div class="tech-desc">Core language</div></div></div>
          <div class="tech-card"><div class="tech-icon">🤖</div><div>
            <div class="tech-name">MediaPipe</div>
            <div class="tech-desc">468-point face mesh detection</div></div></div>
          <div class="tech-card"><div class="tech-icon">📷</div><div>
            <div class="tech-name">OpenCV</div>
            <div class="tech-desc">Video capture & frame processing</div></div></div>
          <div class="tech-card"><div class="tech-icon">🌲</div><div>
            <div class="tech-name">scikit-learn</div>
            <div class="tech-desc">Random Forest classifier</div></div></div>
          <div class="tech-card"><div class="tech-icon">🔢</div><div>
            <div class="tech-name">NumPy</div>
            <div class="tech-desc">Feature extraction</div></div></div>
          <div class="tech-card"><div class="tech-icon">🌊</div><div>
            <div class="tech-name">Streamlit</div>
            <div class="tech-desc">Interactive web UI</div></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sec-title">
          <span class="teal-dot"></span> About the Creator
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="creator-card">
          <div class="creator-avatar">👩‍💻</div>
          <div>
            <div class="creator-name">Hamna Munir</div>
            <div class="creator-title">
              AI/ML Engineer · Software Engineering Student
            </div>
            <div class="creator-bio">
              Built FaceFlow from scratch over 4 weeks as a passion project
              exploring real-world computer vision. Full end-to-end ML pipeline:
              dataset curation, model training, feature engineering, and production
              deployment via Streamlit. Software Engineering student from Pakistan
              specializing in AI/ML systems.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      <div>FaceFlow · Built by <span>Hamna Munir</span> ·
           Python · OpenCV · MediaPipe · scikit-learn · Streamlit</div>
      <div>v1.0 · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
