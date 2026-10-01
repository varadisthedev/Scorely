"""
Streamlit app: SCORELY - Predict Score based on Study Duration
Uses a pre-trained Lasso model (best_model.pkl) and scaler (scaler.pkl).
"""

import os
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scorely · Lasso Score Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── State Management ──────────────────────────────────────────────────────────
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 6.0

# Calculate slider track gradient percentage
track_pct = max(0.0, min(100.0, (st.session_state.study_hours / 24.0) * 100))

# Dynamic active button styling
active_btn_css = ""
if abs(st.session_state.study_hours - 2.0) < 0.15:
    active_btn_css = "div[data-testid='stColumn']:nth-child(1) button { background: #edf6f0 !important; border: 1.5px solid #3b7e55 !important; color: #1a5030 !important; }"
elif abs(st.session_state.study_hours - 5.0) < 0.15:
    active_btn_css = "div[data-testid='stColumn']:nth-child(2) button { background: #edf6f0 !important; border: 1.5px solid #3b7e55 !important; color: #1a5030 !important; }"
elif abs(st.session_state.study_hours - 9.0) < 0.15:
    active_btn_css = "div[data-testid='stColumn']:nth-child(3) button { background: #edf6f0 !important; border: 1.5px solid #3b7e55 !important; color: #1a5030 !important; }"

# ── Custom CSS: Compact, Zero-Scroll, Stitched Card & Scorely Theme ───────────
CUSTOM_CSS = textwrap.dedent(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Caveat:wght@600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* Warm ivory clean background */
.stApp {{
    background: #f7f5ef !important;
    background-image: 
        radial-gradient(circle at 8% 15%, rgba(214, 235, 220, 0.5) 0%, transparent 35%),
        radial-gradient(circle at 92% 80%, rgba(243, 236, 222, 0.65) 0%, transparent 40%) !important;
    color: #1a241c !important;
}}

/* Hide Streamlit default chrome */
#MainMenu, header, footer {{ visibility: hidden !important; }}
.stDeployButton {{ display: none !important; }}

/* Compact Viewport: Wide enough for 2 columns, fits without scroll */
.block-container {{
    max-width: 980px !important;
    padding-top: 1.1rem !important;
    padding-bottom: 1.5rem !important;
}}

/* Main Card Containers */
.scorely-box {{
    background: #ffffff;
    border: 1px solid rgba(220, 226, 218, 0.85);
    border-radius: 24px;
    padding: 1.4rem 1.6rem 1.3rem;
    box-shadow: 0 14px 40px -10px rgba(45, 62, 48, 0.07), 0 2px 6px rgba(0, 0, 0, 0.02);
}}

/* Stitched Card Effect */
.stitched-card {{
    background: #faf8f2;
    border: 2px dashed #9eb49b;
    border-radius: 20px;
    padding: 1.15rem 1.3rem;
    box-shadow: 
        inset 0 0 0 3px #faf8f2,
        inset 0 0 0 4px rgba(158, 180, 155, 0.35),
        0 8px 24px -6px rgba(45, 62, 48, 0.06);
    position: relative;
    margin-top: 0.9rem;
}}

/* Slider Customization */
.stSlider {{
    padding: 0 !important;
}}
.stSlider [data-baseweb="slider"] {{
    margin: 0.15rem 0 !important;
}}
/* Dynamic slider track fill with exact forest green */
[class*="efbyxod5"] {{
    background: linear-gradient(to right, #397a53 0%, #397a53 {track_pct:.1f}%, #e2e5df {track_pct:.1f}%, #e2e5df 100%) !important;
    height: 7px !important;
    border-radius: 999px !important;
}}
/* Slider knob */
[class*="efbyxod3"] {{
    background: #2e6644 !important;
    border: 3px solid #ffffff !important;
    border-radius: 50% !important;
    width: 23px !important;
    height: 23px !important;
    box-shadow: 0 3px 10px rgba(46, 102, 68, 0.38) !important;
}}
/* Hide Streamlit default thumb badge and tick bar */
[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBar"] {{
    display: none !important;
}}

/* Preset Secondary Buttons */
button[data-testid="stBaseButton-secondary"] {{
    background: #ffffff !important;
    color: #1e2d22 !important;
    border: 1px solid #dbe2d8 !important;
    border-radius: 14px !important;
    padding: 0.5rem 0.2rem !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    transition: all 0.18s ease !important;
    line-height: 1.3 !important;
}}
button[data-testid="stBaseButton-secondary"]:hover {{
    background: #edf6f0 !important;
    border-color: #3b7e55 !important;
    color: #1a5030 !important;
    transform: translateY(-1px) !important;
}}

/* Active preset styling */
{active_btn_css}

/* Predict Primary Button */
button[data-testid="stBaseButton-primary"] {{
    width: 100% !important;
    background: #2e6945 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.82rem 1.4rem !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 8px 20px rgba(46, 105, 69, 0.28) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 0.5rem !important;
}}
button[data-testid="stBaseButton-primary"]:hover {{
    background: #245738 !important;
    box-shadow: 0 12px 26px rgba(46, 105, 69, 0.38) !important;
    transform: translateY(-2px) !important;
}}
button[data-testid="stBaseButton-primary"]:active {{
    transform: translateY(0) !important;
}}
</style>
""")
st.html(CUSTOM_CSS)

# ── Load model & scaler ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ── Header & Navbar (Compact, Zero-Scroll) ────────────────────────────────────
NAV_HEADER_HTML = textwrap.dedent("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; padding: 0 0.2rem;">
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:1.35rem;">🌱</span>
        <span style="font-size:1.18rem; font-weight:900; letter-spacing:0.03em; color:#183624;">
            SCORELY
        </span>
        <span style="font-size:0.75rem; font-weight:700; background:#e8f2ea; color:#2e6b47; padding:2px 8px; border-radius:6px; margin-left:6px;">
            Lasso ML
        </span>
    </div>
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #edf5ef;
        border: 1px solid #d4e7d8;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #2b6140;
    ">
        <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#2f855a;"></span>
        Model Online &bull; best_model.pkl
    </div>
</div>

<div style="text-align: center; margin-bottom: 1.1rem;">
    <div style="
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #7b8577;
        margin-bottom: 0.2rem;
    ">STUDY SMARTER</div>
    <h1 style="
        margin: 0;
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.025em;
        line-height: 1.1;
    ">
        <span style="color: #122118;">Score</span>
        <span style="color: #2e6b47;"> Predictor</span>
    </h1>
    <p style="
        color: #636d60;
        font-size: 0.86rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
        font-weight: 400;
    ">Enter study duration to calculate estimated exam score via Lasso regression</p>
</div>
""")
st.html(NAV_HEADER_HTML)

# ── Inference Calculation ─────────────────────────────────────────────────────
input_df = pd.DataFrame([[float(st.session_state.study_hours)]], columns=["Hours"])
raw_score = float(model.predict(scaler.transform(input_df))[0])
display_score = max(0, min(round(raw_score), 100))

# Dynamic tier badge styling
if display_score >= 100:
    badge_label = "🌟 Maximum Score! 🏆"
    badge_bg = "#d7f3df"
    badge_color = "#1e623a"
elif display_score >= 80:
    badge_label = "🌱 Exceptional mastery! 🏆"
    badge_bg = "#d7f3df"
    badge_color = "#1e623a"
elif display_score >= 65:
    badge_label = "🌱 Strong preparation! 📈"
    badge_bg = "#dcf1e2"
    badge_color = "#245f3b"
elif display_score >= 50:
    badge_label = "🌿 Steady progress! 💡"
    badge_bg = "#eef5dc"
    badge_color = "#4c6020"
else:
    badge_label = "🌱 Need more focus! 📚"
    badge_bg = "#fdeeed"
    badge_color = "#99352e"

# ── Two-Column Layout: Fits On-Screen Without Scrolling ────────────────────────
col_left, col_right = st.columns([1.08, 1.0], gap="medium")

# ── Column Left: Input Duration Card ──────────────────────────────────────────
with col_left:
    st.html(textwrap.dedent(f"""
    <div class="scorely-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
            <div style="display:flex; align-items:center; gap:7px;">
                <span style="color:#2f6844; font-size:1.05rem;">🕒</span>
                <span style="font-size:0.92rem; font-weight:700; color:#18261c;">Study Duration</span>
            </div>
            <span style="font-size:0.78rem; font-weight:500; color:#788373;">0 – 24 hours</span>
        </div>

        <!-- Big Center Highlight Box -->
        <div style="
            background: #edf5ef;
            border: 1px solid #d8ebd9;
            border-radius: 18px;
            text-align: center;
            padding: 0.85rem 1rem 0.75rem;
            margin-bottom: 0.9rem;
        ">
            <div style="
                font-size: 2.85rem;
                font-weight: 900;
                color: #14281c;
                line-height: 1;
                letter-spacing: -0.02em;
            ">{float(st.session_state.study_hours):.1f}</div>
            <div style="
                color: #4b5f4f;
                font-size: 0.82rem;
                font-weight: 600;
                margin-top: 0.25rem;
                letter-spacing: 0.01em;
            ">hours of study</div>
        </div>

        <!-- Range Labels -->
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:600; color:#6d7768; margin-bottom:-6px;">
            <span>0</span>
            <span>24</span>
        </div>
    """))

    st.slider(
        "Study Time Slider",
        min_value=0.0,
        max_value=24.0,
        step=0.25,
        label_visibility="collapsed",
        key="study_hours",
    )

    def set_hours(h):
        st.session_state.study_hours = float(h)

    st.html(textwrap.dedent("""<div style="margin-top: 0.6rem; margin-bottom: 0.2rem;"></div>"""))

    p1, p2, p3 = st.columns(3)
    with p1:
        st.button("2h\nQuick Prep", key="p2", on_click=set_hours, args=(2.0,), use_container_width=True, type="secondary")
    with p2:
        st.button("5h\nRegular", key="p5", on_click=set_hours, args=(5.0,), use_container_width=True, type="secondary")
    with p3:
        st.button("9h\nIntensive", key="p9", on_click=set_hours, args=(9.0,), use_container_width=True, type="secondary")

    st.html(textwrap.dedent("""
        <div style="
            text-align: center;
            margin-top: 0.85rem;
            font-size: 0.78rem;
            color: #5d6e5a;
            font-weight: 600;
            background: #edf5ef;
            border: 1px solid #d5e8d8;
            border-radius: 12px;
            padding: 6px 12px;
        ">
            ⚡ <em>Live Inference &bull; Updates instantly as you adjust</em>
        </div>
    """))

    st.html("</div>")  # Close scorely-box

# ── Column Right: Output Card & Stitched Model Info Card ──────────────────────
with col_right:
    # 1. Output Result Card
    st.html(textwrap.dedent(f"""
    <div class="scorely-box" style="display:flex; align-items:center; gap:1.3rem; padding: 1.25rem 1.4rem;">
        <!-- Trophy Icon Box -->
        <div style="
            background: #e2f0e5;
            border: 1px solid #d0e7d5;
            width: 70px;
            height: 70px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
            flex-shrink: 0;
        ">
            🏆
        </div>

        <!-- Score & Feedback Details -->
        <div style="flex-grow: 1;">
            <div style="
                color: #6c7668;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.1rem;
            ">
                Predicted Score
            </div>
            
            <div style="display:flex; align-items:baseline; gap:6px;">
                <span style="
                    font-size: 2.45rem;
                    font-weight: 900;
                    color: #122419;
                    line-height: 1;
                    letter-spacing: -0.03em;
                ">{display_score}</span>
                <span style="
                    font-size: 1.05rem;
                    font-weight: 600;
                    color: #7b8577;
                ">/ 100</span>
            </div>

            <!-- Pill Badge -->
            <div style="
                display: inline-block;
                background: {badge_bg};
                color: {badge_color};
                padding: 3px 11px;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
                margin-top: 0.35rem;
            ">
                {badge_label}
            </div>

            <div style="
                color: #7a8376;
                font-size: 0.77rem;
                margin-top: 0.35rem;
            ">
                Based on <strong>{float(st.session_state.study_hours):.1f}</strong> hours of study
            </div>
        </div>
    </div>
    """))

    # 2. Stitched Card UI: Model Architecture & Imported PKL Artifacts
    st.html(textwrap.dedent("""
    <div class="stitched-card">
        <!-- Stitch header tag with taped pin look -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.65rem;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-size:0.95rem;">🧵</span>
                <span style="
                    font-size:0.74rem;
                    font-weight:800;
                    text-transform:uppercase;
                    letter-spacing:0.08em;
                    color:#3f5843;
                ">
                    Model Architecture &amp; Artifacts
                </span>
            </div>
            <span style="
                font-family: 'Caveat', cursive;
                font-size: 0.95rem;
                color: #5c6c58;
                font-weight: 700;
            ">Lasso Regressor</span>
        </div>

        <!-- Specifications grid -->
        <div style="display:grid; grid-template-columns: 1fr; gap: 0.45rem; font-size:0.79rem;">
            <!-- Algorithm Row -->
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                background:rgba(255,255,255,0.7);
                border:1px solid #e1e9de;
                border-radius:10px;
                padding:6px 10px;
            ">
                <span style="color:#576454; font-weight:600;">🤖 ML Algorithm</span>
                <span style="color:#1b4329; font-weight:800; background:#e4f0e6; padding:2px 7px; border-radius:6px; font-size:0.75rem;">
                    Lasso Regression (L1)
                </span>
            </div>

            <!-- Best Model PKL Row -->
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                background:rgba(255,255,255,0.7);
                border:1px solid #e1e9de;
                border-radius:10px;
                padding:6px 10px;
            ">
                <span style="color:#576454; font-weight:600;">📦 Trained Estimator</span>
                <span style="color:#204b2f; font-weight:700; font-family:monospace; background:#e8eee6; padding:2px 6px; border-radius:5px; font-size:0.75rem;">
                    best_model.pkl
                </span>
            </div>

            <!-- Scaler PKL Row -->
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                background:rgba(255,255,255,0.7);
                border:1px solid #e1e9de;
                border-radius:10px;
                padding:6px 10px;
            ">
                <span style="color:#576454; font-weight:600;">⚖️ Data Scaler</span>
                <span style="color:#204b2f; font-weight:700; font-family:monospace; background:#e8eee6; padding:2px 6px; border-radius:5px; font-size:0.75rem;">
                    scaler.pkl
                </span>
            </div>
        </div>

        <div style="
            margin-top:0.6rem;
            color:#72806e;
            font-size:0.73rem;
            line-height:1.35;
            border-top:1px dashed #cedbd0;
            padding-top:0.45rem;
            display:flex;
            justify-content:space-between;
            align-items:center;
        ">
            <span>🔄 Pipeline: <em>scaler.transform([Hours]) &rarr; model.predict()</em></span>
            <span style="color:#2d6943; font-weight:700;">&check; Synced</span>
        </div>
    </div>
    """))

# ── Footer ────────────────────────────────────────────────────────────────────
st.html(textwrap.dedent("""
<div style="text-align: center; margin-top: 1.2rem; color: #8e968a; font-size: 0.74rem;">
    🌱 Scorely AI &bull; Loaded via Joblib &bull; Built with Streamlit
</div>
"""))
