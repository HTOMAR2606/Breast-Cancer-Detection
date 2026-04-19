import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Cancer Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURE_COLUMNS = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
]
TARGET_COLUMN = "diagnosis"
DATA_PATH = Path(__file__).resolve().parent / "data.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(dataframe):
    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        max_depth=10,
        min_samples_split=4,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def create_pdf(name, age, gender, feature_values, risk_label, risk_score):
    if not REPORTLAB_AVAILABLE:
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Cancer Risk Assessment Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Morphological Metrics", styles["Heading2"]))
    content.append(Paragraph(f"Tumor Size: {feature_values['mean_radius']}", styles["Normal"]))
    content.append(Paragraph(f"Surface Texture: {feature_values['mean_texture']}", styles["Normal"]))
    content.append(Paragraph(f"Tumor Boundary: {feature_values['mean_perimeter']}", styles["Normal"]))
    content.append(Paragraph(f"Tumor Area: {feature_values['mean_area']}", styles["Normal"]))
    content.append(Paragraph(f"Smoothness Index: {feature_values['mean_smoothness']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Risk Level: {risk_label}", styles["Heading2"]))
    content.append(Paragraph(f"Risk Score: {risk_score:.2f}", styles["Normal"]))
    content.append(Spacer(1, 12))

    plt.figure()
    plt.bar(["Risk"], [prob_malignant])
    plt.ylim(0, 1)
    plt.title("Risk Score")
    risk_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(risk_img.name, bbox_inches="tight")
    plt.close()

    content.append(Paragraph("Risk Visualization:", styles["Heading3"]))
    content.append(Image(risk_img.name, width=400, height=200))
    content.append(Spacer(1, 12))
    content.append(
        Paragraph(
            "Disclaimer: This application is for educational purposes only and not a clinical diagnostic tool.",
            styles["Italic"],
        )
    )

    doc.build(content)
    return temp_file.name


def risk_meta(prob_malignant):
    if prob_malignant < 0.35:
        return "Low Risk", "#60A5FA", "Stable pattern detected"
    if prob_malignant < 0.70:
        return "Medium Risk", "#FBBF24", "Closer clinical review suggested"
    return "High Risk", "#F87171", "Escalated diagnostic attention recommended"


def feature_config(series, step):
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "value": float(series.median()),
        "step": step,
    }


if not DATA_PATH.exists():
    st.error("The file 'data.csv' was not found in the same folder as app.py.")
    st.stop()

df = load_data()
model, scaler = train_model(df)

radius_cfg = feature_config(df["mean_radius"], 0.1)
texture_cfg = feature_config(df["mean_texture"], 0.1)
perimeter_cfg = feature_config(df["mean_perimeter"], 0.1)
area_cfg = feature_config(df["mean_area"], 1.0)
smoothness_cfg = feature_config(df["mean_smoothness"], 0.001)

default_slider_values = {
    "mean_radius": radius_cfg["value"],
    "mean_texture": texture_cfg["value"],
    "mean_perimeter": perimeter_cfg["value"],
    "mean_area": area_cfg["value"],
    "mean_smoothness": smoothness_cfg["value"],
}

for key, value in default_slider_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    """
    <style>
    :root {
        --bg: #08111f;
        --panel: rgba(13, 24, 42, 0.88);
        --panel-2: rgba(17, 29, 49, 0.84);
        --stroke: rgba(96, 165, 250, 0.12);
        --text: #E5E7EB;
        --muted: #94A3B8;
        --blue: #60A5FA;
        --blue-strong: #3B82F6;
        --rose: #F472B6;
        --red: #F87171;
        --amber: #FBBF24;
        --shadow: 0 22px 50px rgba(2, 8, 23, 0.45);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(244, 114, 182, 0.10), transparent 18%),
            linear-gradient(180deg, #07101C 0%, #0B1220 100%);
        color: var(--text);
    }

    header[data-testid="stHeader"], div[data-testid="collapsedControl"] {
        display: none;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        border-radius: 22px;
        background: rgba(11, 20, 36, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.10);
        backdrop-filter: blur(18px);
        box-shadow: var(--shadow);
    }

    .brand {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .topnav-links {
        display: flex;
        gap: 1.6rem;
        color: var(--muted);
        font-size: 1rem;
    }

    .topnav-links .active {
        color: var(--text);
        border-bottom: 2px solid var(--blue);
        padding-bottom: 0.25rem;
    }

    .topnav-actions {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .search-pill, .icon-pill {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(96, 165, 250, 0.08);
        color: var(--muted);
        border-radius: 14px;
    }

    .search-pill {
        min-width: 280px;
        padding: 0.85rem 1rem;
    }

    .icon-pill {
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid rgba(96, 165, 250, 0.10);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(22px);
    }

    .hero-wrap {
        position: relative;
        overflow: hidden;
        padding: 2rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.35fr 0.95fr;
        gap: 1.5rem;
        align-items: center;
    }

    .eyebrow {
        color: #93C5FD;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .hero-title {
        margin: 0.65rem 0 0.7rem;
        font-size: clamp(2.4rem, 5vw, 4rem);
        line-height: 1.02;
        font-weight: 850;
        letter-spacing: -0.05em;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 720px;
    }

    .hero-badges {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .hero-badge {
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.12);
        color: #BFDBFE;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .hero-visual {
        padding: 1.1rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(244, 114, 182, 0.18), transparent 26%),
            linear-gradient(160deg, rgba(17, 29, 49, 0.95), rgba(9, 17, 31, 0.98));
        border: 1px solid rgba(244, 114, 182, 0.14);
        min-height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .hero-visual svg {
        width: 100%;
        max-width: 360px;
        height: auto;
        filter: drop-shadow(0 24px 36px rgba(244, 114, 182, 0.14));
    }

    .section-card {
        padding: 1.5rem;
        height: 100%;
    }

    .card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .muted-tag {
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: rgba(244, 114, 182, 0.09);
        color: #F9A8D4;
        border: 1px solid rgba(244, 114, 182, 0.16);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .mini-panel {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 18px;
        background: var(--panel-2);
        border: 1px solid rgba(96, 165, 250, 0.08);
    }

    .mini-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .mini-value {
        font-size: 1.6rem;
        font-weight: 800;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .status-card {
        padding: 1.2rem;
        border-radius: 20px;
        background: rgba(13, 24, 42, 0.86);
        border: 1px solid rgba(96, 165, 250, 0.08);
        box-shadow: var(--shadow);
    }

    .status-title {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
    }

    .status-value {
        margin-top: 0.4rem;
        font-size: 1.7rem;
        font-weight: 800;
    }

    .stTextInput label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
    }

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
        background: rgba(18, 31, 53, 0.92) !important;
        color: var(--text) !important;
        border: 1px solid rgba(96, 165, 250, 0.08) !important;
        border-radius: 16px !important;
        min-height: 3.2rem;
    }

    .stTextInput input::placeholder {
        color: #64748B !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.25rem;
        border-radius: 16px;
        font-weight: 700;
        letter-spacing: 0.03em;
        border: 1px solid rgba(96, 165, 250, 0.10);
        color: var(--text);
        background: rgba(18, 31, 53, 0.88);
        transition: all 0.25s ease;
        box-shadow: 0 12px 28px rgba(2, 8, 23, 0.30);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 32px rgba(59, 130, 246, 0.20);
    }

    .primary-btn .stButton > button {
        background: linear-gradient(135deg, #60A5FA, #3B82F6);
        color: #08111f;
        border: none;
    }

    .secondary-btn .stButton > button {
        background: rgba(18, 31, 53, 0.92);
    }

    div[data-baseweb="slider"] > div > div {
        height: 4px !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

    div[data-baseweb="slider"] [role="slider"] {
        width: 18px !important;
        height: 18px !important;
        background: linear-gradient(135deg, #F9A8D4, #93C5FD) !important;
        border: 2px solid rgba(255,255,255,0.8) !important;
        box-shadow: 0 0 0 6px rgba(96, 165, 250, 0.12) !important;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--text);
    }

    .result-box {
        margin-top: 1.25rem;
        padding: 1.15rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(17, 29, 49, 0.98), rgba(12, 21, 36, 0.96));
        border: 1px solid rgba(96, 165, 250, 0.12);
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .result-subtitle {
        color: var(--muted);
        margin-bottom: 0.85rem;
    }

    .pill-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }

    .pill {
        padding: 0.9rem;
        border-radius: 16px;
        background: rgba(18, 31, 53, 0.85);
        border: 1px solid rgba(96, 165, 250, 0.08);
    }

    .pill-label {
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.35rem;
    }

    .pill-value {
        font-size: 1.08rem;
        font-weight: 700;
    }

    .note {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .footer-note {
        margin-top: 1.2rem;
        color: var(--muted);
        text-align: center;
        font-size: 0.92rem;
    }

    .stProgress > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 999px;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #60A5FA, #F472B6) !important;
    }

    @media (max-width: 1100px) {
        .hero-grid, .status-grid, .pill-grid {
            grid-template-columns: 1fr;
        }

        .topbar {
            flex-direction: column;
            align-items: stretch;
        }

        .topnav-links, .topnav-actions {
            justify-content: center;
            flex-wrap: wrap;
        }

        .search-pill {
            min-width: 220px;
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
        <div class="brand">Cancer Detection System</div>
        <div class="topnav-links">
            <div class="active">Dashboard</div>
            <div>Patient Records</div>
            <div>Clinical Trials</div>
        </div>
        <div class="topnav-actions">
            <div class="search-pill">Search patient ID...</div>
            <div class="icon-pill">🔔</div>
            <div class="icon-pill">🧬</div>
            <div class="icon-pill">👩‍⚕️</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="glass-card hero-wrap">
        <div class="hero-grid">
            <div>
                <div class="eyebrow"></div>
                <div class="hero-title">ML-Based System for Detection of High-Risk Breast Cancer</div>
                <div class="hero-copy">
                    Adjust morphological parameters and patient context to generate a high-clarity
                    breast cancer risk assessment.
                </div>
            </div>
            <div class="hero-visual">
                <svg viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Breast cancer awareness illustration">
                    <defs>
                        <linearGradient id="panelGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#93C5FD" stop-opacity="0.85"/>
                            <stop offset="100%" stop-color="#F472B6" stop-opacity="0.85"/>
                        </linearGradient>
                        <linearGradient id="ribbonFill" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#FDA4AF"/>
                            <stop offset="100%" stop-color="#F472B6"/>
                        </linearGradient>
                    </defs>
                    <rect x="22" y="28" width="376" height="264" rx="28" fill="rgba(8,17,31,0.55)" stroke="url(#panelGlow)" stroke-width="2"/>
                    <circle cx="318" cy="85" r="24" fill="#60A5FA" opacity="0.12"/>
                    <circle cx="108" cy="78" r="36" fill="#F472B6" opacity="0.10"/>
                    <path d="M175 88 C128 124, 136 175, 183 210 L144 274 C137 285, 146 296, 157 289 L212 231 L264 290 C274 302, 290 288, 282 275 L244 211 C288 177, 292 127, 247 89 C223 68, 198 67, 175 88 Z" fill="url(#ribbonFill)" opacity="0.95"/>
                    <path d="M210 106 C188 124, 189 155, 214 180 C239 155, 240 123, 218 106 C215 103, 212 103, 210 106 Z" fill="#FDF2F8" opacity="0.95"/>
                    <circle cx="115" cy="218" r="42" fill="none" stroke="#93C5FD" stroke-width="3" opacity="0.55"/>
                    <path d="M94 218 L108 232 L137 203" fill="none" stroke="#93C5FD" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    <rect x="264" y="176" width="92" height="62" rx="18" fill="#0F172A" stroke="#60A5FA" stroke-opacity="0.35"/>
                    <text x="284" y="202" fill="#94A3B8" font-size="14" font-family="Arial">CASE SCORE</text>
                    <text x="284" y="228" fill="#E5E7EB" font-size="24" font-weight="700" font-family="Arial">98.2%</text>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.02, 1.28], gap="large")

with left_col:
    st.markdown(
        """
        <div class="glass-card section-card">
            <div class="card-head">
                <div class="card-title">Patient Information</div>
                <div class="muted-tag">Clinical Intake</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    name = st.text_input("Patient Name", placeholder="Enter patient name")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=20, step=1)

    st.markdown(
        """
        <div class="mini-panel">
            <div class="mini-label">Screening Summary</div>
            <div class="mini-value">Breast Cancer Triage</div>
            <div class="note">
                This module prioritizes patient cases using five morphological features from the
                diagnostic dataset and highlights probability in a clinical-friendly view.
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        """
        <div class="glass-card section-card">
            <div class="card-head">
                <div class="card-title">Tumor Morphological Details</div>
                <div class="muted-tag">High Precision Metrics</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="metric-row"><div class="mini-label">Tumor Size (mm)</div><div class="metric-value">{st.session_state["mean_radius"]:.1f}</div></div>',
        unsafe_allow_html=True,
    )
    mean_radius = st.slider(
        "Tumor Size (mm)",
        min_value=radius_cfg["min"],
        max_value=radius_cfg["max"],
        value=st.session_state["mean_radius"],
        step=radius_cfg["step"],
        key="mean_radius",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="metric-row"><div class="mini-label">Surface Texture</div><div class="metric-value">{st.session_state["mean_texture"]:.2f}</div></div>',
        unsafe_allow_html=True,
    )
    mean_texture = st.slider(
        "Surface Texture",
        min_value=texture_cfg["min"],
        max_value=texture_cfg["max"],
        value=st.session_state["mean_texture"],
        step=texture_cfg["step"],
        key="mean_texture",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="metric-row"><div class="mini-label">Tumor Boundary</div><div class="metric-value">{st.session_state["mean_perimeter"]:.1f}</div></div>',
        unsafe_allow_html=True,
    )
    mean_perimeter = st.slider(
        "Tumor Boundary",
        min_value=perimeter_cfg["min"],
        max_value=perimeter_cfg["max"],
        value=st.session_state["mean_perimeter"],
        step=perimeter_cfg["step"],
        key="mean_perimeter",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="metric-row"><div class="mini-label">Tumor Area (sq.mm)</div><div class="metric-value">{st.session_state["mean_area"]:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    mean_area = st.slider(
        "Tumor Area (sq.mm)",
        min_value=area_cfg["min"],
        max_value=area_cfg["max"],
        value=st.session_state["mean_area"],
        step=area_cfg["step"],
        key="mean_area",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="metric-row"><div class="mini-label">Smoothness Index</div><div class="metric-value">{st.session_state["mean_smoothness"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )
    mean_smoothness = st.slider(
        "Smoothness Index",
        min_value=smoothness_cfg["min"],
        max_value=smoothness_cfg["max"],
        value=st.session_state["mean_smoothness"],
        step=smoothness_cfg["step"],
        key="mean_smoothness",
        label_visibility="collapsed",
    )

    action_left, action_right = st.columns([1, 1], gap="medium")
    with action_left:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        save_draft = st.button("Save Draft", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with action_right:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        predict = st.button("Analyze Patient", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if save_draft:
        st.toast("Draft values saved in the current session.")

    input_df = pd.DataFrame(
        [
            {
                "mean_radius": mean_radius,
                "mean_texture": mean_texture,
                "mean_perimeter": mean_perimeter,
                "mean_area": mean_area,
                "mean_smoothness": mean_smoothness,
            }
        ]
    )

    if predict:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        prob_malignant = float(model.predict_proba(scaled_input)[0][0])
        risk_label, risk_color, risk_note = risk_meta(prob_malignant)
        diagnosis_text = "High Risk Patient (Malignant)" if prediction == 0 else "Low Risk Patient (Benign)"

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title" style="color:{risk_color};">{risk_label}</div>
                <div class="result-subtitle">{risk_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob_malignant)
        st.markdown(
            f"""
            <div class="pill-grid">
                <div class="pill">
                    <div class="pill-label">Risk Score</div>
                    <div class="pill-value">{prob_malignant:.2f}</div>
                </div>
                <div class="pill">
                    <div class="pill-label">Prediction</div>
                    <div class="pill-value">{diagnosis_text}</div>
                </div>
                <div class="pill">
                    <div class="pill-label">Patient</div>
                    <div class="pill-value">{name or "Unnamed Patient"}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if REPORTLAB_AVAILABLE:
            pdf_path = create_pdf(
                name=name or "Not Provided",
                age=age,
                gender=gender,
                feature_values=input_df.iloc[0].to_dict(),
                risk_label=risk_label,
                risk_score=prob_malignant,
            )
            if pdf_path:
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        "Download PDF Report",
                        data=pdf_file.read(),
                        file_name="cancer_risk_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        else:
            st.info("Install `reportlab` to enable PDF export.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note"></div>',
    unsafe_allow_html=True,
)
