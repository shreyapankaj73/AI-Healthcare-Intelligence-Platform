import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from ai.extractor import extract_medical_data
from ai.history import save_record, get_worker_history, get_all_workers
from ai.analytics import get_risk_distribution, get_avg_vitals, get_high_risk_workers


from ai.analytics import (
    get_risk_distribution,
    get_avg_vitals,
    get_high_risk_workers
)
# =========================================================
# PAGE CONFIG & SESSION STATE INIT
# =========================================================

st.set_page_config(
    page_title="IOCL Health Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

role = st.sidebar.radio(
    "Portal",
    [
        "👷 Worker",
        "🏢 Admin"
    ]
)

# Initialize session state with safe defaults
DEFAULT_VALUES = {
    "ocr_age": 35,
    "ocr_bmi": 24.0,
    "ocr_glucose": 100,
    "ocr_cholesterol": 180,
    "ocr_oxygen": 98,
    "ocr_heart_rate": 75,
    "autofilled_fields": {},
    "report_processed": False
}

for key, default_val in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

os.makedirs("uploads", exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_DIR = "model_artifacts"
model         = joblib.load(os.path.join(MODEL_DIR, "health_risk_model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #0b0f1a; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #0d2137 0%, #0a3d62 50%, #1a1a2e 100%);
    border: 1px solid rgba(0,170,255,0.25); border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero-title { font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0 0 0.4rem; }
.hero-sub { font-size: 1rem; color: #8baabf; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(0,170,255,0.15);
    border: 1px solid rgba(0,170,255,0.4); color: #00aaff;
    border-radius: 20px; padding: 2px 12px; font-size: 0.78rem;
    font-weight: 600; margin-bottom: 0.8rem;
}
.section-header { display: flex; align-items: center; gap: 10px; margin: 2rem 0 1rem; }
.section-header h2 { font-size: 1.35rem; font-weight: 700; color: #e2e8f0; margin: 0; }
.section-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,170,255,0.4), transparent); }
.card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }

.autofill-badge { display: inline-block; background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); color: #86efac; border-radius: 8px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; margin-left: 4px; }
.input-container { position: relative; }

.risk-high   { background: linear-gradient(135deg,#1a0505,#2d0a0a); border:1px solid rgba(239,68,68,0.5); border-radius:16px; padding:1.8rem 2rem; }
.risk-medium { background: linear-gradient(135deg,#1a1005,#2d1f0a); border:1px solid rgba(251,146,60,0.5); border-radius:16px; padding:1.8rem 2rem; }
.risk-low    { background: linear-gradient(135deg,#051a0d,#0a2d17); border:1px solid rgba(34,197,94,0.5);  border-radius:16px; padding:1.8rem 2rem; }
.risk-label  { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.risk-conf   { font-size: 0.9rem; color: #94a3b8; margin: 4px 0 0; }

.health-score { font-size: 3rem; font-weight: 900; color: #00aaff; text-align: center; margin: 1rem 0; }
.health-score-label { font-size: 0.9rem; color: #64748b; text-align: center; text-transform: uppercase; letter-spacing: 1px; }

.metric-pill  { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-value { font-size:1.8rem; font-weight:700; line-height:1; color:#f1f5f9; margin:0; }
.metric-label { font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; margin:5px 0 0; }
.metric-normal { color:#22c55e !important; }
.metric-warn   { color:#f59e0b !important; }
.metric-danger { color:#ef4444 !important; }

.alert-chip { display:flex; align-items:center; gap:10px; background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:10px 14px; margin-bottom:8px; color:#fca5a5; font-size:0.9rem; }
.rec-chip   { display:flex; align-items:center; gap:10px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); border-radius:10px; padding:10px 14px; margin-bottom:8px; color:#a5b4fc; font-size:0.9rem; }
.ok-chip    { display:flex; align-items:center; gap:10px; background:rgba(34,197,94,0.1);  border:1px solid rgba(34,197,94,0.3);  border-radius:10px; padding:10px 14px; color:#86efac; font-size:0.9rem; }

.info-row { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1e293b; font-size:0.9rem; }
.info-key { color:#64748b; }
.info-val { color:#e2e8f0; font-weight:600; }

.trend-badge-High   { background:rgba(239,68,68,0.2);  color:#ef4444; border-radius:6px; padding:2px 8px; font-size:0.8rem; font-weight:700; }
.trend-badge-Medium { background:rgba(251,146,60,0.2); color:#f97316; border-radius:6px; padding:2px 8px; font-size:0.8rem; font-weight:700; }
.trend-badge-Low    { background:rgba(34,197,94,0.2);  color:#22c55e; border-radius:6px; padding:2px 8px; font-size:0.8rem; font-weight:700; }

.rec-box { background: rgba(99,102,241,0.08); border-left: 4px solid #6366f1; padding: 12px 14px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem; }
.stat-box { background: rgba(0,170,255,0.08); border-left: 4px solid #00aaff; padding: 12px 14px; border-radius: 6px; margin-bottom: 10px; font-size: 0.85rem; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <h2>{title}</h2>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

def metric_color(value, low_ok, high_warn, high_danger=None):
    if high_danger and value >= high_danger: return "metric-danger"
    if value >= high_warn or value < low_ok: return "metric-warn"
    return "metric-normal"

def render_metric(label, value, unit="", low_ok=None, high_warn=None, high_danger=None):
    cls = ""
    if low_ok is not None:
        cls = metric_color(value, low_ok, high_warn, high_danger)
    st.markdown(f"""
    <div class="metric-pill">
        <p class="metric-value {cls}">{value}<span style="font-size:0.9rem;font-weight:400;color:#64748b"> {unit}</span></p>
        <p class="metric-label">{label}</p>
    </div>""", unsafe_allow_html=True)

def calculate_health_score(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking):
    """Calculate comprehensive health score (0-100)"""
    score = 100
    
    if age < 25 or age > 65: score -= 5
    elif age > 55: score -= 2
    
    if bmi < 18.5 or bmi > 30: score -= 10
    elif bmi > 25: score -= 5
    
    if glucose < 70 or glucose > 150: score -= 15
    elif glucose > 120: score -= 8
    
    if cholesterol > 240: score -= 15
    elif cholesterol > 200: score -= 8
    
    if oxygen < 94: score -= 15
    elif oxygen < 96: score -= 8
    
    if heart_rate < 50 or heart_rate > 100: score -= 8
    elif heart_rate > 90: score -= 3
    
    if smoking: score -= 10
    
    return max(0, min(100, score))

def identify_risk_factors(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking, sleep_hours, heat_exposure):
    """Identify specific risk factors"""
    risks = []
    
    if smoking: risks.append("❌ Active smoking detected")
    if bmi > 30: risks.append("⚠️ Overweight (BMI > 30)")
    if bmi < 18.5: risks.append("⚠️ Underweight (BMI < 18.5)")
    if glucose > 126: risks.append("🔴 High glucose (pre-diabetic)")
    if glucose < 70: risks.append("🔴 Low glucose (hypoglycemia risk)")
    if cholesterol > 240: risks.append("🔴 High cholesterol")
    if oxygen < 95: risks.append("🔴 Low oxygen saturation")
    if heart_rate > 100: risks.append("⚠️ Elevated heart rate")
    if heart_rate < 50: risks.append("⚠️ Low heart rate")
    if sleep_hours < 6: risks.append("⚠️ Insufficient sleep")
    if heat_exposure > 7: risks.append("🌡️ High heat exposure")
    if age > 55: risks.append("📊 Age-related concerns")
    
    return risks if risks else ["✅ No major risk factors identified"]

def get_recommendations(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking, sleep_hours, heat_exposure):
    """Generate personalized recommendations"""
    recs = []
    
    if smoking: recs.append("🚭 Enroll in smoking cessation program")
    if bmi > 25: recs.append("🏃 Increase physical activity - target 150 min/week")
    if glucose > 120: recs.append("🥗 Reduce sugar intake, monitor carbohydrates")
    if cholesterol > 200: recs.append("❤️ Increase fiber intake and regular check-ups")
    if oxygen < 96: recs.append("🫁 Ensure adequate ventilation at workplace")
    if heart_rate > 90: recs.append("😌 Practice stress management and relaxation")
    if sleep_hours < 7: recs.append("😴 Aim for 7-9 hours of quality sleep")
    if heat_exposure > 5: recs.append("💧 Increase hydration and use cooling measures")
    if age > 55: recs.append("👨‍⚕️ Schedule annual health screening")
    
    if not recs: recs.append("✅ Maintain current healthy lifestyle habits")
    
    return recs

def get_health_assessment_summary(health_score, risk_factors, age, bmi, glucose, cholesterol, oxygen, heart_rate):
    """Generate comprehensive health assessment summary"""
    summary = f"""
    ### Overall Assessment
    Your health score is **{health_score}/100**, indicating a {'🔴 HIGH RISK' if health_score < 50 else '🟠 MODERATE RISK' if health_score < 70 else '🟢 GOOD HEALTH'} status.
    
    #### Key Findings:
    - **Age**: {age} years
    - **BMI**: {bmi:.1f} {'(Normal)' if 18.5 <= bmi <= 24.9 else '(Elevated)' if bmi <= 30 else '(High)'}
    - **Glucose**: {glucose} mg/dL {'(Optimal)' if glucose <= 100 else '(Elevated)' if glucose <= 150 else '(High)'}
    - **Cholesterol**: {cholesterol} mg/dL {'(Optimal)' if cholesterol <= 200 else '(Borderline)' if cholesterol <= 239 else '(High)'}
    - **Oxygen**: {oxygen}% {'(Normal)' if oxygen >= 96 else '(Low)' if oxygen >= 94 else '(Critical)'}
    - **Heart Rate**: {heart_rate} bpm {'(Normal)' if 60 <= heart_rate <= 80 else '(Elevated)'}
    
    #### Risk Factors Identified: {len(risk_factors)}
    {chr(10).join(['- ' + rf for rf in risk_factors[:5]])}
    
    #### Next Steps:
    1. Review personalized recommendations below
    2. Schedule follow-up health check
    3. Monitor vitals regularly
    4. Implement recommended lifestyle changes
    """
    return summary

# =========================================================
# SIDEBAR WITH IMPROVED AUTOFILL
# =========================================================

with st.sidebar:
    st.markdown("### 👷 Worker Health Inputs")
    st.markdown("<p style='color:#64748b;font-size:0.82rem;margin-top:-8px'>Upload report to auto-fill values</p>", unsafe_allow_html=True)
    st.markdown("---")

    worker_id = st.text_input("Worker ID / Employee No.", value="IOCL-001", placeholder="e.g. IOCL-001")

    st.markdown("**🧬 Demographics**")
    
    # Age with autofill indicator
    age_col1, age_col2 = st.columns([3, 1])
    with age_col1:
        age = st.slider("Age (years)", 18, 70, int(st.session_state.get("ocr_age", 35)))
    with age_col2:
        if "ocr_age" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)
    
    # BMI with autofill indicator
    bmi_col1, bmi_col2 = st.columns([3, 1])
    with bmi_col1:
        bmi = st.slider("BMI", 15.0, 45.0, float(st.session_state.get("ocr_bmi", 24.0)), step=0.1)
    with bmi_col2:
        if "ocr_bmi" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)
    
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x else "Non-Smoker")

    st.markdown("---")
    st.markdown("**🩺 Vitals**")
    
    # Glucose with autofill indicator
    glucose_col1, glucose_col2 = st.columns([3, 1])
    with glucose_col1:
        glucose = st.slider("Glucose (mg/dL)", 50, 300, int(st.session_state.get("ocr_glucose", 100)))
    with glucose_col2:
        if "ocr_glucose" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)
    
    # Cholesterol with autofill indicator
    chol_col1, chol_col2 = st.columns([3, 1])
    with chol_col1:
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, int(st.session_state.get("ocr_cholesterol", 180)))
    with chol_col2:
        if "ocr_cholesterol" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)
    
    # Oxygen with autofill indicator
    oxygen_col1, oxygen_col2 = st.columns([3, 1])
    with oxygen_col1:
        oxygen = st.slider("SpO₂ (%)", 70, 100, int(st.session_state.get("ocr_oxygen", 98)))
    with oxygen_col2:
        if "ocr_oxygen" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)
    
    # Heart Rate with autofill indicator
    hr_col1, hr_col2 = st.columns([3, 1])
    with hr_col1:
        heart_rate = st.slider("Heart Rate (bpm)", 40, 180, int(st.session_state.get("ocr_heart_rate", 75)))
    with hr_col2:
        if "ocr_heart_rate" in st.session_state.autofilled_fields:
            st.markdown('<span class="autofill-badge">AUTO</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏭 Occupational Factors**")
    heat_exposure = st.slider("Heat Exposure (0–10)", 0, 10, 5)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    
    st.markdown("---")
    
    # Reset autofill button
    if st.session_state.autofilled_fields:
        if st.button("🔄 Reset Auto-filled Values"):
            st.session_state.autofilled_fields = {}
            st.session_state.report_processed = False
            st.rerun()
    
    st.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %H:%M')}")

# =========================================================
# NAVIGATION TABS
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">🛢️ &nbsp;IOCL · OCCUPATIONAL HEALTH</div>
    <p class="hero-title">AI Healthcare Intelligence Platform</p>
    <p class="hero-sub">AI-powered occupational healthcare monitoring for industrial & refinery workers</p>
</div>
""", unsafe_allow_html=True)

if role == "👷 Worker":

    tab1, tab2 = st.tabs([
        "🩺 Health Assessment",
        "📈 Worker History"
    ])

else:

    tab3 = st.tabs([
        "🏢 Management Dashboard"
    ])[0]

# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 1 — HEALTH ASSESSMENT (ADVANCED)                    ║
# ╚══════════════════════════════════════════════════════════╝

if role == "👷 Worker":
    with tab1:
    # ── Upload ────────────────────────────────────────────
        section("📄", "Medical Report Upload")

    uploaded_file = st.file_uploader(
        "Upload lab report (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    medical_values = {}

    if uploaded_file:
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🧠 Vision AI analysing report..."):
            medical_values = extract_medical_data(save_path)

        # Update session state with autofilled values
        key_map = {
            "age": "ocr_age",
            "glucose": "ocr_glucose",
            "cholesterol": "ocr_cholesterol",
            "oxygen": "ocr_oxygen",
            "heart_rate": "ocr_heart_rate",
        }

        for k, sk in key_map.items():
            if k in medical_values and medical_values[k] is not None:
                try:
                    st.session_state[sk] = int(float(medical_values[k]))
                    st.session_state.autofilled_fields[k] = True
                except:
                    pass

        st.session_state.report_processed = True

        patient_keys = ["name", "gender", "mobile", "uhid"]
        vital_keys = ["age", "glucose", "cholesterol", "oxygen", "heart_rate", "hba1c", "hemoglobin", "platelets", "creatinine"]

        patient_info = {k: medical_values[k] for k in patient_keys if k in medical_values}

        left, right = st.columns(2)

        with left:
            st.markdown('<div class="card"><b>👤 Patient Information</b>', unsafe_allow_html=True)
            labels = {"name": "Name", "gender": "Gender", "mobile": "Mobile", "uhid": "UHID"}
            rows = "".join(
                f'<div class="info-row"><span class="info-key">{labels.get(k,k)}</span><span class="info-val">{v}</span></div>'
                for k, v in patient_info.items()
            )
            st.markdown(rows + "</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card"><b>🩸 Extracted Medical Values</b>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
                for k, v in medical_values.items()
                if k != "summary" and v is not None
            )
            st.markdown(rows + "</div>", unsafe_allow_html=True)

        section("🧠", "AI Medical Summary")
        st.info(medical_values.get("summary", "No summary available"))
        st.success("✅ Medical report processed using AI Vision")
         

       
    # ══════════════════════════════════════════════════════════
    # COMPREHENSIVE HEALTH ASSESSMENT
    # ══════════════════════════════════════════════════════════

    section("📊", "Comprehensive Health Assessment")

    # Calculate health score
    health_score = calculate_health_score(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking)
    risk_factors = identify_risk_factors(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking, sleep_hours, heat_exposure)
    recommendations = get_recommendations(age, bmi, glucose, cholesterol, oxygen, heart_rate, smoking, sleep_hours, heat_exposure)

    # Display health score prominently
    score_color = "#ef4444" if health_score < 50 else "#f97316" if health_score < 70 else "#22c55e"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {score_color}20, {score_color}05); border: 2px solid {score_color}; border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
        <div class="health-score-label">Overall Health Score</div>
        <div class="health-score" style="color: {score_color};">{health_score}/100</div>
        <div style="color: #94a3b8; font-size: 0.95rem;">
            {'🔴 High Risk - Immediate Medical Attention Recommended' if health_score < 50 else '🟠 Moderate Risk - Regular Monitoring Required' if health_score < 70 else '🟢 Good Health - Maintain Current Lifestyle'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Summary
    st.markdown("---")
    st.markdown("**📋 Quick Health Summary**")
    summary_text = get_health_assessment_summary(health_score, risk_factors, age, bmi, glucose, cholesterol, oxygen, heart_rate)
    st.markdown(summary_text)

    # Vital Signs Summary
    st.markdown("---")
    st.markdown("**🩺 Vital Signs Summary**")
    vs1, vs2, vs3, vs4, vs5 = st.columns(5)
    with vs1: render_metric("Glucose", glucose, "mg/dL", 70, 120, 150)
    with vs2: render_metric("Cholesterol", cholesterol, "mg/dL", 0, 200, 240)
    with vs3: render_metric("SpO₂", oxygen, "%", 96, 94, 90)
    with vs4: render_metric("Heart Rate", heart_rate, "bpm", 60, 90, 100)
    with vs5: render_metric("BMI", round(bmi, 1), "", 18.5, 25, 30)

    # Detailed Analysis Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚠️ Identified Risk Factors**")
        for risk in risk_factors[:7]:
            st.markdown(f'<div class="alert-chip"><span>{risk}</span></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**💡 Personalized Recommendations**")
        for rec in recommendations[:7]:
            st.markdown(f'<div class="rec-chip"><span>{rec}</span></div>', unsafe_allow_html=True)

    # Health Metrics Comparison with Normal Range
    st.markdown("---")
    section("📉", "Metrics vs. Normal Range Analysis")

    metrics_data = {
        'Metric': ['Glucose', 'Cholesterol', 'SpO₂', 'Heart Rate', 'BMI'],
        'Current': [glucose, cholesterol, oxygen, heart_rate, round(bmi, 1)],
        'Normal Range': ['70-100', '< 200', '≥ 96', '60-80', '18.5-24.9'],
        'Deviation': [
            f"{glucose - 85}%" if glucose > 85 else "Normal",
            f"{cholesterol - 180}" if cholesterol > 180 else "Normal",
            f"{oxygen - 98}%" if oxygen < 98 else "Normal",
            f"{heart_rate - 70}" if heart_rate > 70 else "Normal",
            f"{bmi - 22:.1f}" if bmi > 22 else "Normal"
        ],
        'Status': [
            '✅' if 70 <= glucose <= 100 else '⚠️' if 70 <= glucose <= 150 else '🔴',
            '✅' if cholesterol <= 200 else '⚠️' if cholesterol <= 240 else '🔴',
            '✅' if oxygen >= 96 else '⚠️' if oxygen >= 94 else '🔴',
            '✅' if 60 <= heart_rate <= 80 else '⚠️' if 40 <= heart_rate <= 100 else '🔴',
            '✅' if 18.5 <= bmi <= 24.9 else '⚠️' if 18.5 <= bmi <= 30 else '🔴',
        ]
    }

    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Lifestyle Factors
    st.markdown("---")
    section("🏠", "Lifestyle & Environmental Factors")

    lf1, lf2, lf3 = st.columns(3)
    with lf1: render_metric("Sleep Hours", sleep_hours, "hrs", 7, 6, 5)
    with lf2: render_metric("Heat Exposure", heat_exposure, "/10", 3, 6, 8)
    with lf3:
        st.markdown(f"""
        <div class="metric-pill">
            <p class="metric-value metric-{'danger' if smoking else 'normal'}">{'Yes' if smoking else 'No'}</p>
            <p class="metric-label">Smoking Status</p>
        </div>""", unsafe_allow_html=True)

    # Multi-Metric Visualizations
    st.markdown("---")
    section("📊", "Health Profile Visualizations")

    # Radar Chart
    glucose_norm = 100 - (abs(glucose - 100) / 100 * 100)
    cholesterol_norm = 100 - (min(cholesterol, 240) / 240 * 100)
    oxygen_norm = (oxygen / 100) * 100
    heart_rate_norm = 100 - (abs(heart_rate - 75) / 75 * 100)
    bmi_norm = 100 - (abs(bmi - 22) / 22 * 100)
    sleep_norm = (min(sleep_hours, 9) / 9) * 100

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[glucose_norm, cholesterol_norm, oxygen_norm, heart_rate_norm, bmi_norm, sleep_norm],
        theta=['Glucose', 'Cholesterol', 'SpO₂', 'Heart Rate', 'BMI', 'Sleep'],
        fill='toself',
        name='Your Profile',
        line_color='#00aaff',
        fillcolor='rgba(0, 170, 255, 0.3)'
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e293b")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(t=30, b=10),
        showlegend=False,
    )
    
    chart1, chart2 = st.columns(2)
    
    with chart1:
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # Vitals Comparison Bar Chart
    with chart2:
        vitals_comparison = {
            'Metric': ['Glucose', 'Cholesterol', 'SpO₂', 'Heart Rate'],
            'Current': [glucose, cholesterol, oxygen, heart_rate],
            'Normal': [90, 180, 97, 72]
        }
        df_comparison = pd.DataFrame(vitals_comparison)
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            x=df_comparison['Metric'],
            y=df_comparison['Current'],
            name='Your Value',
            marker_color='#00aaff'
        ))
        fig_comparison.add_trace(go.Bar(
            x=df_comparison['Metric'],
            y=df_comparison['Normal'],
            name='Normal Range',
            marker_color='rgba(34,197,94,0.5)'
        ))
        
        fig_comparison.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            barmode='group',
            title="Your Vitals vs. Normal",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1e293b"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_comparison, use_container_width=True, config={"displayModeBar": False})

    # Health Risk Gauge
    st.markdown("---")
    section("🎯", "Risk Level Gauge")
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns([2, 2, 2])
    
    with gauge_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239,68,68,0.1)"},
                    {'range': [50, 70], 'color': "rgba(249,115,22,0.1)"},
                    {'range': [70, 100], 'color': "rgba(34,197,94,0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with gauge_col2:
        # Risk Score Breakdown
        risk_score_data = {
            'Factor': ['Vitals', 'Lifestyle', 'Age', 'Smoking'],
            'Impact': [health_score * 0.5, health_score * 0.25, health_score * 0.15, health_score * 0.1]
        }
        fig_breakdown = go.Figure(go.Pie(
            labels=risk_score_data['Factor'],
            values=risk_score_data['Impact'],
            marker_colors=['#00aaff', '#6366f1', '#f59e0b', '#ef4444']
        ))
        fig_breakdown.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            title="Score Breakdown",
            showlegend=True
        )
        st.plotly_chart(fig_breakdown, use_container_width=True, config={"displayModeBar": False})

    with gauge_col3:
        # Critical Metrics Alert
        critical_metrics = []
        if glucose < 70 or glucose > 150: critical_metrics.append(f"Glucose: {glucose}")
        if cholesterol > 240: critical_metrics.append(f"Cholesterol: {cholesterol}")
        if oxygen < 94: critical_metrics.append(f"SpO₂: {oxygen}%")
        if heart_rate < 50 or heart_rate > 100: critical_metrics.append(f"HR: {heart_rate}")
        
        st.markdown("**🚨 Alert Status**")
        if critical_metrics:
            for metric in critical_metrics:
                st.markdown(f'<div class="alert-chip">⚠️ {metric}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-chip">✅ All metrics within normal range</div>', unsafe_allow_html=True)

    # Risk Factor Impact Analysis
    st.markdown("---")
    section("🔍", "Detailed Risk Factor Analysis")

    risk_impact_data = {
        'Risk Factor': ['Smoking', 'High BMI', 'High Glucose', 'High Cholesterol', 'Low Oxygen', 'Poor Sleep'],
        'Present': [
            'Yes' if smoking else 'No',
            'Yes' if bmi > 30 else 'No',
            'Yes' if glucose > 150 else 'No',
            'Yes' if cholesterol > 240 else 'No',
            'Yes' if oxygen < 94 else 'No',
            'Yes' if sleep_hours < 6 else 'No'
        ],
        'Severity': [
            'Critical' if smoking else '-',
            'High' if bmi > 30 else '-',
            'High' if glucose > 150 else '-',
            'High' if cholesterol > 240 else '-',
            'Critical' if oxygen < 94 else '-',
            'High' if sleep_hours < 6 else '-'
        ],
        'Health Score Impact': [
            '-10' if smoking else '0',
            '-10' if bmi > 30 else '-5' if bmi > 25 else '0',
            '-15' if glucose > 150 else '-8' if glucose > 120 else '0',
            '-15' if cholesterol > 240 else '-8' if cholesterol > 200 else '0',
            '-15' if oxygen < 94 else '-8' if oxygen < 96 else '0',
            '-8' if sleep_hours < 6 else '-3' if sleep_hours < 7 else '0'
        ]
    }

    df_impact = pd.DataFrame(risk_impact_data)
    st.dataframe(df_impact, use_container_width=True, hide_index=True)

    # Save Assessment
    st.markdown("---")
    st.markdown("**💾 Save Assessment**")
    
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("📥 Save Health Assessment", use_container_width=True):
            record = {
                'worker_id': worker_id,
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'cholesterol': cholesterol,
                'oxygen': oxygen,
                'heart_rate': heart_rate,
                'health_score': health_score,
                'risk': 'High' if health_score < 50 else 'Medium' if health_score < 70 else 'Low',
                'confidence': min(95, 70 + health_score // 5),
                'timestamp': datetime.now().isoformat()
            }
            save_record(
    worker_id,
    record,
    record["risk"],
    record["confidence"]
)
            st.success("✅ Assessment saved successfully!")
    
    with save_col2:
        if st.button("📊 Generate PDF Report", use_container_width=True):
            st.info("📄 PDF report generation feature coming soon!")

# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 2 — WORKER HISTORY (ADVANCED)                       ║
# ╚══════════════════════════════════════════════════════════╝
if role == "👷 Worker":
   with tab2:
    section("📈", "Worker Health History & Analytics")

    all_workers = get_all_workers()

    if not all_workers:
        st.info("No history yet. Save a record from the Health Assessment tab first.")
    else:
        selected = st.selectbox("Select Worker", all_workers)
        df_hist = get_worker_history(selected)

        if df_hist.empty:
            st.warning("No records found.")
        else:
            st.markdown(f"**{len(df_hist)} record(s) found for `{selected}`**")

            # Latest record summary
            latest = df_hist.iloc[-1]
            badge_cls = f"trend-badge-{latest['risk']}"
            st.markdown(
                f"Latest risk: <span class='{badge_cls}'>{latest['risk'].upper()}</span> "
                f"&nbsp;({latest['confidence']}% confidence) &nbsp;·&nbsp; {latest['timestamp']}",
                unsafe_allow_html=True
            )

            # Summary Statistics
            st.markdown("---")
            st.markdown("**📊 Summary Statistics**")
            
            stat_cols = st.columns(4)
            numeric_cols = df_hist.select_dtypes(include='number').columns
            
            if 'glucose' in numeric_cols:
                with stat_cols[0]:
                    avg_glucose = df_hist['glucose'].mean()
                    min_glucose = df_hist['glucose'].min()
                    max_glucose = df_hist['glucose'].max()
                    st.metric(
                        label="Glucose (mg/dL)", 
                        value=f"{avg_glucose:.0f}",
                        delta=f"Range: {min_glucose:.0f}-{max_glucose:.0f}"
                    )
            
            if 'cholesterol' in numeric_cols:
                with stat_cols[1]:
                    avg_chol = df_hist['cholesterol'].mean()
                    st.metric(label="Cholesterol (mg/dL)", value=f"{avg_chol:.0f}")
            
            if 'oxygen' in numeric_cols:
                with stat_cols[2]:
                    avg_o2 = df_hist['oxygen'].mean()
                    st.metric(label="SpO₂ (%)", value=f"{avg_o2:.1f}")
            
            if 'heart_rate' in numeric_cols:
                with stat_cols[3]:
                    avg_hr = df_hist['heart_rate'].mean()
                    st.metric(label="Heart Rate (bpm)", value=f"{avg_hr:.0f}")

            st.markdown("")

            # Trend charts
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

            metrics_to_plot = ["glucose", "cholesterol", "oxygen", "heart_rate", "bmi"]
            metric_labels = ["Glucose (mg/dL)", "Cholesterol (mg/dL)", "SpO₂ (%)", "Heart Rate (bpm)", "BMI"]
            ref_lines = [100, 180, 97, 75, 22]
            colors = ["#00aaff", "#6366f1", "#10b981", "#f59e0b", "#ef4444"]

            col_a, col_b = st.columns(2)
            cols_cycle = [col_a, col_b, col_a, col_b, col_a]

            for metric, label, ref, col, color in zip(metrics_to_plot, metric_labels, ref_lines, cols_cycle, colors):
                if metric not in df_hist.columns:
                    continue
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_hist["timestamp"], y=df_hist[metric],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        fill='tozeroy',
                        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
                    ))
                    fig.add_hline(y=ref, line_dash="dash", line_color="#374151",
                                  annotation_text="Normal", annotation_position="bottom right")
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=300, margin=dict(t=30, b=10, l=10, r=10),
                        title=dict(text=label, font=dict(size=12, color="#94a3b8")),
                        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"),
                        showlegend=False,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Risk progression
            st.markdown("---")
            section("🎯", "Risk Level Progression")
            
            risk_map = {"Low": 0, "Medium": 1, "High": 2}
            df_hist["risk_num"] = df_hist["risk"].map(risk_map)
            fig_risk = go.Figure(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["risk_num"],
                mode="lines+markers",
                line=dict(color="#f97316", width=3),
                marker=dict(size=12, color=df_hist["risk_num"],
                            colorscale=[[0, "#22c55e"], [0.5, "#f97316"], [1, "#ef4444"]],
                            showscale=False),
                text=df_hist["risk"], hovertemplate="%{text}<extra></extra>",
                fill='tozeroy',
                fillcolor='rgba(249, 115, 22, 0.1)'
            ))
            fig_risk.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(t=30, b=10),
                title=dict(text="Risk Level Over Time", font=dict(size=13, color="#94a3b8")),
                yaxis=dict(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"], gridcolor="#1e293b"),
                xaxis=dict(showgrid=False),
                hovermode="x unified"
            )
            st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

            # Advanced Analytics
            st.markdown("---")
            section("🔬", "Advanced Analytics")

            # Correlation heatmap
            if len(df_hist) > 2:
                numeric_df = df_hist.select_dtypes(include='number')
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig_corr.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        margin=dict(l=100, b=100)
                    )
                    
                    st.markdown("**Metric Correlations**")
                    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

            # Distribution analysis
            st.markdown("---")
            st.markdown("**Distribution Analysis**")
            
            dist_cols = st.columns(3)
            dist_metrics = ["glucose", "cholesterol", "heart_rate"]
            
            for idx, metric in enumerate(dist_metrics):
                if metric in df_hist.columns:
                    with dist_cols[idx % 3]:
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=df_hist[metric],
                            nbinsx=8,
                            marker_color="#00aaff",
                            name=metric.title()
                        ))
                        fig_dist.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300,
                            title=f"{metric.title()} Distribution",
                            showlegend=False,
                            xaxis=dict(showgrid=False),
                            yaxis=dict(gridcolor="#1e293b")
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

            # Volatility Analysis
            st.markdown("---")
            st.markdown("**Volatility & Stability Index**")
            
            vol_data = {}
            for col in ['glucose', 'cholesterol', 'oxygen', 'heart_rate']:
                if col in df_hist.columns:
                    vol_data[col] = df_hist[col].std()
            
            if vol_data:
                fig_vol = go.Figure(go.Bar(
                    x=list(vol_data.keys()),
                    y=list(vol_data.values()),
                    marker_color=['#ef4444' if v > 15 else '#f97316' if v > 8 else '#22c55e' for v in vol_data.values()]
                ))
                fig_vol.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    title="Volatility Index (Standard Deviation)",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#1e293b")
                )
                st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

            # Raw data table
            st.markdown("---")
            with st.expander("📋 Raw History Table"):
                st.dataframe(df_hist.drop(columns=["id", "risk_num"], errors="ignore"),
                             use_container_width=True, hide_index=True)

# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 3 — MANAGEMENT DASHBOARD (ADVANCED)                 ║
# ╚══════════════════════════════════════════════════════════╝
if role == "🏢 Admin":
    with tab3:
         section("🏢", "Management Dashboard & Analytics")

    risk_df = get_risk_distribution()

    if risk_df.empty:
        st.info("No worker data yet. Save records from the Health Assessment tab to see analytics.")
    else:
        total = len(risk_df)
        n_high = len(risk_df[risk_df["risk"].str.lower() == "high"])
        n_med = len(risk_df[risk_df["risk"].str.lower() == "medium"])
        n_low = len(risk_df[risk_df["risk"].str.lower() == "low"])

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1: render_metric("Total Workers", total)
        with k2: render_metric("High Risk", n_high, "", 0, 1, 3)
        with k3: render_metric("Medium Risk", n_med)
        with k4: render_metric("Low Risk", n_low)

        # Risk Summary Stats
        st.markdown("---")
        st.markdown("**📊 Risk Distribution & Key Metrics**")
        
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.markdown(f'<div class="stat-box">🔴 <b>High Risk</b><br>{(n_high/total*100):.1f}%</div>', unsafe_allow_html=True)
        with summary_cols[1]:
            st.markdown(f'<div class="stat-box">🟠 <b>Medium Risk</b><br>{(n_med/total*100):.1f}%</div>', unsafe_allow_html=True)
        with summary_cols[2]:
            st.markdown(f'<div class="stat-box">🟢 <b>Low Risk</b><br>{(n_low/total*100):.1f}%</div>', unsafe_allow_html=True)
        with summary_cols[3]:
            st.markdown(f'<div class="stat-box">👥 <b>Total Monitored</b><br>{total}</div>', unsafe_allow_html=True)

        st.markdown("")

        pie_col, bar_col = st.columns(2)

        with pie_col:
            counts = risk_df["risk"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.55,
                marker_colors=["#ef4444" if l == "High" else "#f97316" if l == "Medium" else "#22c55e" for l in counts.index],
                textinfo="label+percent+value",
            ))
            fig_pie.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(t=30, b=10),
                title=dict(text="Risk Distribution Pie Chart", font=dict(size=13, color="#94a3b8")),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with bar_col:
            avg_df = get_avg_vitals()
            if not avg_df.empty:
                means = avg_df.select_dtypes(include="number").mean().round(1)
                normals = {"glucose": 100, "cholesterol": 180, "oxygen": 98, "heart_rate": 75, "bmi": 22}
                labels = {"glucose": "Glucose", "cholesterol": "Cholesterol", "oxygen": "SpO₂", "heart_rate": "HR", "bmi": "BMI"}
                keys = list(normals.keys())
                deviations = [round((means.get(k, 0) - normals[k]) / normals[k] * 100, 1) for k in keys]
                bar_c = ["#ef4444" if abs(d) > 15 else "#f97316" if abs(d) > 8 else "#22c55e" for d in deviations]
                fig_avg = go.Figure(go.Bar(
                    x=[labels[k] for k in keys], y=deviations,
                    marker_color=bar_c,
                    text=[f"{'+' if d >= 0 else ''}{d}%" for d in deviations],
                    textposition="outside",
                ))
                fig_avg.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=350, margin=dict(t=30, b=10),
                    title=dict(text="Avg Vitals vs Normal Range", font=dict(size=13, color="#94a3b8")),
                    yaxis=dict(gridcolor="#1e293b", zeroline=True, zerolinecolor="#374151"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_avg, use_container_width=True, config={"displayModeBar": False})

        # Trend Analysis
        st.markdown("---")
        section("📈", "Trend & Forecast Analysis")

        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            # Simulated trend data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            trend_data = pd.DataFrame({
                'date': dates,
                'high_risk_count': np.random.randint(2, 8, 30),
                'medium_risk_count': np.random.randint(3, 10, 30),
                'low_risk_count': np.random.randint(5, 15, 30),
            })
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['high_risk_count'],
                                           mode='lines+markers', name='High Risk', 
                                           line=dict(color='#ef4444', width=2),
                                           marker=dict(size=6),
                                           fill='tozeroy',
                                           fillcolor='rgba(239,68,68,0.1)'))
            fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['medium_risk_count'],
                                           mode='lines+markers', name='Medium Risk',
                                           line=dict(color='#f97316', width=2),
                                           marker=dict(size=6)))
            fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['low_risk_count'],
                                           mode='lines+markers', name='Low Risk',
                                           line=dict(color='#22c55e', width=2),
                                           marker=dict(size=6)))
            
            fig_trend.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(t=30, b=10),
                title=dict(text="30-Day Risk Trend", font=dict(size=13, color="#94a3b8")),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        with trend_col2:
            # Department breakdown
            departments = ['Refining', 'Processing', 'Maintenance', 'Admin', 'Safety']
            dept_high = [3, 2, 4, 1, 0]
            dept_medium = [5, 6, 3, 2, 1]
            dept_low = [12, 14, 8, 10, 5]
            
            fig_dept = go.Figure(data=[
                go.Bar(name='High Risk', y=departments, x=dept_high, orientation='h', marker_color='#ef4444'),
                go.Bar(name='Medium Risk', y=departments, x=dept_medium, orientation='h', marker_color='#f97316'),
                go.Bar(name='Low Risk', y=departments, x=dept_low, orientation='h', marker_color='#22c55e'),
            ])
            
            fig_dept.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(t=30, b=10),
                title=dict(text="Risk by Department", font=dict(size=13, color="#94a3b8")),
                barmode='stack', xaxis=dict(gridcolor="#1e293b"), yaxis=dict(showgrid=False),
                hovermode="y unified"
            )
            st.plotly_chart(fig_dept, use_container_width=True, config={"displayModeBar": False})

        # Age & Risk Correlation
        st.markdown("---")
        section("🔗", "Age & Health Profile Analysis")
        
        age_risk_col1, age_risk_col2 = st.columns(2)
        
        with age_risk_col1:
            age_groups = ['18-30', '31-40', '41-50', '51-60', '60+']
            age_counts = [8, 15, 12, 10, 5]
            age_risk = ['Low', 'Low', 'Medium', 'Medium', 'High']
            
            colors_age = ['#22c55e' if r == 'Low' else '#f97316' if r == 'Medium' else '#ef4444' for r in age_risk]
            
            fig_age = go.Figure(go.Bar(
                x=age_groups,
                y=age_counts,
                marker_color=colors_age,
                text=age_counts,
                textposition='outside'
            ))
            fig_age.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                title="Worker Distribution by Age Group",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#1e293b"),
                showlegend=False
            )
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

        with age_risk_col2:
            health_metrics_age = pd.DataFrame({
                'Age Group': age_groups,
                'Avg Glucose': [95, 102, 108, 115, 125],
                'Avg Cholesterol': [175, 185, 195, 215, 235],
                'Avg SpO2': [98, 97.5, 97, 96, 95]
            })
            
            st.markdown("**Age Group Health Metrics**")
            st.dataframe(health_metrics_age, use_container_width=True, hide_index=True)

        # High risk workers table
        st.markdown("---")
        section("🚨", "High Risk Workers — Action Required")
        hr_df = get_high_risk_workers()
        if hr_df.empty:
            st.markdown('<div class="ok-chip"><span>✅</span> No high-risk workers currently</div>', unsafe_allow_html=True)
        else:
            st.dataframe(hr_df, use_container_width=True, hide_index=True)

        # Recommendations
        st.markdown("---")
        section("💡", "Strategic Recommendations & Action Plan")
        
        rec_cols = st.columns(3)
        
        with rec_cols[0]:
            st.markdown('<div class="rec-box">🎯 <b>Immediate Actions</b><br>• Schedule health screening for high-risk workers<br>• Provide counseling for smoking cessation<br>• Monitor oxygen levels daily</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-box">📋 <b>Weekly Monitoring</b><br>• Check vitals 2-3 times per week<br>• Track medication adherence<br>• Monitor lifestyle changes</div>', unsafe_allow_html=True)
        
        with rec_cols[1]:
            st.markdown('<div class="rec-box">👨‍⚕️ <b>Medical Support</b><br>• Assign occupational health specialist<br>• Provide nutritionist consultation<br>• Arrange ECG/stress tests if needed</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-box">🌡️ <b>Environmental</b><br>• Review workplace heat controls<br>• Improve ventilation systems<br>• Adjust shift schedules as needed</div>', unsafe_allow_html=True)
        
        with rec_cols[2]:
            st.markdown('<div class="rec-box">🏆 <b>Preventive Programs</b><br>• Wellness workshops monthly<br>• Fitness classes for staff<br>• Stress management training</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-box">📊 <b>Analytics & Reporting</b><br>• Monthly health reports<br>• Trend analysis reviews<br>• ROI measurement</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    f"<p style='color:#334155;font-size:0.8rem;text-align:center'>"
    f"IOCL AI Healthcare Intelligence Platform &nbsp;·&nbsp; "
    f"Generated {datetime.now().strftime('%d %B %Y at %H:%M:%S')} &nbsp;·&nbsp; "
    f"For internal occupational health use only</p>",
    unsafe_allow_html=True
)