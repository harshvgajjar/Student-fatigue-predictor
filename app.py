import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Fatigue Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fc; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4f8ef7;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #4f8ef7; }
    .metric-card p  { margin: 4px 0 0; color: #888; font-size: 0.85rem; }

    .result-fatigued {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(238,90,36,0.3);
    }
    .result-ok {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(32,191,107,0.3);
    }
    .result-title { font-size: 2rem; font-weight: 700; margin: 0; }
    .result-sub   { font-size: 1rem; margin: 8px 0 0; opacity: 0.9; }

    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #4f8ef7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 8px;
        padding-bottom: 4px;
        border-bottom: 2px solid #e8f0fe;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a9e 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #cce0ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model   = joblib.load("models/xgb_clean.pkl")
    scaler  = joblib.load("models/scaler_safe.pkl")
    features = json.load(open("models/features_safe.json"))
    results  = json.load(open("models/results.json"))
    return model, scaler, features, results

model, scaler, features, results = load_models()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Fatigue Predictor")
    st.markdown("*MSc Data Science Dissertation*")
    st.markdown("---")

    st.markdown("### 📊 Model Performance")
    xgb = results["xgb"]
    st.markdown(f"**Model:** XGBoost")
    st.markdown(f"**Accuracy:** {xgb['test_acc']*100:.1f}%")
    st.markdown(f"**AUC-ROC:** {xgb['test_auc']:.4f}")
    st.markdown(f"**F1 Score:** {xgb['test_f1']:.4f}")
    st.markdown(f"**Precision:** {xgb['test_prec']:.4f}")
    st.markdown(f"**Recall:** {xgb['test_rec']:.4f}")
    st.markdown("---")

    st.markdown("### 🏆 All Models")
    for key, label, color in [("xgb","XGBoost ⭐","#4f8ef7"),("rf","Random Forest","#26de81"),("lr","Logistic Reg.","#fd9644")]:
        r = results[key]
        st.markdown(f"**{label}**")
        st.markdown(f"AUC: `{r['test_auc']:.4f}` | F1: `{r['test_f1']:.4f}`")

    st.markdown("---")
    st.markdown("*Dataset: OULAD (10.6M interactions)*")
    st.markdown("*Students: 26,074 | Modules: 7*")

# ── Main page ──────────────────────────────────────────────────
st.title("🎓 Student Digital Fatigue Predictor")
st.markdown("Predict whether a student is experiencing **digital fatigue** based on their VLE session behaviour.")
st.markdown("---")

# ── Model metrics row ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><h2>{xgb['test_acc']*100:.1f}%</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><h2>{xgb['test_auc']:.3f}</h2><p>AUC-ROC</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><h2>{xgb['test_f1']:.3f}</h2><p>F1 Score</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><h2>33</h2><p>Features Used</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────
st.markdown("## 📝 Enter Session Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">⏱ Temporal Features</div>', unsafe_allow_html=True)
    week_of_module   = st.slider("Week of Module", 0, 40, 5)
    day_of_week      = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    session_number   = st.number_input("Session Number", min_value=1, max_value=500, value=10)

    st.markdown('<div class="section-header">📅 Deadline Features</div>', unsafe_allow_html=True)
    days_to_deadline = st.slider("Days to Next Deadline", 0, 60, 14)
    is_pre_deadline  = st.selectbox("Is Pre-Deadline Period?", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
    module_progress  = st.slider("Module Progress (0–1)", 0.0, 1.0, 0.3, step=0.01)

    st.markdown('<div class="section-header">📊 Assessment Features</div>', unsafe_allow_html=True)
    latest_score     = st.slider("Latest Score", 0, 100, 65)
    rolling_score    = st.slider("Rolling Avg Score", 0.0, 100.0, 62.0, step=0.5)
    score_trend      = st.slider("Score Trend", -50.0, 50.0, 0.0, step=0.5)
    avg_submission_lag = st.slider("Avg Submission Lag (days)", 0.0, 30.0, 2.0, step=0.5)
    n_assessments    = st.number_input("Number of Assessments", 0, 20, 3)

with col2:
    st.markdown('<div class="section-header">💻 Session Behaviour</div>', unsafe_allow_html=True)
    intensity_ratio  = st.slider("Intensity Ratio", 0.0, 5.0, 1.0, step=0.1)
    mean_click_rate  = st.slider("Mean Click Rate", 0.0, 50.0, 3.0, step=0.5)

    st.markdown('<div class="section-header">🖱 Activity Type Clicks</div>', unsafe_allow_html=True)
    act_cols = [f for f in features if f.startswith("act_")]

    act_values = {}
    rows = [act_cols[i:i+2] for i in range(0, len(act_cols), 2)]
    for row in rows:
        rc1, rc2 = st.columns(2)
        with rc1:
            label = row[0].replace("act_", "").capitalize()
            act_values[row[0]] = st.number_input(label, 0, 500, 0, key=row[0])
        if len(row) > 1:
            with rc2:
                label = row[1].replace("act_", "").capitalize()
                act_values[row[1]] = st.number_input(label, 0, 500, 0, key=row[1])

# ── Predict button ─────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Fatigue", type="primary", use_container_width=True)

if predict_btn:
    input_dict = {
        "week_of_module":     week_of_module,
        "day_of_week":        day_of_week,
        "days_to_deadline":   days_to_deadline,
        "is_pre_deadline":    is_pre_deadline,
        "module_progress":    module_progress,
        "latest_score":       latest_score,
        "rolling_score":      rolling_score,
        "score_trend":        score_trend,
        "avg_submission_lag": avg_submission_lag,
        "n_assessments":      n_assessments,
        "intensity_ratio":    intensity_ratio,
        "session_number":     session_number,
        "mean_click_rate":    mean_click_rate,
        **act_values
    }

    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= 0.5)

    st.markdown("## 🎯 Prediction Result")
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        if pred == 1:
            st.markdown(f"""
            <div class="result-fatigued">
                <p class="result-title">⚠️ FATIGUED</p>
                <p class="result-sub">Fatigue probability: <strong>{prob*100:.1f}%</strong></p>
                <p class="result-sub">This student shows signs of digital fatigue.<br>Consider recommending a break or reducing workload.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ok">
                <p class="result-title">✅ NOT FATIGUED</p>
                <p class="result-sub">Fatigue probability: <strong>{prob*100:.1f}%</strong></p>
                <p class="result-sub">This student appears to be engaging normally.<br>No immediate intervention required.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bar
    st.markdown("### 📈 Confidence")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Not Fatigued", f"{(1-prob)*100:.1f}%")
        st.progress(float(1 - prob))
    with p2:
        st.metric("Fatigued", f"{prob*100:.1f}%")
        st.progress(float(prob))

    # Top features
    st.markdown("### 🔑 Top Contributing Features")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances,
        "Your Value": [input_dict[f] for f in features]
    }).sort_values("Importance", ascending=False).head(10)

    st.dataframe(feat_imp.reset_index(drop=True), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>MSc Data Science Dissertation · OULAD Dataset · XGBoost Model · AUC 0.9763</small></center>",
    unsafe_allow_html=True
)
