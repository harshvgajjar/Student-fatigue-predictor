import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Digital Fatigue Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
}

h1, h2, h3 { font-family: 'Nunito', sans-serif !important; }

.main { background-color: #f0f4ff; }
.block-container { padding-top: 2rem !important; max-width: 1100px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 50%, #0288d1 100%);
    border-radius: 24px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -30px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero h1 { font-size: 2.4rem; font-weight: 900; margin: 0 0 12px; color: white !important; }
.hero p  { font-size: 1.1rem; opacity: 0.9; margin: 0; line-height: 1.6; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    margin-bottom: 16px;
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Info cards */
.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 2rem; }
.info-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(26,35,126,0.08);
    border-top: 4px solid #1565c0;
}
.info-card .icon { font-size: 2rem; margin-bottom: 8px; }
.info-card h3 { font-size: 1rem; font-weight: 800; color: #1a237e; margin: 0 0 6px; }
.info-card p  { font-size: 0.85rem; color: #666; margin: 0; line-height: 1.5; }

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-card .num { font-size: 1.8rem; font-weight: 900; color: #1565c0; font-family: 'Nunito', sans-serif; }
.stat-card .lbl { font-size: 0.78rem; color: #888; margin-top: 2px; }

/* Section */
.section-box {
    background: white;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(26,35,126,0.06);
}
.section-title {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #1a237e;
    margin: 0 0 6px;
}
.section-sub { color: #666; font-size: 0.9rem; margin: 0 0 24px; }

/* Preset buttons */
.preset-label { font-size: 0.85rem; color: #555; margin-bottom: 8px; font-weight: 600; }

/* Result cards */
.result-fatigued {
    background: linear-gradient(135deg, #c62828, #e53935);
    color: white;
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(198,40,40,0.3);
    margin: 1rem 0;
}
.result-ok {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white;
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(27,94,32,0.3);
    margin: 1rem 0;
}
.result-icon { font-size: 3.5rem; margin-bottom: 8px; }
.result-title { font-size: 2rem; font-weight: 900; margin: 0 0 8px; font-family: 'Nunito', sans-serif; }
.result-prob  { font-size: 3rem; font-weight: 900; margin: 8px 0; font-family: 'Nunito', sans-serif; }
.result-desc  { font-size: 1rem; opacity: 0.9; margin: 0; line-height: 1.6; }

/* Advice box */
.advice-box {
    background: #e8f4fd;
    border-left: 4px solid #1565c0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 16px;
}
.advice-box h4 { color: #1565c0; margin: 0 0 8px; font-family: 'Nunito', sans-serif; }
.advice-box ul { margin: 0; padding-left: 20px; color: #444; }
.advice-box li { margin-bottom: 4px; font-size: 0.92rem; }

/* Input styling */
.stSlider > div > div > div { background: #1565c0 !important; }
.stSelectbox > div { border-radius: 10px !important; }

/* Question label */
.q-label {
    font-weight: 700;
    color: #1a237e;
    font-size: 0.95rem;
    margin-bottom: 4px;
    font-family: 'Nunito', sans-serif;
}
.q-hint { font-size: 0.8rem; color: #888; margin-bottom: 8px; }

/* How it works */
.how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.how-card {
    text-align: center;
    padding: 20px 16px;
    background: #f8f9ff;
    border-radius: 14px;
}
.how-num {
    width: 36px; height: 36px;
    background: #1565c0;
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800;
    margin: 0 auto 12px;
    font-family: 'Nunito', sans-serif;
    font-size: 1rem;
}
.how-card h4 { font-size: 0.9rem; font-weight: 700; color: #1a237e; margin: 0 0 6px; }
.how-card p  { font-size: 0.8rem; color: #666; margin: 0; line-height: 1.5; }

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: #888;
    font-size: 0.82rem;
    border-top: 1px solid #e0e0e0;
    margin-top: 2rem;
}

/* Divider */
.divider { border: none; border-top: 2px solid #f0f0f0; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model    = joblib.load("xgb_clean.pkl")
    scaler   = joblib.load("scaler_safe.pkl")
    features = json.load(open("features_safe.json"))
    results  = json.load(open("results.json"))
    return model, scaler, features, results

model, scaler, features, results = load_models()
xgb = results["xgb"]

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎓 MSc Data Science Dissertation · University Project</div>
    <h1>🧠 Student Digital Fatigue Predictor</h1>
    <p>This tool uses Artificial Intelligence to detect signs of <strong>digital fatigue</strong> in students
    based on how they interact with their online learning platform.<br>
    Simply answer a few questions about a student's study session to get an instant prediction.</p>
</div>
""", unsafe_allow_html=True)

# ── WHAT IS DIGITAL FATIGUE ──────────────────────────────────────────────────
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="icon">😴</div>
        <h3>What is Digital Fatigue?</h3>
        <p>Digital fatigue happens when students become mentally exhausted from too much screen time and online learning activity, making it hard to concentrate or retain information.</p>
    </div>
    <div class="info-card">
        <div class="icon">📚</div>
        <h3>Why Does It Matter?</h3>
        <p>Fatigued students are more likely to disengage, submit work late, or drop out. Early detection allows tutors and parents to provide support before performance drops.</p>
    </div>
    <div class="info-card">
        <div class="icon">🤖</div>
        <h3>How Does the AI Work?</h3>
        <p>Our AI was trained on over <strong>10 million learning interactions</strong> from 26,074 real students. It recognises patterns that indicate fatigue with 93.8% accuracy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MODEL STATS ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
    <div class="stat-card">
        <div class="num">93.8%</div>
        <div class="lbl">Prediction Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="num">0.976</div>
        <div class="lbl">AUC-ROC Score</div>
    </div>
    <div class="stat-card">
        <div class="num">10.6M</div>
        <div class="lbl">Learning Interactions</div>
    </div>
    <div class="stat-card">
        <div class="num">26,074</div>
        <div class="lbl">Students Studied</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HOW IT WORKS ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-box">
    <div class="section-title">⚙️ How It Works</div>
    <div class="section-sub">Four simple steps from data to prediction</div>
    <div class="how-grid">
        <div class="how-card">
            <div class="how-num">1</div>
            <h4>Collect Session Data</h4>
            <p>Information about a student's online study session is gathered — how long, what they clicked, deadlines etc.</p>
        </div>
        <div class="how-card">
            <div class="how-num">2</div>
            <h4>Extract 33 Features</h4>
            <p>The AI analyses 33 behavioural signals including click patterns, deadline pressure, and assessment scores.</p>
        </div>
        <div class="how-card">
            <div class="how-num">3</div>
            <h4>XGBoost Prediction</h4>
            <p>Our trained XGBoost model — the same technology used by Google and Amazon — processes the signals instantly.</p>
        </div>
        <div class="how-card">
            <div class="how-num">4</div>
            <h4>Get a Result</h4>
            <p>You receive a clear prediction with a probability score and personalised advice for the student.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PREDICTION FORM ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-box">
    <div class="section-title">📋 Predict a Student's Fatigue Level</div>
    <div class="section-sub">Answer the questions below — no technical knowledge needed!</div>
""", unsafe_allow_html=True)

# Preset profiles
st.markdown('<div class="preset-label">💡 Quick Start — Try an example student profile:</div>', unsafe_allow_html=True)
pc1, pc2, pc3, pc4 = st.columns(4)
preset = None
with pc1:
    if st.button("😰 Stressed Student", use_container_width=True):
        preset = "stressed"
with pc2:
    if st.button("😊 Relaxed Student", use_container_width=True):
        preset = "relaxed"
with pc3:
    if st.button("📅 Pre-Deadline Crunch", use_container_width=True):
        preset = "deadline"
with pc4:
    if st.button("🌟 Top Performer", use_container_width=True):
        preset = "top"

# Define presets
presets = {
    "stressed":  dict(week=8,  day=0, sess=45, dtd=2,  pre=1, prog=0.6, ls=45, rs=42.0, st_=-5.0, lag=4.0, na=4, ir=4.0, mcr=18.0),
    "relaxed":   dict(week=3,  day=2, sess=10, dtd=25, pre=0, prog=0.2, ls=72, rs=70.0, st_=2.0,  lag=1.0, na=2, ir=0.8, mcr=2.5),
    "deadline":  dict(week=10, day=4, sess=60, dtd=1,  pre=1, prog=0.8, ls=55, rs=52.0, st_=-8.0, lag=5.0, na=5, ir=4.8, mcr=22.0),
    "top":       dict(week=5,  day=1, sess=20, dtd=14, pre=0, prog=0.5, ls=88, rs=85.0, st_=3.0,  lag=0.5, na=3, ir=1.2, mcr=4.0),
}

p = presets.get(preset, presets["relaxed"]) if preset else None

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── FORM INPUTS ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🗓 About the Study Session")

    st.markdown('<div class="q-label">Which week of the course is the student in?</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-hint">Week 1 = just started, Week 40 = nearly finished</div>', unsafe_allow_html=True)
    week_of_module = st.slider("", 0, 40, p["week"] if p else 5, key="week", label_visibility="collapsed")

    st.markdown('<div class="q-label">What day of the week is it?</div>', unsafe_allow_html=True)
    day_of_week = st.selectbox("", [0,1,2,3,4,5,6],
                                index=p["day"] if p else 0,
                                format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
                                key="day", label_visibility="collapsed")

    st.markdown('<div class="q-label">How many study sessions has the student had so far?</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-hint">A "session" is one visit to the online learning platform</div>', unsafe_allow_html=True)
    session_number = st.number_input("", min_value=1, max_value=500, value=p["sess"] if p else 10, key="sess", label_visibility="collapsed")

    st.markdown("#### ⏰ Deadlines & Progress")

    st.markdown('<div class="q-label">How many days until the next assignment deadline?</div>', unsafe_allow_html=True)
    days_to_deadline = st.slider("", 0, 60, p["dtd"] if p else 14, key="dtd", label_visibility="collapsed")

    st.markdown('<div class="q-label">Is the student in a pre-deadline rush period?</div>', unsafe_allow_html=True)
    is_pre_deadline = st.selectbox("", [0, 1],
                                    index=p["pre"] if p else 0,
                                    format_func=lambda x: "✅ Yes — assignment due soon" if x else "❌ No — plenty of time",
                                    key="pre", label_visibility="collapsed")

    st.markdown('<div class="q-label">How far through the course is the student? (0% = just started, 100% = finished)</div>', unsafe_allow_html=True)
    module_progress = st.slider("", 0, 100, int((p["prog"] if p else 0.3)*100), key="prog", label_visibility="collapsed") / 100

with col2:
    st.markdown("#### 📊 Academic Performance")

    st.markdown('<div class="q-label">What was the student\'s most recent assignment score? (out of 100)</div>', unsafe_allow_html=True)
    latest_score = st.slider("", 0, 100, p["ls"] if p else 65, key="ls", label_visibility="collapsed")

    st.markdown('<div class="q-label">What is the student\'s average score across recent assignments?</div>', unsafe_allow_html=True)
    rolling_score = st.slider("", 0, 100, int(p["rs"] if p else 62), key="rs", label_visibility="collapsed") * 1.0

    st.markdown('<div class="q-label">Is their score going up or down recently?</div>', unsafe_allow_html=True)
    score_trend_label = st.selectbox("", ["Improving a lot", "Improving slightly", "Staying the same", "Dropping slightly", "Dropping a lot"],
                                      index=2 if not p else (0 if p["st_"] > 3 else (1 if p["st_"] > 0 else (2 if p["st_"] == 0 else (3 if p["st_"] > -5 else 4)))),
                                      key="st_label", label_visibility="collapsed")
    score_trend_map = {"Improving a lot": 8.0, "Improving slightly": 3.0, "Staying the same": 0.0, "Dropping slightly": -3.0, "Dropping a lot": -8.0}
    score_trend = score_trend_map[score_trend_label]

    st.markdown('<div class="q-label">On average, how many days late does the student submit work?</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-hint">0 = always on time, 5+ = regularly late</div>', unsafe_allow_html=True)
    avg_submission_lag = st.slider("", 0, 15, int(p["lag"] if p else 2), key="lag", label_visibility="collapsed") * 1.0

    st.markdown('<div class="q-label">How many assignments have they completed so far?</div>', unsafe_allow_html=True)
    n_assessments = st.number_input("", 0, 20, p["na"] if p else 3, key="na", label_visibility="collapsed")

    st.markdown("#### 💻 Online Activity Intensity")

    st.markdown('<div class="q-label">How intensely is the student clicking and navigating? (1 = light, 5 = very heavy)</div>', unsafe_allow_html=True)
    intensity_ratio = st.slider("", 0.0, 5.0, p["ir"] if p else 1.0, step=0.1, key="ir", label_visibility="collapsed")

    st.markdown('<div class="q-label">How quickly is the student clicking on average per minute?</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-hint">1-5 = normal browsing, 10+ = very rapid clicking</div>', unsafe_allow_html=True)
    mean_click_rate = st.slider("", 0.0, 50.0, p["mcr"] if p else 3.0, step=0.5, key="mcr", label_visibility="collapsed")

# Activity values — set sensible defaults
act_defaults = {
    "stressed":  {"act_subpage": 18, "act_homepage": 8, "act_oucontent": 12, "act_resource": 5, "act_quiz": 3},
    "relaxed":   {"act_subpage": 4,  "act_homepage": 2, "act_oucontent": 5,  "act_resource": 2, "act_quiz": 1},
    "deadline":  {"act_subpage": 25, "act_homepage": 12,"act_oucontent": 15, "act_resource": 8, "act_quiz": 4},
    "top":       {"act_subpage": 6,  "act_homepage": 3, "act_oucontent": 8,  "act_resource": 3, "act_quiz": 2},
}
act_vals = act_defaults.get(preset, {}) if preset else {}
act_values = {}
for f in features:
    if f.startswith("act_"):
        act_values[f] = act_vals.get(f, 0)

st.markdown("</div>", unsafe_allow_html=True)  # close section-box

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Check Fatigue Level Now", type="primary", use_container_width=True)

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

    input_df     = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    prob         = float(model.predict_proba(input_scaled)[0][1])
    pred         = int(prob >= 0.5)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        if pred == 1:
            st.markdown(f"""
            <div class="result-fatigued">
                <div class="result-icon">⚠️</div>
                <div class="result-title">DIGITAL FATIGUE DETECTED</div>
                <div class="result-prob">{prob*100:.0f}%</div>
                <div class="result-desc">The AI predicts this student is showing signs of digital fatigue.<br>
                Early support could prevent further disengagement.</div>
            </div>""", unsafe_allow_html=True)

            # Severity
            if prob > 0.85:
                severity = "🔴 High Risk"
                sev_color = "#c62828"
                sev_desc = "Immediate check-in recommended"
            elif prob > 0.65:
                severity = "🟠 Moderate Risk"
                sev_color = "#e65100"
                sev_desc = "Monitor closely over next few sessions"
            else:
                severity = "🟡 Low-Moderate Risk"
                sev_color = "#f57f17"
                sev_desc = "Keep an eye on engagement levels"

            st.markdown(f"""
            <div style="background:{sev_color}22; border:1px solid {sev_color}44; border-radius:10px; padding:12px 16px; margin-top:12px; text-align:center;">
                <strong style="color:{sev_color}">{severity}</strong> — {sev_desc}
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="advice-box">
                <h4>💡 Recommended Actions</h4>
                <ul>
                    <li>Reach out to the student to check how they are feeling</li>
                    <li>Consider reducing workload or extending upcoming deadlines</li>
                    <li>Encourage breaks between study sessions (Pomodoro technique)</li>
                    <li>Suggest offline study alternatives to reduce screen time</li>
                    <li>Review whether academic support or counselling may help</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ok">
                <div class="result-icon">✅</div>
                <div class="result-title">NO FATIGUE DETECTED</div>
                <div class="result-prob">{(1-prob)*100:.0f}%</div>
                <div class="result-desc">The AI predicts this student is engaging normally and shows<br>
                no significant signs of digital fatigue at this time.</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="advice-box" style="background:#e8f5e9; border-color:#2e7d32;">
                <h4 style="color:#2e7d32;">✅ All Looking Good!</h4>
                <ul>
                    <li>Continue monitoring — fatigue can develop quickly near deadlines</li>
                    <li>Encourage maintaining a healthy study schedule</li>
                    <li>Regular breaks and good sleep remain important even when doing well</li>
                </ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence bars
    st.markdown("#### 📊 Confidence Breakdown")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.metric("🟢 Not Fatigued Probability", f"{(1-prob)*100:.1f}%")
        st.progress(float(1 - prob))
    with bc2:
        st.metric("🔴 Fatigued Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

    # Key factors
    st.markdown("#### 🔑 Key Factors That Influenced This Prediction")
    st.markdown("*These are the behavioural signals the AI considered most important:*")

    importances = model.feature_importances_
    feature_labels = {
        "act_subpage": "Subpage clicks", "act_homepage": "Homepage visits",
        "act_resource": "Resource downloads", "act_oucontent": "Content page views",
        "act_quiz": "Quiz attempts", "act_forumng": "Forum activity",
        "mean_click_rate": "Click speed", "intensity_ratio": "Session intensity",
        "days_to_deadline": "Days to deadline", "week_of_module": "Course week",
        "session_number": "Session count", "module_progress": "Course progress",
        "latest_score": "Latest score", "rolling_score": "Average score",
        "score_trend": "Score trend", "avg_submission_lag": "Submission lateness",
        "is_pre_deadline": "Pre-deadline period", "day_of_week": "Day of week",
        "n_assessments": "Assessments completed",
    }

    feat_imp = pd.DataFrame({
        "Behavioural Signal": [feature_labels.get(f, f.replace("act_","").replace("_"," ").title()) for f in features],
        "Importance (%)": [round(i*100, 1) for i in importances],
        "Student's Value": [round(input_dict[f], 2) for f in features]
    }).sort_values("Importance (%)", ascending=False).head(8).reset_index(drop=True)

    st.dataframe(feat_imp, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-box">
    <div class="section-title">🏆 AI Model Performance</div>
    <div class="section-sub">Three machine learning models were trained and compared. XGBoost performed best.</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
model_info = [
    ("xgb", "⭐ XGBoost", "#1565c0", "Best performing model — chosen for this app"),
    ("rf",  "🌲 Random Forest", "#2e7d32", "Good accuracy, uses many decision trees"),
    ("lr",  "📈 Logistic Regression", "#e65100", "Simple baseline model"),
]
for col, (key, name, color, desc) in zip([mc1, mc2, mc3], model_info):
    r = results[key]
    with col:
        st.markdown(f"""
        <div style="background:white; border-radius:14px; padding:20px; border-top:4px solid {color}; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-weight:800; font-family:'Nunito',sans-serif; font-size:1rem; color:{color}; margin-bottom:4px;">{name}</div>
            <div style="font-size:0.78rem; color:#888; margin-bottom:12px;">{desc}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:900; color:{color}; font-family:'Nunito',sans-serif;">{r['test_acc']*100:.1f}%</div>
                    <div style="font-size:0.72rem; color:#888;">Accuracy</div>
                </div>
                <div style="background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:900; color:{color}; font-family:'Nunito',sans-serif;">{r['test_auc']:.3f}</div>
                    <div style="font-size:0.72rem; color:#888;">AUC-ROC</div>
                </div>
                <div style="background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:900; color:{color}; font-family:'Nunito',sans-serif;">{r['test_f1']:.3f}</div>
                    <div style="font-size:0.72rem; color:#888;">F1 Score</div>
                </div>
                <div style="background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:900; color:{color}; font-family:'Nunito',sans-serif;">{r['test_rec']:.3f}</div>
                    <div style="font-size:0.72rem; color:#888;">Recall</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── ABOUT DATASET ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-box">
    <div class="section-title">📚 About the Dataset</div>
    <div class="section-sub">This AI was trained on the Open University Learning Analytics Dataset (OULAD) — one of the largest publicly available educational datasets in the world.</div>
    <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:12px; margin-top:8px;">
        <div style="background:#e8eaf6; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:1.5rem; font-weight:900; color:#1a237e; font-family:'Nunito',sans-serif;">10.6M</div>
            <div style="font-size:0.8rem; color:#555;">VLE Interactions</div>
        </div>
        <div style="background:#e8eaf6; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:1.5rem; font-weight:900; color:#1a237e; font-family:'Nunito',sans-serif;">26,074</div>
            <div style="font-size:0.8rem; color:#555;">Students</div>
        </div>
        <div style="background:#e8eaf6; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:1.5rem; font-weight:900; color:#1a237e; font-family:'Nunito',sans-serif;">7</div>
            <div style="font-size:0.8rem; color:#555;">Course Modules</div>
        </div>
        <div style="background:#e8eaf6; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:1.5rem; font-weight:900; color:#1a237e; font-family:'Nunito',sans-serif;">1.8M</div>
            <div style="font-size:0.8rem; color:#555;">Study Sessions</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🎓 <strong>MSc Data Science Dissertation</strong> &nbsp;·&nbsp;
    Built with XGBoost &amp; Streamlit &nbsp;·&nbsp;
    Dataset: OULAD (Open University) &nbsp;·&nbsp;
    AUC: 0.9763 &nbsp;·&nbsp; Accuracy: 93.8%
    <br><br>
    <em>This tool is for educational and research purposes only.
    Predictions should be used to support, not replace, human judgement.</em>
</div>
""", unsafe_allow_html=True)
