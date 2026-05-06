import streamlit as st
import joblib, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Digital Fatigue Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.main { background: #f1f5f9; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* HEADER */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1565c0 60%, #0288d1 100%);
    padding: 26px 40px;
    text-align: center;
    color: white;
}
.app-header h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 6px; color: white !important; letter-spacing: -0.3px; }
.app-header p  { font-size: 0.88rem; opacity: 0.82; margin: 0; }

/* CARD */
.card {
    background: white;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 6px 16px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1565c0;
    margin: 0 0 6px;
}
.card-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0 0 14px;
    line-height: 1.5;
}

/* FIELD LABELS */
.fl  { font-size: 0.83rem; font-weight: 600; color: #1e293b; margin: 10px 0 2px; }
.fh  { font-size: 0.73rem; color: #94a3b8; margin: 0 0 3px; }

/* SECTION LABEL */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 14px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #f1f5f9;
}

/* PRESET BUTTONS */
div[data-testid="stHorizontalBlock"] div.stButton > button {
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
    border-radius: 8px !important;
    font-size: 0.77rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    padding: 7px 4px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    border-color: #1565c0 !important;
    color: #1565c0 !important;
    background: #eff6ff !important;
}

/* PREDICT BUTTON */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0, #0288d1) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px !important;
    width: 100% !important;
    box-shadow: 0 3px 10px rgba(21,101,192,0.35) !important;
    letter-spacing: 0.2px;
}

/* SLIDER TRACK */
div[data-testid="stSlider"] > div > div > div > div { background: #1565c0 !important; }

/* RESULT — FATIGUED */
.res-bad {
    background: linear-gradient(135deg, #b91c1c, #dc2626);
    color: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(185,28,28,0.28);
}
/* RESULT — OK */
.res-good {
    background: linear-gradient(135deg, #15803d, #16a34a);
    color: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(21,128,61,0.28);
}
.r-icon  { font-size: 2.4rem; margin-bottom: 4px; }
.r-title { font-size: 1.3rem; font-weight: 800; margin: 0 0 3px; }
.r-prob  { font-size: 2.6rem; font-weight: 800; margin: 3px 0 6px; line-height: 1; }
.r-sub   { font-size: 0.82rem; opacity: 0.9; line-height: 1.5; }

/* SEVERITY BADGES */
.sev-h { background:#fef2f2; border:1.5px solid #fca5a5; border-radius:8px; padding:9px 13px; margin-top:10px; font-size:0.82rem; font-weight:600; color:#b91c1c; }
.sev-m { background:#fff7ed; border:1.5px solid #fb923c; border-radius:8px; padding:9px 13px; margin-top:10px; font-size:0.82rem; font-weight:600; color:#c2410c; }
.sev-l { background:#fefce8; border:1.5px solid #fde047; border-radius:8px; padding:9px 13px; margin-top:10px; font-size:0.82rem; font-weight:600; color:#854d0e; }

/* ADVICE BOX */
.advice {
    background: #f0f9ff;
    border-left: 3px solid #1565c0;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    margin-top: 12px;
}
.advice b { color: #1565c0; font-size: 0.82rem; display: block; margin-bottom: 5px; }
.advice ul { margin: 0; padding-left: 15px; color: #334155; }
.advice li { font-size: 0.79rem; margin-bottom: 3px; line-height: 1.4; }

.advice-ok {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    margin-top: 12px;
}
.advice-ok b { color: #15803d; font-size: 0.82rem; display: block; margin-bottom: 5px; }
.advice-ok ul { margin: 0; padding-left: 15px; color: #334155; }
.advice-ok li { font-size: 0.79rem; margin-bottom: 3px; line-height: 1.4; }

/* AWAITING */
.awaiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 32px 20px;
    text-align: center;
}
.aw-icon  { font-size: 2.8rem; margin-bottom: 10px; }
.aw-title { font-size: 0.98rem; font-weight: 700; color: #334155; margin-bottom: 5px; }
.aw-sub   { font-size: 0.82rem; color: #64748b; line-height: 1.6; }

/* STATS GRID */
.sg { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:12px; }
.sc { background:#eff6ff; border-radius:8px; padding:10px 6px; text-align:center; }
.sn { font-size:1.1rem; font-weight:800; color:#1565c0; }
.sl { font-size:0.68rem; color:#64748b; margin-top:1px; font-weight:500; }

/* ABOUT TEXT */
.about-p { font-size:0.85rem; color:#475569; line-height:1.7; margin:0 0 8px; }
.about-p strong { color:#1e3a8a; }

/* DISCLAIMER */
.disc { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:11px 14px; font-size:0.78rem; color:#78350f; line-height:1.5; }

/* CONFIDENCE BAR */
.cbar-row { display:flex; align-items:center; gap:10px; margin-bottom:7px; }
.cbar-lbl { font-size:0.78rem; font-weight:600; color:#334155; min-width:115px; }
.cbar-bg  { flex:1; background:#e2e8f0; border-radius:99px; height:9px; overflow:hidden; }
.cbar-red   { background:#dc2626; height:100%; border-radius:99px; }
.cbar-green { background:#16a34a; height:100%; border-radius:99px; }
.cbar-pct { font-size:0.8rem; font-weight:700; min-width:38px; text-align:right; }

/* FOOTER */
.footer { text-align:center; padding:14px; color:#94a3b8; font-size:0.75rem; border-top:1px solid #e2e8f0; background:white; margin-top:4px; }

/* How it works steps */
.how-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px; }
.how-item { background:#f8fafc; border-radius:10px; padding:12px; text-align:center; }
.how-num { width:28px;height:28px;background:#1565c0;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;margin:0 auto 8px; }
.how-title { font-size:0.8rem; font-weight:700; color:#1e293b; margin-bottom:4px; }
.how-body { font-size:0.73rem; color:#64748b; line-height:1.4; }
</style>
""", unsafe_allow_html=True)

# ── Load ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = joblib.load("xgb_clean.pkl")
    s = joblib.load("scaler_safe.pkl")
    f = json.load(open("features_safe.json"))
    r = json.load(open("results.json"))
    return m, s, f, r
model, scaler, features, results = load_models()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 Student Digital Fatigue Predictor</h1>
    <p>AI-powered early detection of digital fatigue in online learners &nbsp;·&nbsp; MSc Data Science Dissertation &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; AUC 0.976</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Main columns ──────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="medium")

# ════════════════════════════════════════════
# LEFT COLUMN
# ════════════════════════════════════════════
with left:
    st.markdown("""
    <div class="card">
    <div class="card-title">📋 Enter Student Session Details</div>
    <div class="card-sub">Answer the questions below — no technical knowledge needed.
    Each field has a short hint to help you.</div>
    """, unsafe_allow_html=True)

    # ── Presets ──
    st.markdown('<p style="font-size:0.78rem;font-weight:600;color:#475569;margin:0 0 6px;">💡 Try a quick example profile:</p>', unsafe_allow_html=True)
    pb1, pb2, pb3, pb4 = st.columns(4)
    preset = None
    with pb1:
        if st.button("😰 Stressed",  use_container_width=True): preset = "stressed"
    with pb2:
        if st.button("😊 Relaxed",   use_container_width=True): preset = "relaxed"
    with pb3:
        if st.button("📅 Deadline",  use_container_width=True): preset = "deadline"
    with pb4:
        if st.button("🌟 Top",       use_container_width=True): preset = "top"

    PD = {
        "stressed": dict(week=8,  day=0, sess=45, dtd=2,  pre=1, prog=60, ls=45, rs=42, stx=3, lag=4, na=4, ir=4.0, mcr=18.0),
        "relaxed":  dict(week=3,  day=2, sess=10, dtd=25, pre=0, prog=20, ls=72, rs=70, stx=1, lag=1, na=2, ir=0.8, mcr=2.5),
        "deadline": dict(week=10, day=4, sess=60, dtd=1,  pre=1, prog=80, ls=55, rs=52, stx=4, lag=5, na=5, ir=4.8, mcr=22.0),
        "top":      dict(week=5,  day=1, sess=20, dtd=14, pre=0, prog=50, ls=88, rs=85, stx=0, lag=0, na=3, ir=1.2, mcr=4.0),
    }
    p = PD.get(preset)

    st.markdown('<hr style="border:none;border-top:1px solid #f1f5f9;margin:12px 0;">', unsafe_allow_html=True)

    # ── SESSION INFO ──
    st.markdown('<div class="sec-label">📅 Session & Timeline</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="fl">Which week of the course?</div><div class="fh">Week 1 = just started · Week 40 = nearly finished</div>', unsafe_allow_html=True)
        week_of_module = st.slider("", 0, 40, p["week"] if p else 5, key="wk", label_visibility="collapsed")

        st.markdown('<div class="fl">Day of the week</div>', unsafe_allow_html=True)
        day_of_week = st.selectbox("", [0,1,2,3,4,5,6], index=p["day"] if p else 0,
                                    format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
                                    key="dw", label_visibility="collapsed")

        st.markdown('<div class="fl">Total sessions so far</div><div class="fh">Each visit to the online platform = 1 session</div>', unsafe_allow_html=True)
        session_number = st.number_input("", 1, 500, p["sess"] if p else 10, key="sn", label_visibility="collapsed")

    with cb:
        st.markdown('<div class="fl">Days until next assignment deadline</div><div class="fh">0 = deadline is today · 30+ = plenty of time</div>', unsafe_allow_html=True)
        days_to_deadline = st.slider("", 0, 60, p["dtd"] if p else 14, key="dd", label_visibility="collapsed")

        st.markdown('<div class="fl">Is the student in a pre-deadline rush?</div>', unsafe_allow_html=True)
        is_pre_deadline = st.selectbox("", [0,1], index=p["pre"] if p else 0,
                                        format_func=lambda x: "✅  Yes — assignment due soon" if x else "❌  No — plenty of time",
                                        key="pd", label_visibility="collapsed")

        st.markdown('<div class="fl">How far through the course? (%)</div><div class="fh">0% = just started · 100% = completed</div>', unsafe_allow_html=True)
        module_progress = st.slider("", 0, 100, p["prog"] if p else 30, key="mp", label_visibility="collapsed") / 100

    # ── ACADEMIC PERFORMANCE ──
    st.markdown('<div class="sec-label">📊 Academic Performance</div>', unsafe_allow_html=True)
    cd, ce = st.columns(2)
    with cd:
        st.markdown('<div class="fl">Most recent assignment score (out of 100)</div>', unsafe_allow_html=True)
        latest_score = st.slider("", 0, 100, p["ls"] if p else 65, key="ls", label_visibility="collapsed")

        st.markdown('<div class="fl">Average score across recent assignments</div>', unsafe_allow_html=True)
        rolling_score = float(st.slider("", 0, 100, p["rs"] if p else 62, key="rs", label_visibility="collapsed"))

        st.markdown('<div class="fl">Is their score improving or dropping?</div>', unsafe_allow_html=True)
        ST_OPTS = ["Improving a lot","Improving slightly","Staying steady","Dropping slightly","Dropping a lot"]
        stlabel = st.selectbox("", ST_OPTS, index=p["stx"] if p else 2, key="stl", label_visibility="collapsed")
        score_trend = {"Improving a lot":8.0,"Improving slightly":3.0,"Staying steady":0.0,"Dropping slightly":-3.0,"Dropping a lot":-8.0}[stlabel]

    with ce:
        st.markdown('<div class="fl">Average days late submitting work</div><div class="fh">0 = always on time · 5+ = frequently late</div>', unsafe_allow_html=True)
        avg_submission_lag = float(st.slider("", 0, 15, p["lag"] if p else 2, key="al", label_visibility="collapsed"))

        st.markdown('<div class="fl">Number of assignments completed so far</div>', unsafe_allow_html=True)
        n_assessments = st.number_input("", 0, 20, p["na"] if p else 3, key="na", label_visibility="collapsed")

    # ── ONLINE BEHAVIOUR ──
    st.markdown('<div class="sec-label">💻 Online Activity Intensity</div>', unsafe_allow_html=True)
    cf, cg = st.columns(2)
    with cf:
        st.markdown('<div class="fl">How intensely is the student clicking & navigating?</div><div class="fh">1 = light casual browsing · 5 = very heavy, rapid activity</div>', unsafe_allow_html=True)
        intensity_ratio = st.slider("", 0.0, 5.0, p["ir"] if p else 1.0, step=0.1, key="ir", label_visibility="collapsed")
    with cg:
        st.markdown('<div class="fl">Average number of clicks per minute</div><div class="fh">1–5 = normal pace · 10+ = very rapid clicking</div>', unsafe_allow_html=True)
        mean_click_rate = st.slider("", 0.0, 50.0, p["mcr"] if p else 3.0, step=0.5, key="mcr", label_visibility="collapsed")

    # ── Activity defaults from preset ──
    AD = {
        "stressed": {"act_subpage":18,"act_homepage":8, "act_oucontent":12,"act_resource":5},
        "relaxed":  {"act_subpage":4, "act_homepage":2, "act_oucontent":5, "act_resource":2},
        "deadline": {"act_subpage":25,"act_homepage":12,"act_oucontent":15,"act_resource":8},
        "top":      {"act_subpage":6, "act_homepage":3, "act_oucontent":8, "act_resource":3},
    }
    av = AD.get(preset, {}) if preset else {}
    act_values = {f: av.get(f, 0) for f in features if f.startswith("act_")}

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Fatigue Level", type="primary", use_container_width=True)


# ════════════════════════════════════════════
# RIGHT COLUMN
# ════════════════════════════════════════════
with right:

    # ── ABOUT ────────────────────────────────
    st.markdown("""
    <div class="card">
    <div class="card-title">🧠 About Digital Fatigue</div>
    <p class="about-p"><strong>Digital fatigue</strong> is the mental exhaustion that builds up when students
    spend prolonged periods interacting with online learning platforms. Unlike physical tiredness, it often goes
    unnoticed by tutors until grades start to drop or a student disengages entirely.</p>
    <p class="about-p">Common signs include <strong>rapid unfocused clicking</strong>, repeated homepage visits,
    jumping between content pages, and increased session intensity — all detectable from VLE clickstream data.</p>
    <p class="about-p">This AI was trained on <strong>10.6 million real VLE interactions</strong> from 26,074
    Open University students across 7 course modules, achieving <strong>93.8% prediction accuracy</strong>
    without any physical sensors or specialist hardware.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────
    st.markdown("""
    <div class="card">
    <div class="card-title">⚙️ How It Works</div>
    <div class="how-grid">
        <div class="how-item">
            <div class="how-num">1</div>
            <div class="how-title">Enter Details</div>
            <div class="how-body">Fill in student session behaviour on the left</div>
        </div>
        <div class="how-item">
            <div class="how-num">2</div>
            <div class="how-title">33 Signals</div>
            <div class="how-body">AI analyses click patterns, deadlines &amp; scores</div>
        </div>
        <div class="how-item">
            <div class="how-num">3</div>
            <div class="how-title">XGBoost Model</div>
            <div class="how-body">Trained on 1.8M sessions detects fatigue patterns</div>
        </div>
        <div class="how-item">
            <div class="how-num">4</div>
            <div class="how-title">Instant Result</div>
            <div class="how-body">Prediction + probability + recommended actions</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PREDICTION RESULT ─────────────────────
    st.markdown('<div class="card"><div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        inp = {
            "week_of_module": week_of_module, "day_of_week": day_of_week,
            "days_to_deadline": days_to_deadline, "is_pre_deadline": is_pre_deadline,
            "module_progress": module_progress, "latest_score": latest_score,
            "rolling_score": rolling_score, "score_trend": score_trend,
            "avg_submission_lag": avg_submission_lag, "n_assessments": n_assessments,
            "intensity_ratio": intensity_ratio, "session_number": session_number,
            "mean_click_rate": mean_click_rate, **act_values
        }
        df   = pd.DataFrame([inp])[features]
        prob = float(model.predict_proba(scaler.transform(df))[0][1])
        pred = int(prob >= 0.5)

        if pred == 1:
            st.markdown(f"""
            <div class="res-bad">
                <div class="r-icon">⚠️</div>
                <div class="r-title">DIGITAL FATIGUE DETECTED</div>
                <div class="r-prob">{prob*100:.0f}%</div>
                <div class="r-sub">This student is showing clear signs of digital fatigue.<br>
                Early intervention is strongly recommended.</div>
            </div>""", unsafe_allow_html=True)

            if prob > 0.85:
                st.markdown('<div class="sev-h">🔴 High Risk — Immediate check-in with the student recommended</div>', unsafe_allow_html=True)
            elif prob > 0.65:
                st.markdown('<div class="sev-m">🟠 Moderate Risk — Monitor closely over the next few sessions</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sev-l">🟡 Low-Moderate Risk — Keep an eye on engagement levels</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="advice">
                <b>💡 Recommended Actions for Tutors / Parents</b>
                <ul>
                    <li>Reach out to the student to check how they are feeling</li>
                    <li>Consider reducing workload or extending upcoming deadlines</li>
                    <li>Encourage short breaks between study sessions (Pomodoro technique)</li>
                    <li>Suggest offline study materials to reduce screen time</li>
                    <li>Discuss whether academic or wellbeing support would help</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-good">
                <div class="r-icon">✅</div>
                <div class="r-title">NO FATIGUE DETECTED</div>
                <div class="r-prob">{(1-prob)*100:.0f}%</div>
                <div class="r-sub">This student is engaging normally with the platform.<br>
                No immediate intervention required at this time.</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="advice-ok">
                <b>✅ All Looking Good!</b>
                <ul>
                    <li>Continue regular monitoring — fatigue can develop quickly near deadlines</li>
                    <li>Encourage the student to maintain a healthy study schedule</li>
                    <li>Regular breaks remain important even when engagement is high</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        # Confidence bars
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;font-weight:700;color:#334155;margin:0 0 8px;">📈 Confidence Breakdown</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cbar-row">
            <span class="cbar-lbl">🟢 Not Fatigued</span>
            <div class="cbar-bg"><div class="cbar-green" style="width:{(1-prob)*100:.0f}%"></div></div>
            <span class="cbar-pct" style="color:#15803d">{(1-prob)*100:.1f}%</span>
        </div>
        <div class="cbar-row">
            <span class="cbar-lbl">🔴 Fatigued</span>
            <div class="cbar-bg"><div class="cbar-red" style="width:{prob*100:.0f}%"></div></div>
            <span class="cbar-pct" style="color:#dc2626">{prob*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        # Top features
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;font-weight:700;color:#334155;margin:0 0 6px;">🔑 Key Factors That Influenced This Prediction</p>', unsafe_allow_html=True)
        FLABELS = {
            "act_subpage":"Subpage clicks","act_homepage":"Homepage revisits",
            "act_oucontent":"Content page views","act_resource":"Resource downloads",
            "act_quiz":"Quiz attempts","act_forumng":"Forum activity",
            "intensity_ratio":"Session intensity","mean_click_rate":"Click speed",
            "days_to_deadline":"Days to deadline","week_of_module":"Course week",
            "session_number":"Session count","module_progress":"Course progress",
            "latest_score":"Latest score","rolling_score":"Average score",
            "score_trend":"Score trend","avg_submission_lag":"Submission lateness",
            "is_pre_deadline":"Pre-deadline period","day_of_week":"Day of week",
            "n_assessments":"Assessments done",
        }
        fi = pd.DataFrame({
            "Behavioural Signal": [FLABELS.get(f, f.replace("act_","").title()) for f in features],
            "Importance (%)": [round(v*100,1) for v in model.feature_importances_],
            "Student Value": [round(inp[f],1) for f in features]
        }).sort_values("Importance (%)", ascending=False).head(7).reset_index(drop=True)
        st.dataframe(fi, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="awaiting">
            <div class="aw-icon">🔍</div>
            <div class="aw-title">Awaiting Prediction</div>
            <div class="aw-sub">Fill in the student session details on the left,<br>
            then click <strong>Predict Fatigue Level</strong> to get an instant result.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    # ── MODEL INFO ────────────────────────────
    st.markdown(f"""
    <div class="card">
    <div class="card-title">🤖 Model Information</div>
    <p class="about-p">This app uses an <strong>XGBoost</strong> gradient boosting model — the same
    technology behind many real-world AI systems at Google and Amazon. It was trained on the
    <strong>Open University Learning Analytics Dataset (OULAD)</strong>, one of the largest
    publicly available educational datasets in the world.</p>
    <p class="about-p">The model analyses <strong>33 behavioural signals</strong> per session,
    including click patterns, content navigation, deadline proximity, and academic scores.
    No physical sensors, webcams, or specialist hardware are required.</p>
    <p class="about-p">Three models were trained and compared — Logistic Regression, Random Forest,
    and XGBoost. <strong>XGBoost achieved the best performance</strong> across all metrics:</p>
    <div class="sg">
        <div class="sc"><div class="sn">93.8%</div><div class="sl">Accuracy</div></div>
        <div class="sc"><div class="sn">0.976</div><div class="sl">AUC-ROC</div></div>
        <div class="sc"><div class="sn">0.872</div><div class="sl">F1 Score</div></div>
        <div class="sc"><div class="sn">10.6M</div><div class="sl">Interactions</div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DISCLAIMER ────────────────────────────
    st.markdown("""
    <div class="disc">
    ⚠️ <strong>Disclaimer:</strong> This application is a research prototype developed for
    MSc dissertation purposes only. Predictions should be used to <em>support</em>, not replace,
    professional human judgement. It should not be used as the sole basis for real academic or
    clinical decisions about individual students.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    🎓 <strong>MSc Data Science Dissertation</strong> &nbsp;·&nbsp;
    XGBoost Model &nbsp;·&nbsp; OULAD Dataset (Open University Learning Analytics) &nbsp;·&nbsp;
    AUC: 0.9763 &nbsp;·&nbsp; Accuracy: 93.8% &nbsp;·&nbsp; 26,074 Students &nbsp;·&nbsp; 7 Modules<br>
    <span style="margin-top:3px;display:block;font-style:italic;">
    For educational and research purposes only. Not for clinical or academic decision-making.</span>
</div>
""", unsafe_allow_html=True)
