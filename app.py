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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Background ── */
.main { background: #f1f5f9; }

/* ── Constrain width ── */
.block-container {
    max-width: 1100px !important;
    padding: 0 24px 40px !important;
    margin: 0 auto !important;
}

/* ── Remove default streamlit spacing ── */
.stApp > header { display: none; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1e3a8a, #1565c0, #0277bd);
    padding: 24px 32px;
    text-align: center;
    color: white;
    margin: 0 -24px 24px;
}
.app-header h1 {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 5px;
    color: white !important;
}
.app-header p { font-size: 0.83rem; opacity: 0.75; margin: 0; }

/* ── White cards ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 10px rgba(0,0,0,0.04);
    margin-bottom: 14px;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1565c0;
    margin: 0 0 5px;
}
.card-sub {
    font-size: 0.79rem;
    color: #64748b;
    margin: 0 0 12px;
    line-height: 1.5;
}

/* ── Section label ── */
.sec {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 12px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Field labels ── */
.fl { font-size: 0.8rem; font-weight: 600; color: #1e293b; margin: 8px 0 1px; }
.fh { font-size: 0.71rem; color: #94a3b8; margin: 0 0 3px; }

/* ── Preset buttons ── */
div[data-testid="stHorizontalBlock"] div.stButton > button {
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    padding: 6px 2px !important;
    width: 100%;
    transition: all 0.15s ease !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    border-color: #1565c0 !important;
    color: #1565c0 !important;
    background: #eff6ff !important;
}

/* ── Predict button ── */
div.stButton > button[kind="primary"] {
    background: #1565c0 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 13px !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.3) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #0d47a1 !important;
}

/* ── Slider colour ── */
div[data-testid="stSlider"] > div > div > div > div { background: #1565c0 !important; }

/* ── Result boxes ── */
.res-bad {
    background: #dc2626;
    color: white;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(220,38,38,0.25);
}
.res-good {
    background: #16a34a;
    color: white;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(22,163,74,0.25);
}
.r-icon  { font-size: 2rem; margin-bottom: 3px; }
.r-title { font-size: 1.15rem; font-weight: 800; margin: 0 0 2px; }
.r-prob  { font-size: 2.3rem; font-weight: 800; margin: 2px 0 5px; line-height: 1; }
.r-sub   { font-size: 0.79rem; opacity: 0.88; line-height: 1.4; }

/* ── Severity ── */
.sev-h { background:#fef2f2; border:1.5px solid #fca5a5; border-radius:7px; padding:8px 12px; margin-top:8px; font-size:0.79rem; font-weight:600; color:#b91c1c; }
.sev-m { background:#fff7ed; border:1.5px solid #fb923c; border-radius:7px; padding:8px 12px; margin-top:8px; font-size:0.79rem; font-weight:600; color:#c2410c; }
.sev-l { background:#fefce8; border:1.5px solid #fde047; border-radius:7px; padding:8px 12px; margin-top:8px; font-size:0.79rem; font-weight:600; color:#854d0e; }

/* ── Advice ── */
.adv {
    background: #eff6ff;
    border-left: 3px solid #1565c0;
    border-radius: 0 7px 7px 0;
    padding: 10px 12px;
    margin-top: 10px;
}
.adv b { color:#1565c0; font-size:0.79rem; display:block; margin-bottom:4px; }
.adv ul { margin:0; padding-left:14px; color:#334155; }
.adv li { font-size:0.77rem; margin-bottom:2px; line-height:1.4; }
.adv-ok { background:#f0fdf4; border-left:3px solid #16a34a; border-radius:0 7px 7px 0; padding:10px 12px; margin-top:10px; }
.adv-ok b { color:#15803d; font-size:0.79rem; display:block; margin-bottom:4px; }
.adv-ok ul { margin:0; padding-left:14px; color:#334155; }
.adv-ok li { font-size:0.77rem; margin-bottom:2px; line-height:1.4; }

/* ── Awaiting ── */
.await {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 10px;
    padding: 28px 16px;
    text-align: center;
}
.aw-icon  { font-size: 2.2rem; margin-bottom: 8px; }
.aw-title { font-size: 0.9rem; font-weight: 700; color: #334155; margin-bottom: 4px; }
.aw-sub   { font-size: 0.78rem; color: #64748b; line-height: 1.55; }

/* ── Confidence bars ── */
.cbar { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.cbar-lbl { font-size:0.75rem; font-weight:600; color:#334155; min-width:105px; }
.cbar-bg  { flex:1; background:#e2e8f0; border-radius:99px; height:8px; overflow:hidden; }
.cbar-r   { background:#dc2626; height:100%; border-radius:99px; }
.cbar-g   { background:#16a34a; height:100%; border-radius:99px; }
.cbar-pct { font-size:0.77rem; font-weight:700; min-width:36px; text-align:right; }

/* ── Stats grid ── */
.sg { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:10px; }
.sc { background:#eff6ff; border-radius:8px; padding:9px 4px; text-align:center; }
.sn { font-size:1rem; font-weight:800; color:#1565c0; }
.sl { font-size:0.65rem; color:#64748b; margin-top:1px; }

/* ── How-it-works ── */
.how { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:10px; }
.hw  { background:#f8fafc; border-radius:8px; padding:10px 8px; text-align:center; }
.hwn { width:26px; height:26px; background:#1565c0; color:white; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; margin:0 auto 6px; }
.hwt { font-size:0.77rem; font-weight:700; color:#1e293b; margin-bottom:3px; }
.hwb { font-size:0.7rem; color:#64748b; line-height:1.35; }

/* ── About text ── */
.ab  { font-size:0.82rem; color:#475569; line-height:1.65; margin:0 0 8px; }
.ab strong { color:#1e3a8a; }

/* ── Disclaimer ── */
.disc { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:10px 12px; font-size:0.76rem; color:#78350f; line-height:1.5; }

/* ── Footer ── */
.footer { text-align:center; padding:12px; color:#94a3b8; font-size:0.72rem; border-top:1px solid #e2e8f0; background:white; margin:4px -24px 0; }
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

# ── Two balanced columns ──────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="medium")

# ════════════════════════════════════════════
# LEFT — Input Form
# ════════════════════════════════════════════
with left:
    st.markdown("""
    <div class="card">
        <div class="card-title">📋 Enter Student Session Details</div>
        <div class="card-sub">No technical knowledge needed — just answer the questions below.</div>
    """, unsafe_allow_html=True)

    # Presets
    st.markdown('<p style="font-size:0.76rem;font-weight:600;color:#64748b;margin:0 0 5px;">💡 Quick examples:</p>', unsafe_allow_html=True)
    pb1, pb2, pb3, pb4 = st.columns(4)
    preset = None
    with pb1:
        if st.button("😰 Stressed",  use_container_width=True): preset="stressed"
    with pb2:
        if st.button("😊 Relaxed",   use_container_width=True): preset="relaxed"
    with pb3:
        if st.button("📅 Deadline",  use_container_width=True): preset="deadline"
    with pb4:
        if st.button("🌟 Top",       use_container_width=True): preset="top"

    PD = {
        "stressed": dict(week=8,  day=0, sess=45, dtd=2,  pre=1, prog=60, ls=45, rs=42, stx=3, lag=4, na=4, ir=4.0, mcr=18.0),
        "relaxed":  dict(week=3,  day=2, sess=10, dtd=25, pre=0, prog=20, ls=72, rs=70, stx=1, lag=1, na=2, ir=0.8, mcr=2.5),
        "deadline": dict(week=10, day=4, sess=60, dtd=1,  pre=1, prog=80, ls=55, rs=52, stx=4, lag=5, na=5, ir=4.8, mcr=22.0),
        "top":      dict(week=5,  day=1, sess=20, dtd=14, pre=0, prog=50, ls=88, rs=85, stx=0, lag=0, na=3, ir=1.2, mcr=4.0),
    }
    p = PD.get(preset)

    st.markdown('<hr style="border:none;border-top:1px solid #f1f5f9;margin:10px 0 6px;">', unsafe_allow_html=True)

    # SESSION SECTION
    st.markdown('<div class="sec">📅 Session & Timeline</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="fl">Which week of the course?</div><div class="fh">Week 1 = start · Week 40 = end</div>', unsafe_allow_html=True)
        week_of_module = st.slider("_w", 0, 40, p["week"] if p else 5, label_visibility="collapsed")

        st.markdown('<div class="fl">Day of the week</div>', unsafe_allow_html=True)
        day_of_week = st.selectbox("_d", [0,1,2,3,4,5,6], index=p["day"] if p else 0,
                                   format_func=lambda x:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
                                   label_visibility="collapsed")

        st.markdown('<div class="fl">Total sessions so far</div><div class="fh">Each platform visit = 1 session</div>', unsafe_allow_html=True)
        session_number = st.number_input("_sn", 1, 500, p["sess"] if p else 10, label_visibility="collapsed")

    with b:
        st.markdown('<div class="fl">Days until next deadline</div><div class="fh">0 = today · 30+ = plenty of time</div>', unsafe_allow_html=True)
        days_to_deadline = st.slider("_dd", 0, 60, p["dtd"] if p else 14, label_visibility="collapsed")

        st.markdown('<div class="fl">In a pre-deadline rush?</div>', unsafe_allow_html=True)
        is_pre_deadline = st.selectbox("_pd", [0,1], index=p["pre"] if p else 0,
                                       format_func=lambda x:"✅ Yes — due soon" if x else "❌ No — plenty of time",
                                       label_visibility="collapsed")

        st.markdown('<div class="fl">Course progress (%)</div><div class="fh">0% = start · 100% = complete</div>', unsafe_allow_html=True)
        module_progress = st.slider("_mp", 0, 100, p["prog"] if p else 30, label_visibility="collapsed") / 100

    # ACADEMIC SECTION
    st.markdown('<div class="sec">📊 Academic Performance</div>', unsafe_allow_html=True)
    c, d = st.columns(2)
    with c:
        st.markdown('<div class="fl">Latest assignment score (out of 100)</div>', unsafe_allow_html=True)
        latest_score = st.slider("_ls", 0, 100, p["ls"] if p else 65, label_visibility="collapsed")

        st.markdown('<div class="fl">Average score (recent assignments)</div>', unsafe_allow_html=True)
        rolling_score = float(st.slider("_rs", 0, 100, p["rs"] if p else 62, label_visibility="collapsed"))

        st.markdown('<div class="fl">Is the score improving or dropping?</div>', unsafe_allow_html=True)
        ST_OPTS = ["Improving a lot","Improving slightly","Staying steady","Dropping slightly","Dropping a lot"]
        stlbl = st.selectbox("_st", ST_OPTS, index=p["stx"] if p else 2, label_visibility="collapsed")
        score_trend = {"Improving a lot":8.0,"Improving slightly":3.0,"Staying steady":0.0,"Dropping slightly":-3.0,"Dropping a lot":-8.0}[stlbl]

    with d:
        st.markdown('<div class="fl">Avg days late submitting work</div><div class="fh">0 = always on time · 5+ = often late</div>', unsafe_allow_html=True)
        avg_submission_lag = float(st.slider("_al", 0, 15, p["lag"] if p else 2, label_visibility="collapsed"))

        st.markdown('<div class="fl">Assignments completed so far</div>', unsafe_allow_html=True)
        n_assessments = st.number_input("_na", 0, 20, p["na"] if p else 3, label_visibility="collapsed")

    # ONLINE ACTIVITY SECTION
    st.markdown('<div class="sec">💻 Online Activity</div>', unsafe_allow_html=True)
    e, f_ = st.columns(2)
    with e:
        st.markdown('<div class="fl">Session intensity (1–5)</div><div class="fh">1 = light browsing · 5 = very heavy</div>', unsafe_allow_html=True)
        intensity_ratio = st.slider("_ir", 0.0, 5.0, p["ir"] if p else 1.0, step=0.1, label_visibility="collapsed")
    with f_:
        st.markdown('<div class="fl">Clicks per minute</div><div class="fh">1–5 = normal · 10+ = very rapid</div>', unsafe_allow_html=True)
        mean_click_rate = st.slider("_mcr", 0.0, 50.0, p["mcr"] if p else 3.0, step=0.5, label_visibility="collapsed")

    # Activity defaults
    AD = {
        "stressed":{"act_subpage":18,"act_homepage":8,"act_oucontent":12,"act_resource":5},
        "relaxed": {"act_subpage":4, "act_homepage":2,"act_oucontent":5, "act_resource":2},
        "deadline":{"act_subpage":25,"act_homepage":12,"act_oucontent":15,"act_resource":8},
        "top":     {"act_subpage":6, "act_homepage":3,"act_oucontent":8, "act_resource":3},
    }
    av = AD.get(preset, {}) if preset else {}
    act_values = {feat: av.get(feat, 0) for feat in features if feat.startswith("act_")}

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Fatigue Level", type="primary", use_container_width=True)


# ════════════════════════════════════════════
# RIGHT — Info + Results
# ════════════════════════════════════════════
with right:

    # ABOUT
    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 About Digital Fatigue</div>
        <p class="ab"><strong>Digital fatigue</strong> is mental exhaustion that builds up when students
        spend prolonged periods on online learning platforms. Unlike physical tiredness, it often goes
        unnoticed by tutors until grades drop or students disengage entirely.</p>
        <p class="ab">Common signs include <strong>rapid unfocused clicking</strong>, repeated homepage
        visits, and jumping between content pages — all detectable from VLE clickstream data without
        any specialist hardware.</p>
        <p class="ab" style="margin:0;">This AI was trained on <strong>10.6 million real VLE interactions</strong>
        from 26,074 Open University students, achieving <strong>93.8% prediction accuracy</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # HOW IT WORKS
    st.markdown("""
    <div class="card">
        <div class="card-title">⚙️ How It Works</div>
        <div class="how">
            <div class="hw"><div class="hwn">1</div><div class="hwt">Enter Details</div><div class="hwb">Fill in student session info on the left</div></div>
            <div class="hw"><div class="hwn">2</div><div class="hwt">33 Signals</div><div class="hwb">AI reads click patterns, deadlines &amp; scores</div></div>
            <div class="hw"><div class="hwn">3</div><div class="hwt">XGBoost</div><div class="hwb">Model trained on 1.8M sessions predicts fatigue</div></div>
            <div class="hw"><div class="hwn">4</div><div class="hwt">Result</div><div class="hwb">Prediction + probability + advice</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PREDICTION RESULT
    st.markdown('<div class="card"><div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        inp = {
            "week_of_module":week_of_module, "day_of_week":day_of_week,
            "days_to_deadline":days_to_deadline, "is_pre_deadline":is_pre_deadline,
            "module_progress":module_progress, "latest_score":latest_score,
            "rolling_score":rolling_score, "score_trend":score_trend,
            "avg_submission_lag":avg_submission_lag, "n_assessments":n_assessments,
            "intensity_ratio":intensity_ratio, "session_number":session_number,
            "mean_click_rate":mean_click_rate, **act_values
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
                <div class="r-sub">This student is showing signs of digital fatigue.<br>Early intervention is recommended.</div>
            </div>""", unsafe_allow_html=True)
            if prob > 0.85:
                st.markdown('<div class="sev-h">🔴 High Risk — Immediate check-in recommended</div>', unsafe_allow_html=True)
            elif prob > 0.65:
                st.markdown('<div class="sev-m">🟠 Moderate Risk — Monitor closely</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sev-l">🟡 Low-Moderate — Keep an eye on engagement</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="adv"><b>💡 Recommended Actions</b><ul>
                <li>Reach out to check how the student is feeling</li>
                <li>Consider extending upcoming deadlines</li>
                <li>Encourage breaks between sessions (Pomodoro technique)</li>
                <li>Suggest offline study to reduce screen time</li>
            </ul></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-good">
                <div class="r-icon">✅</div>
                <div class="r-title">NO FATIGUE DETECTED</div>
                <div class="r-prob">{(1-prob)*100:.0f}%</div>
                <div class="r-sub">Student is engaging normally.<br>No immediate intervention required.</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="adv-ok"><b>✅ All Looking Good!</b><ul>
                <li>Continue monitoring — fatigue can develop near deadlines</li>
                <li>Encourage a healthy study schedule with regular breaks</li>
            </ul></div>""", unsafe_allow_html=True)

        # Confidence bars
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.77rem;font-weight:700;color:#334155;margin:0 0 7px;">📈 Confidence</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cbar">
            <span class="cbar-lbl">🟢 Not Fatigued</span>
            <div class="cbar-bg"><div class="cbar-g" style="width:{(1-prob)*100:.0f}%"></div></div>
            <span class="cbar-pct" style="color:#15803d">{(1-prob)*100:.1f}%</span>
        </div>
        <div class="cbar">
            <span class="cbar-lbl">🔴 Fatigued</span>
            <div class="cbar-bg"><div class="cbar-r" style="width:{prob*100:.0f}%"></div></div>
            <span class="cbar-pct" style="color:#dc2626">{prob*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        # Top features table
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.77rem;font-weight:700;color:#334155;margin:0 0 5px;">🔑 Key Factors</p>', unsafe_allow_html=True)
        FLABELS = {
            "act_subpage":"Subpage clicks","act_homepage":"Homepage revisits",
            "act_oucontent":"Content views","act_resource":"Resource downloads",
            "act_quiz":"Quiz attempts","act_forumng":"Forum activity",
            "intensity_ratio":"Session intensity","mean_click_rate":"Click speed",
            "days_to_deadline":"Days to deadline","week_of_module":"Course week",
            "session_number":"Session count","module_progress":"Progress",
            "latest_score":"Latest score","rolling_score":"Avg score",
            "score_trend":"Score trend","avg_submission_lag":"Sub. lateness",
        }
        fi = pd.DataFrame({
            "Signal": [FLABELS.get(feat, feat.replace("act_","").title()) for feat in features],
            "Importance": [f"{v*100:.1f}%" for v in model.feature_importances_],
            "Value": [round(inp[feat],1) for feat in features]
        }).sort_values("Importance", ascending=False).head(6).reset_index(drop=True)
        st.dataframe(fi, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="await">
            <div class="aw-icon">🔍</div>
            <div class="aw-title">Awaiting Prediction</div>
            <div class="aw-sub">Fill in the student details on the left,<br>then click <strong>Predict Fatigue Level</strong>.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # MODEL INFO
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 Model Information</div>
        <p class="ab">This app uses an <strong>XGBoost</strong> model trained on the <strong>Open University
        Learning Analytics Dataset (OULAD)</strong>. Three models were trained and compared — Logistic
        Regression, Random Forest, and XGBoost. XGBoost achieved the best performance across all metrics
        using 33 behavioural signals per session. No physical sensors or specialist hardware required.</p>
        <div class="sg">
            <div class="sc"><div class="sn">93.8%</div><div class="sl">Accuracy</div></div>
            <div class="sc"><div class="sn">0.976</div><div class="sl">AUC-ROC</div></div>
            <div class="sc"><div class="sn">0.872</div><div class="sl">F1 Score</div></div>
            <div class="sc"><div class="sn">10.6M</div><div class="sl">Interactions</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # DISCLAIMER
    st.markdown("""
    <div class="disc">
        ⚠️ <strong>Disclaimer:</strong> This is a research prototype for MSc dissertation purposes only.
        Predictions should <em>support</em>, not replace, professional human judgement. Not for real
        clinical or academic decision-making about individual students.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    🎓 <strong>MSc Data Science Dissertation</strong> &nbsp;·&nbsp;
    XGBoost &nbsp;·&nbsp; OULAD Dataset &nbsp;·&nbsp; AUC: 0.9763 &nbsp;·&nbsp; Accuracy: 93.8% &nbsp;·&nbsp; 26,074 Students<br>
    <span style="font-style:italic;">For educational and research purposes only.</span>
</div>
""", unsafe_allow_html=True)
