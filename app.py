import streamlit as st
import joblib, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Digital Fatigue Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Lato:wght@300;400;700&display=swap');
html,body,[class*="css"]{font-family:'Lato',sans-serif;}
h1,h2,h3{font-family:'Nunito',sans-serif!important;}
.main{background-color:#f0f4ff;}
.block-container{padding-top:1rem!important;max-width:1200px;}
.hbar{background:linear-gradient(135deg,#1a237e,#1565c0,#0288d1);border-radius:16px;padding:18px 26px;color:white;margin-bottom:1.2rem;}
.hbar h1{font-size:1.6rem;font-weight:900;margin:0;color:white!important;}
.hbar p{font-size:0.86rem;opacity:0.85;margin:4px 0 0;}
.about-box{background:white;border-radius:16px;padding:20px 22px;box-shadow:0 2px 12px rgba(26,35,126,0.07);}
.about-box h4{font-family:'Nunito',sans-serif;font-weight:800;color:#1565c0;font-size:1rem;margin:0 0 10px;}
.about-box p{color:#475569;font-size:0.88rem;line-height:1.6;margin:0 0 8px;}
.result-waiting{background:#f8faff;border:2px dashed #c7d2fe;border-radius:16px;padding:28px;text-align:center;color:#94a3b8;font-size:0.9rem;}
.result-fatigued{background:linear-gradient(135deg,#c62828,#e53935);color:white;border-radius:16px;padding:24px;text-align:center;box-shadow:0 6px 24px rgba(198,40,40,0.25);}
.result-ok{background:linear-gradient(135deg,#1b5e20,#2e7d32);color:white;border-radius:16px;padding:24px;text-align:center;box-shadow:0 6px 24px rgba(27,94,32,0.25);}
.result-title{font-size:1.5rem;font-weight:900;margin:0 0 4px;font-family:'Nunito',sans-serif;}
.result-prob{font-size:2.4rem;font-weight:900;margin:4px 0;font-family:'Nunito',sans-serif;}
.result-desc{font-size:0.88rem;opacity:0.92;margin:0;line-height:1.5;}
.advice-box{background:#eff6ff;border-left:4px solid #1565c0;border-radius:0 10px 10px 0;padding:12px 16px;margin-top:10px;}
.advice-box h5{color:#1565c0;margin:0 0 6px;font-family:'Nunito',sans-serif;font-size:0.9rem;}
.advice-box ul{margin:0;padding-left:18px;color:#334155;}
.advice-box li{font-size:0.82rem;margin-bottom:3px;}
.mbox{background:white;border-radius:16px;padding:18px 22px;box-shadow:0 2px 12px rgba(26,35,126,0.07);}
.mbox h4{font-family:'Nunito',sans-serif;font-weight:800;color:#1565c0;font-size:1rem;margin:0 0 10px;}
.stat-row{display:flex;gap:10px;margin-bottom:10px;}
.stat-chip{background:#e8eaf6;border-radius:8px;padding:6px 12px;text-align:center;flex:1;}
.stat-chip .n{font-size:1.1rem;font-weight:900;color:#1a237e;font-family:'Nunito',sans-serif;}
.stat-chip .l{font-size:0.7rem;color:#666;}
.disc{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:0 10px 10px 0;padding:10px 14px;font-size:0.78rem;color:#78350f;}
.ql{font-weight:700;color:#1a237e;font-size:0.88rem;margin-bottom:2px;font-family:'Nunito',sans-serif;}
.qh{font-size:0.75rem;color:#94a3b8;margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    m = joblib.load("xgb_clean.pkl")
    s = joblib.load("scaler_safe.pkl")
    f = json.load(open("features_safe.json"))
    r = json.load(open("results.json"))
    return m, s, f, r

model, scaler, features, results = load_models()

st.markdown('<div class="hbar"><h1>🧠 Student Digital Fatigue Predictor</h1><p>AI-powered detection of digital fatigue in online learners &nbsp;·&nbsp; MSc Data Science Dissertation &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; AUC 0.976</p></div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div style="background:white;border-radius:16px;padding:20px;box-shadow:0 2px 16px rgba(26,35,126,0.08);"><h3 style="font-family:Nunito,sans-serif;font-weight:800;color:#1a237e;font-size:1.05rem;margin:0 0 4px;">📋 Enter Student Session Details</h3><p style="color:#64748b;font-size:0.82rem;margin:0 0 12px;">No technical knowledge needed — just answer the questions!</p>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.8rem;color:#555;font-weight:700;margin-bottom:6px;">💡 Try an example:</div>', unsafe_allow_html=True)
    pc1,pc2,pc3,pc4 = st.columns(4)
    preset = None
    with pc1:
        if st.button("😰 Stressed", use_container_width=True): preset="stressed"
    with pc2:
        if st.button("😊 Relaxed",  use_container_width=True): preset="relaxed"
    with pc3:
        if st.button("📅 Deadline", use_container_width=True): preset="deadline"
    with pc4:
        if st.button("🌟 Top",      use_container_width=True): preset="top"

    PD = {
        "stressed": dict(week=8,day=0,sess=45,dtd=2,pre=1,prog=60,ls=45,rs=42,st_=3,lag=4,na=4,ir=4.0,mcr=18.0),
        "relaxed":  dict(week=3,day=2,sess=10,dtd=25,pre=0,prog=20,ls=72,rs=70,st_=1,lag=1,na=2,ir=0.8,mcr=2.5),
        "deadline": dict(week=10,day=4,sess=60,dtd=1,pre=1,prog=80,ls=55,rs=52,st_=4,lag=5,na=5,ir=4.8,mcr=22.0),
        "top":      dict(week=5,day=1,sess=20,dtd=14,pre=0,prog=50,ls=88,rs=85,st_=0,lag=0,na=3,ir=1.2,mcr=4.0),
    }
    p = PD.get(preset)

    st.markdown("<hr style='border:none;border-top:2px solid #f1f5f9;margin:10px 0;'>", unsafe_allow_html=True)

    f1,f2 = st.columns(2)
    with f1:
        st.markdown('<div class="ql">📅 Course Week</div><div class="qh">1=just started · 40=nearly done</div>', unsafe_allow_html=True)
        week_of_module = st.slider("", 0, 40, p["week"] if p else 5, key="week", label_visibility="collapsed")
        st.markdown('<div class="ql">🗓 Day of Week</div>', unsafe_allow_html=True)
        day_of_week = st.selectbox("", [0,1,2,3,4,5,6], index=p["day"] if p else 0, format_func=lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], key="day", label_visibility="collapsed")
        st.markdown('<div class="ql">🔢 Sessions So Far</div><div class="qh">Total visits to the platform</div>', unsafe_allow_html=True)
        session_number = st.number_input("", 1, 500, p["sess"] if p else 10, key="sess", label_visibility="collapsed")
        st.markdown('<div class="ql">⏰ Days to Next Deadline</div>', unsafe_allow_html=True)
        days_to_deadline = st.slider("", 0, 60, p["dtd"] if p else 14, key="dtd", label_visibility="collapsed")
        st.markdown('<div class="ql">🚨 Pre-Deadline Period?</div>', unsafe_allow_html=True)
        is_pre_deadline = st.selectbox("", [0,1], index=p["pre"] if p else 0, format_func=lambda x:"✅ Yes" if x else "❌ No", key="pre", label_visibility="collapsed")
        st.markdown('<div class="ql">📈 Course Progress %</div>', unsafe_allow_html=True)
        module_progress = st.slider("", 0, 100, p["prog"] if p else 30, key="prog", label_visibility="collapsed") / 100

    with f2:
        st.markdown('<div class="ql">🏆 Latest Score (out of 100)</div>', unsafe_allow_html=True)
        latest_score = st.slider("", 0, 100, p["ls"] if p else 65, key="ls", label_visibility="collapsed")
        st.markdown('<div class="ql">📊 Average Score</div>', unsafe_allow_html=True)
        rolling_score = float(st.slider("", 0, 100, p["rs"] if p else 62, key="rs", label_visibility="collapsed"))
        st.markdown('<div class="ql">📉 Score Trend</div>', unsafe_allow_html=True)
        OPTS = ["Improving a lot","Improving slightly","Steady","Dropping slightly","Dropping a lot"]
        score_trend_label = st.selectbox("", OPTS, index=p["st_"] if p else 2, key="stl", label_visibility="collapsed")
        score_trend = {"Improving a lot":8.0,"Improving slightly":3.0,"Steady":0.0,"Dropping slightly":-3.0,"Dropping a lot":-8.0}[score_trend_label]
        st.markdown('<div class="ql">📬 Avg Days Late Submitting</div><div class="qh">0=always on time</div>', unsafe_allow_html=True)
        avg_submission_lag = float(st.slider("", 0, 15, p["lag"] if p else 2, key="lag", label_visibility="collapsed"))
        st.markdown('<div class="ql">📝 Assignments Completed</div>', unsafe_allow_html=True)
        n_assessments = st.number_input("", 0, 20, p["na"] if p else 3, key="na", label_visibility="collapsed")
        st.markdown('<div class="ql">⚡ Session Intensity (1–5)</div><div class="qh">1=light browsing · 5=very heavy</div>', unsafe_allow_html=True)
        intensity_ratio = st.slider("", 0.0, 5.0, p["ir"] if p else 1.0, step=0.1, key="ir", label_visibility="collapsed")
        st.markdown('<div class="ql">🖱 Click Rate (per minute)</div><div class="qh">1–5=normal · 10+=rapid</div>', unsafe_allow_html=True)
        mean_click_rate = st.slider("", 0.0, 50.0, p["mcr"] if p else 3.0, step=0.5, key="mcr", label_visibility="collapsed")

    AD = {
        "stressed":{"act_subpage":18,"act_homepage":8,"act_oucontent":12,"act_resource":5},
        "relaxed": {"act_subpage":4, "act_homepage":2,"act_oucontent":5, "act_resource":2},
        "deadline":{"act_subpage":25,"act_homepage":12,"act_oucontent":15,"act_resource":8},
        "top":     {"act_subpage":6, "act_homepage":3,"act_oucontent":8, "act_resource":3},
    }
    av = AD.get(preset,{}) if preset else {}
    act_values = {f: av.get(f,0) for f in features if f.startswith("act_")}

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Fatigue Level", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="about-box"><h4>🧠 About Digital Fatigue</h4><p><strong>Digital fatigue</strong> happens when students become mentally exhausted from prolonged online learning. Signs include rapid unfocused clicking, homepage revisits, and disengagement — often invisible to tutors until grades drop.</p><p>This AI was trained on <strong>10.6 million VLE interactions</strong> from 26,074 real Open University students, detecting fatigue with <strong>93.8% accuracy</strong>.</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        inp = {"week_of_module":week_of_module,"day_of_week":day_of_week,"days_to_deadline":days_to_deadline,"is_pre_deadline":is_pre_deadline,"module_progress":module_progress,"latest_score":latest_score,"rolling_score":rolling_score,"score_trend":score_trend,"avg_submission_lag":avg_submission_lag,"n_assessments":n_assessments,"intensity_ratio":intensity_ratio,"session_number":session_number,"mean_click_rate":mean_click_rate,**act_values}
        df   = pd.DataFrame([inp])[features]
        prob = float(model.predict_proba(scaler.transform(df))[0][1])
        pred = int(prob >= 0.5)

        st.markdown('<div style="background:white;border-radius:16px;padding:20px;box-shadow:0 2px 16px rgba(26,35,126,0.08);"><h4 style="font-family:Nunito,sans-serif;font-weight:800;color:#1a237e;font-size:1rem;margin:0 0 12px;">📊 Prediction Result</h4>', unsafe_allow_html=True)

        if pred==1:
            st.markdown(f'<div class="result-fatigued"><div style="font-size:2.5rem">⚠️</div><div class="result-title">DIGITAL FATIGUE DETECTED</div><div class="result-prob">{prob*100:.0f}%</div><div class="result-desc">This student shows signs of digital fatigue.<br>Early support is recommended.</div></div>', unsafe_allow_html=True)
            if prob>0.85:   sev,sc="🔴 High Risk — Immediate check-in recommended","#fef2f2;border:1px solid #fca5a5"
            elif prob>0.65: sev,sc="🟠 Moderate Risk — Monitor closely","#fff7ed;border:1px solid #fdba74"
            else:           sev,sc="🟡 Low-Moderate — Keep an eye on engagement","#fefce8;border:1px solid #fde047"
            st.markdown(f'<div style="background:{sc};border-radius:10px;padding:10px 14px;margin-top:10px;font-size:0.85rem;font-weight:700;color:#333;">{sev}</div>', unsafe_allow_html=True)
            st.markdown('<div class="advice-box"><h5>💡 Recommended Actions</h5><ul><li>Reach out to check how the student is feeling</li><li>Consider extending upcoming deadlines</li><li>Encourage breaks (Pomodoro technique)</li><li>Suggest offline study alternatives</li></ul></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-ok"><div style="font-size:2.5rem">✅</div><div class="result-title">NO FATIGUE DETECTED</div><div class="result-prob">{(1-prob)*100:.0f}%</div><div class="result-desc">Student appears to be engaging normally.<br>No immediate intervention required.</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="advice-box" style="background:#f0fdf4;border-color:#22c55e;"><h5 style="color:#15803d;">✅ Looking Good!</h5><ul><li>Continue monitoring near deadlines</li><li>Maintain healthy study schedule</li></ul></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.metric("🟢 Not Fatigued", f"{(1-prob)*100:.1f}%")
            st.progress(float(1-prob))
        with c2:
            st.metric("🔴 Fatigued", f"{prob*100:.1f}%")
            st.progress(float(prob))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-waiting"><div style="font-size:2.5rem;margin-bottom:8px;">🔍</div><strong>Awaiting Prediction</strong><div style="margin-top:6px;">Fill in the student details on the left,<br>then click <strong>Predict Fatigue Level</strong></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="mbox"><h4>🤖 Model Information</h4><p style="color:#475569;font-size:0.85rem;margin:0 0 12px;line-height:1.5;">This app uses an <strong>XGBoost</strong> model trained on the OULAD dataset. It analyses 33 behavioural signals to detect digital fatigue with <strong>93.8% accuracy</strong> — no sensors or specialist hardware required.</p><div class="stat-row"><div class="stat-chip"><div class="n">93.8%</div><div class="l">Accuracy</div></div><div class="stat-chip"><div class="n">0.976</div><div class="l">AUC-ROC</div></div><div class="stat-chip"><div class="n">0.872</div><div class="l">F1 Score</div></div><div class="stat-chip"><div class="n">10.6M</div><div class="l">Interactions</div></div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="disc">⚠️ <strong>Disclaimer:</strong> This is a research prototype for dissertation purposes only. It should not be used as the sole basis for real academic decisions. Always combine AI predictions with professional human judgement.</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:20px;color:#94a3b8;font-size:0.78rem;border-top:1px solid #e2e8f0;margin-top:1.5rem;">🎓 <strong>MSc Data Science Dissertation</strong> &nbsp;·&nbsp; XGBoost · OULAD Dataset · AUC: 0.9763 · Accuracy: 93.8%<br><em>For educational and research purposes only.</em></div>', unsafe_allow_html=True)
