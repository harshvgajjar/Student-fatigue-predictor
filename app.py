"""
FatigueSense · Student Digital Fatigue Predictor
Futuristic Streamlit App — MSc Data Science Dissertation
"""

import streamlit as st
import joblib, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FatigueSense · AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark / Light ──────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
dark = st.session_state.dark_mode

if dark:
    BG="0a0a0a"; CARD="rgba(18,18,24,0.95)"; CB="#2a2a38"; TEXT="#e8e8ec"; TEXT2="#7a7a8a"
    A="#38bdf8"; A2="#818cf8"; GLOW="#38bdf8"; SIDEBAR="#070710"
    FB="rgba(220,38,38,0.18)"; FBD="#ef4444"; OB="rgba(16,185,129,0.18)"; OBD="#10b981"
    SC="255,255,255"; N1="rgba(56,189,248,0.06)"; N2="rgba(129,140,248,0.05)"
else:
    BG="f0f4ff"; CARD="rgba(255,255,255,0.95)"; CB="#bfdbfe"; TEXT="#0f172a"; TEXT2="#475569"
    A="#2563eb"; A2="#7c3aed"; GLOW="#3b82f6"; SIDEBAR="#1e1b4b"
    FB="rgba(254,226,226,0.95)"; FBD="#dc2626"; OB="rgba(209,250,229,0.95)"; OBD="#059669"
    SC="80,100,200"; N1="rgba(37,99,235,0.08)"; N2="rgba(124,58,237,0.06)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu{{display:none!important}}
[data-testid="stHeader"]{{background:#{BG}!important;border:none!important;box-shadow:none!important}}
/* Hide only the text label, keep buttons visible */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] p {{display:none!important}}
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] p {{display:none!important}}

/* Collapse button — inside sidebar */
[data-testid="stSidebarCollapseButton"] {{
    background:{A}!important;border:none!important;
    border-radius:8px!important;width:32px!important;height:32px!important;
    margin:8px!important;box-shadow:0 0 12px {A}60!important;
    display:flex!important;align-items:center!important;justify-content:center!important;
}}
[data-testid="stSidebarCollapseButton"] svg {{color:#fff!important;fill:#fff!important;display:block!important;width:16px!important;height:16px!important}}
[data-testid="stSidebarCollapseButton"]:hover {{opacity:.85!important}}

/* Reopen button — visible on main page when sidebar is collapsed */
[data-testid="collapsedControl"] {{
    background:{A}!important;border:none!important;
    border-radius:0 8px 8px 0!important;
    width:28px!important;height:52px!important;
    display:flex!important;align-items:center!important;justify-content:center!important;
    position:fixed!important;left:0!important;top:50%!important;
    transform:translateY(-50%)!important;z-index:9999!important;
    box-shadow:4px 0 16px {A}55!important;cursor:pointer!important;
}}
[data-testid="collapsedControl"] svg {{color:#fff!important;fill:#fff!important;display:block!important;width:14px!important;height:14px!important}}
[data-testid="collapsedControl"]:hover {{width:36px!important;opacity:.9!important}}

.stApp{{background:#{BG}!important;font-family:'Exo 2',sans-serif!important;color:{TEXT}!important;min-height:100vh;overflow-x:hidden}}
.stApp::before{{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 70% 60% at 15% 10%,{N1},transparent),radial-gradient(ellipse 60% 70% at 85% 90%,{N2},transparent),#{BG};z-index:0;pointer-events:none}}
#star-canvas{{position:fixed;inset:0;z-index:0;pointer-events:none}}
.main .block-container{{position:relative;z-index:1;padding:1rem 2rem 3rem!important;max-width:1350px!important}}

[data-testid="stSidebar"]{{background:{SIDEBAR}!important;border-right:1px solid {CB}!important;z-index:10}}
[data-testid="stSidebar"] *{{color:#e2e8f0!important;font-family:'Exo 2',sans-serif!important}}

h1,h2,h3{{font-family:'Orbitron',monospace!important;letter-spacing:.05em}}

/* ── HERO ── */
.hero{{text-align:center;padding:28px 0 4px;position:relative}}
.hero-title{{font-family:'Orbitron',monospace!important;font-size:2.6rem;font-weight:900;background:linear-gradient(135deg,{A},{A2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:.12em;margin:0;animation:glow 3s ease-in-out infinite}}
@keyframes glow{{0%,100%{{filter:drop-shadow(0 0 8px {GLOW}50)}}50%{{filter:drop-shadow(0 0 22px {GLOW}90)}}}}
.hero-badge{{display:inline-block;background:linear-gradient(135deg,{A}22,{A2}22);border:1px solid {A}44;border-radius:99px;padding:5px 18px;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:{A};letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px}}
.hero-sub{{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:{TEXT2};letter-spacing:.18em;text-transform:uppercase;margin-top:8px}}

/* ── CARDS ── */
.card{{background:{CARD};border:1px solid {CB};border-radius:18px;padding:22px 24px;margin-bottom:14px;backdrop-filter:blur(24px);box-shadow:0 4px 40px rgba(0,0,0,0.2),inset 0 1px 0 rgba(255,255,255,0.05);position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,{A},transparent);opacity:.5}}
.card-title{{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:700;color:{A};text-transform:uppercase;letter-spacing:.12em;margin:0 0 4px}}
.card-sub{{font-size:.78rem;color:{TEXT2};font-family:'Exo 2',sans-serif;margin:0 0 16px}}

/* ── SEC HEADER ── */
.sec{{font-family:'Orbitron',monospace;font-size:.72rem;font-weight:600;color:{A};text-transform:uppercase;letter-spacing:.16em;border-bottom:1px solid {CB};padding-bottom:7px;margin:20px 0 14px}}

/* ── KPI ── */
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}}
.kpi{{background:{CARD};border:1px solid {CB};border-radius:12px;padding:14px 16px;backdrop-filter:blur(16px);position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}}
.kpi:hover{{transform:translateY(-3px);box-shadow:0 8px 32px {A}25}}
.kpi::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{A},{A2})}}
.kpi h3{{font-family:'JetBrains Mono',monospace!important;font-size:.6rem!important;color:{TEXT2}!important;text-transform:uppercase;letter-spacing:.14em;margin:0 0 5px!important}}
.kpi p{{font-family:'Orbitron',monospace;font-size:1.35rem;font-weight:700;color:{A}!important;margin:0}}

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{{background:{CARD}!important;border:1px solid {CB}!important;border-radius:12px!important;padding:4px!important;gap:3px!important;backdrop-filter:blur(16px)}}
[data-testid="stTabs"] [data-baseweb="tab"]{{font-family:'Exo 2',sans-serif!important;font-weight:500!important;color:{TEXT2}!important;border-radius:9px!important;padding:8px 20px!important;border:none!important;background:transparent!important;font-size:.83rem!important;letter-spacing:.04em}}
[data-testid="stTabs"] [aria-selected="true"]{{background:linear-gradient(135deg,{A},{A2})!important;color:white!important;font-weight:700!important}}

/* ── VERDICT ── */
.v-fat{{background:{FB};border:1px solid {FBD};border-left:4px solid {FBD};border-radius:14px;padding:22px 26px;margin:14px 0;backdrop-filter:blur(16px);animation:slidein .4s ease}}
.v-ok{{background:{OB};border:1px solid {OBD};border-left:4px solid {OBD};border-radius:14px;padding:22px 26px;margin:14px 0;backdrop-filter:blur(16px);animation:slidein .4s ease}}
@keyframes slidein{{from{{opacity:0;transform:translateX(-14px)}}to{{opacity:1;transform:translateX(0)}}}}
.v-fat h2{{font-family:'Orbitron',monospace;color:{FBD}!important;margin:0 0 6px;font-size:1.4rem}}
.v-ok h2{{font-family:'Orbitron',monospace;color:{OBD}!important;margin:0 0 6px;font-size:1.4rem}}
.v-fat p,.v-ok p{{color:{TEXT}!important;margin:2px 0;font-size:.9rem}}

/* ── BUTTONS ── */
.stButton>button{{background:linear-gradient(135deg,{A},{A2})!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Exo 2',sans-serif!important;font-weight:700!important;font-size:.88rem!important;padding:10px 20px!important;letter-spacing:.06em;transition:all .2s!important;box-shadow:0 4px 16px {A}40!important}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 28px {A}65!important}}

/* ── INPUTS ── */
div[data-testid="stSlider"]>div>div>div>div{{background:{A}!important}}
.fl{{font-size:.75rem;font-weight:600;color:{A};margin:10px 0 2px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.08em}}
.fh{{font-size:.7rem;color:{TEXT2};margin:0 0 3px;font-family:'Exo 2',sans-serif}}

/* ── PILLS ── */
.pill{{display:inline-block;padding:5px 14px;border-radius:99px;font-family:'Exo 2',sans-serif;font-size:.8rem;font-weight:500;margin:3px 2px;backdrop-filter:blur(8px)}}
.pr{{background:rgba(239,68,68,.15);border:1px solid #ef444450;color:#f87171}}
.pg{{background:rgba(16,185,129,.15);border:1px solid #10b98150;color:#34d399}}
.pa{{background:rgba(245,158,11,.15);border:1px solid #f59e0b50;color:#fbbf24}}

/* ── CBAR ── */
.cbar{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.cbar-lbl{{font-size:.72rem;font-weight:600;color:{TEXT2};min-width:115px;font-family:'JetBrains Mono',monospace}}
.cbar-bg{{flex:1;background:{CB};border-radius:99px;height:7px;overflow:hidden}}
.cbar-r{{background:linear-gradient(90deg,#ef4444,#f87171);height:100%;border-radius:99px}}
.cbar-g{{background:linear-gradient(90deg,#10b981,#34d399);height:100%;border-radius:99px}}
.cbar-pct{{font-size:.78rem;font-weight:700;min-width:44px;text-align:right;font-family:'Orbitron',monospace}}

/* ── AWAIT ── */
.await{{background:{CARD};border:2px dashed {CB};border-radius:16px;padding:40px 20px;text-align:center;backdrop-filter:blur(16px)}}
.aw-icon{{font-size:3rem;margin-bottom:12px;animation:float 3s ease-in-out infinite}}
@keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
.aw-t{{font-family:'Orbitron',monospace;font-size:.95rem;color:{TEXT};margin-bottom:6px}}
.aw-s{{font-size:.8rem;color:{TEXT2};font-family:'Exo 2',sans-serif;line-height:1.6}}

/* ── DISCLAIMER ── */
.disc{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-left:3px solid #f59e0b;border-radius:8px;padding:10px 14px;font-size:.75rem;color:{TEXT2};font-family:'JetBrains Mono',monospace;line-height:1.6;margin-top:12px}}

/* ── ABOUT ── */
.ab{{font-size:.86rem;color:{TEXT2};line-height:1.75;margin:0 0 8px;font-family:'Exo 2',sans-serif}}
.ab strong{{color:{A}}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:5px}}
::-webkit-scrollbar-track{{background:#{BG}}}
::-webkit-scrollbar-thumb{{background:{CB};border-radius:3px}}

/* ── SIDEBAR SPEC ── */
.spec{{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#64748b;line-height:2;padding:0 4px}}
.spec span{{color:{A}!important}}
</style>

<canvas id="star-canvas"></canvas>
<script>
(function(){{
  const c=document.getElementById('star-canvas');if(!c)return;
  const x=c.getContext('2d');let W,H,stars=[],shoots=[];
  const SC='{SC}';
  function resize(){{W=c.width=window.innerWidth;H=c.height=window.innerHeight}}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<200;i++)stars.push({{x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.5+.2,a:Math.random(),da:(Math.random()-.5)*.007,s:Math.random()*.05+.01}});
  setInterval(()=>shoots.push({{x:Math.random()*W*.7,y:Math.random()*H*.4,l:Math.random()*130+60,sp:Math.random()*9+5,a:1}}),4500);
  function draw(){{
    x.clearRect(0,0,W,H);
    stars.forEach(s=>{{s.a+=s.da;if(s.a<=0||s.a>=1)s.da*=-1;s.y+=s.s;if(s.y>H)s.y=0;x.beginPath();x.arc(s.x,s.y,s.r,0,Math.PI*2);x.fillStyle=`rgba(${{SC}},${{s.a.toFixed(2)}})`;x.fill()}});
    shoots.forEach((s,i)=>{{x.beginPath();const g=x.createLinearGradient(s.x,s.y,s.x+s.l,s.y+s.l*.4);g.addColorStop(0,`rgba(${{SC}},0)`);g.addColorStop(1,`rgba(${{SC}},${{s.a.toFixed(2)}})`);x.strokeStyle=g;x.lineWidth=1.5;x.moveTo(s.x,s.y);x.lineTo(s.x+s.l,s.y+s.l*.4);x.stroke();s.x+=s.sp;s.y+=s.sp*.4;s.a-=.016;if(s.a<=0)shoots.splice(i,1)}});
    requestAnimationFrame(draw);
  }}
  draw();
}})();
</script>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = joblib.load("xgb_clean.pkl")
    s = joblib.load("scaler_safe.pkl")
    f = json.load(open("features_safe.json"))
    r = json.load(open("results.json"))
    return m, s, f, r
model, scaler, features, results = load_models()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='font-family:Orbitron,monospace;font-size:1.1rem;font-weight:900;background:linear-gradient(135deg,{A},{A2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.1em;margin:0'>◈ FATIGUESENSE</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.6rem;color:#475569;letter-spacing:.15em;margin-top:2px'>STUDENT FATIGUE DETECTION AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    mode_label = "☀️ LIGHT MODE" if dark else "🌙 DARK MODE"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:#94a3b8;letter-spacing:.12em'>◈ MODEL SPECS</p>", unsafe_allow_html=True)
    xgb = results["xgb"]
    st.markdown(f"""
    <div class="spec">
    ◈ Algorithm: <span>XGBoost</span><br>
    ◈ Dataset: <span>OULAD</span><br>
    ◈ Sessions: <span>1,808,119</span><br>
    ◈ Students: <span>26,074</span><br>
    ◈ Features: <span>33</span><br>
    ◈ Accuracy: <span>{xgb['test_acc']*100:.1f}%</span><br>
    ◈ AUC-ROC: <span>{xgb['test_auc']:.4f}</span><br>
    ◈ F1 Score: <span>{xgb['test_f1']:.4f}</span><br>
    ◈ Precision: <span>{xgb['test_prec']:.4f}</span><br>
    ◈ Recall: <span>{xgb['test_rec']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:#94a3b8;letter-spacing:.12em'>◈ ALL MODELS</p>", unsafe_allow_html=True)
    for k, lbl in [("xgb","XGBoost ⭐"),("rf","Random Forest"),("lr","Logistic Reg.")]:
        r = results[k]
        st.markdown(f"<p style='font-family:Exo 2,sans-serif;font-size:.75rem;color:{A};font-weight:700;margin:6px 0 1px'>{lbl}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:#64748b;margin:0'>AUC {r['test_auc']:.3f} · F1 {r['test_f1']:.3f}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.58rem;color:#374151;text-align:center;letter-spacing:.1em'>DS7010 DISSERTATION · 2025-2026</p>", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">🎓 MSc Data Science Dissertation · University Project</div>
    <h1 class="hero-title">◈ FATIGUESENSE</h1>
    <p class="hero-sub">◈ Student Digital Fatigue Prediction · XGBoost AI · OULAD Dataset ◈</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi"><h3>Accuracy</h3><p>93.8%</p></div>
    <div class="kpi"><h3>AUC-ROC</h3><p>0.976</p></div>
    <div class="kpi"><h3>Sessions</h3><p>1.8M</p></div>
    <div class="kpi"><h3>Students</h3><p>26,074</p></div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧠  Predict Fatigue", "📊  Model Results", "📖  About"])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.05, 1], gap="medium")

    with left:
        st.markdown('<div class="card"><div class="card-title">◈ Enter Student Session Details</div><div class="card-sub">No technical knowledge needed — answer the questions below</div>', unsafe_allow_html=True)

        st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:{TEXT2};margin:0 0 7px;letter-spacing:.08em'>◈ QUICK EXAMPLES:</p>", unsafe_allow_html=True)
        pb1,pb2,pb3,pb4 = st.columns(4)
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
            "stressed": dict(week=8,  day=0, sess=45, dtd=2,  pre=1, prog=60, ls=45, rs=42, stx=3, lag=4, na=4, ir=4.0,  mcr=18.0),
            "relaxed":  dict(week=3,  day=2, sess=10, dtd=25, pre=0, prog=20, ls=72, rs=70, stx=1, lag=1, na=2, ir=0.8,  mcr=2.5),
            "deadline": dict(week=10, day=4, sess=60, dtd=1,  pre=1, prog=80, ls=55, rs=52, stx=4, lag=5, na=5, ir=4.8,  mcr=22.0),
            "top":      dict(week=5,  day=1, sess=20, dtd=14, pre=0, prog=50, ls=88, rs=85, stx=0, lag=0, na=3, ir=1.2,  mcr=4.0),
        }
        p = PD.get(preset)

        st.markdown(f"<hr style='border:none;border-top:1px solid {CB};margin:12px 0'>", unsafe_allow_html=True)

        st.markdown('<div class="sec">◈ Session & Timeline</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="fl">Course Week</div><div class="fh">1 = start · 40 = end</div>', unsafe_allow_html=True)
            week_of_module = st.slider("_wk",0,40,p["week"] if p else 5,label_visibility="collapsed")
            st.markdown('<div class="fl">Day of Week</div><div class="fh">Select the day the student studied</div>', unsafe_allow_html=True)
            day_of_week = st.selectbox("_dw",[0,1,2,3,4,5,6],index=p["day"] if p else 0,format_func=lambda x:["📅 Monday","📅 Tuesday","📅 Wednesday","📅 Thursday","📅 Friday","📅 Saturday","📅 Sunday"][x],label_visibility="collapsed")
            st.markdown('<div class="fl">Sessions So Far</div><div class="fh">Each platform visit = 1 session</div>', unsafe_allow_html=True)
            session_number = st.number_input("_sn",1,500,p["sess"] if p else 10,label_visibility="collapsed")
        with b:
            st.markdown('<div class="fl">Days to Deadline</div><div class="fh">0 = today · 30+ = plenty</div>', unsafe_allow_html=True)
            days_to_deadline = st.slider("_dd",0,60,p["dtd"] if p else 14,label_visibility="collapsed")
            st.markdown('<div class="fl">Pre-Deadline Rush?</div>', unsafe_allow_html=True)
            is_pre_deadline = st.selectbox("_pd",[0,1],index=p["pre"] if p else 0,format_func=lambda x:"✅ Yes — due soon" if x else "❌ No — plenty of time",label_visibility="collapsed")
            st.markdown('<div class="fl">Course Progress %</div><div class="fh">0% = start · 100% = complete</div>', unsafe_allow_html=True)
            module_progress = st.slider("_mp",0,100,p["prog"] if p else 30,label_visibility="collapsed")/100

        st.markdown('<div class="sec">◈ Academic Performance</div>', unsafe_allow_html=True)
        c, d = st.columns(2)
        with c:
            st.markdown('<div class="fl">Latest Score (out of 100)</div>', unsafe_allow_html=True)
            latest_score = st.slider("_ls",0,100,p["ls"] if p else 65,label_visibility="collapsed")
            st.markdown('<div class="fl">Average Score</div>', unsafe_allow_html=True)
            rolling_score = float(st.slider("_rs",0,100,p["rs"] if p else 62,label_visibility="collapsed"))
            st.markdown('<div class="fl">Score Trend</div><div class="fh">Are recent assignment scores going up or down?</div>', unsafe_allow_html=True)
            OPTS=["📈 Improving a lot (+8)","📈 Improving slightly (+3)","➡️ Staying steady (0)","📉 Dropping slightly (-3)","📉 Dropping a lot (-8)"]
            stlbl=st.selectbox("_stl",OPTS,index=p["stx"] if p else 2,label_visibility="collapsed")
            score_trend={"📈 Improving a lot (+8)":8.0,"📈 Improving slightly (+3)":3.0,"➡️ Staying steady (0)":0.0,"📉 Dropping slightly (-3)":-3.0,"📉 Dropping a lot (-8)":-8.0}[stlbl]
        with d:
            st.markdown('<div class="fl">Avg Days Late Submitting</div><div class="fh">0 = always on time</div>', unsafe_allow_html=True)
            avg_submission_lag = float(st.slider("_al",0,15,p["lag"] if p else 2,label_visibility="collapsed"))
            st.markdown('<div class="fl">Assignments Done</div>', unsafe_allow_html=True)
            n_assessments = st.number_input("_na",0,20,p["na"] if p else 3,label_visibility="collapsed")

        st.markdown('<div class="sec">◈ Online Activity</div>', unsafe_allow_html=True)
        e, f_ = st.columns(2)
        with e:
            st.markdown('<div class="fl">Session Intensity (1–5)</div><div class="fh">1 = light · 5 = very heavy</div>', unsafe_allow_html=True)
            intensity_ratio = st.slider("_ir",0.0,5.0,p["ir"] if p else 1.0,step=0.1,label_visibility="collapsed")
        with f_:
            st.markdown('<div class="fl">Clicks Per Minute</div><div class="fh">1–5 = normal · 10+ = rapid</div>', unsafe_allow_html=True)
            mean_click_rate = st.slider("_mcr",0.0,50.0,p["mcr"] if p else 3.0,step=0.5,label_visibility="collapsed")

        AD={"stressed":{"act_subpage":18,"act_homepage":8,"act_oucontent":12,"act_resource":5},"relaxed":{"act_subpage":4,"act_homepage":2,"act_oucontent":5,"act_resource":2},"deadline":{"act_subpage":25,"act_homepage":12,"act_oucontent":15,"act_resource":8},"top":{"act_subpage":6,"act_homepage":3,"act_oucontent":8,"act_resource":3}}
        av=AD.get(preset,{}) if preset else {}
        act_values={feat:av.get(feat,0) for feat in features if feat.startswith("act_")}

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  ANALYSE FATIGUE LEVEL", type="primary", use_container_width=True)

    with right:
        # About section
        st.markdown(f"""<div class="card">
        <div class="card-title">◈ About Digital Fatigue</div>
        <p class="ab"><strong>Digital fatigue</strong> is the mental exhaustion that builds up when students spend prolonged periods on online learning platforms. Signs include rapid unfocused clicking, repeated homepage visits, and jumping between content pages — often invisible to tutors until grades drop.</p>
        <p class="ab">This AI was trained on <strong>10.6 million real VLE interactions</strong> from 26,074 Open University students, detecting fatigue patterns with <strong>93.8% accuracy</strong> — no sensors or surveys required.</p>
        </div>""", unsafe_allow_html=True)

        # How it works
        st.markdown(f"""<div class="card">
        <div class="card-title">◈ How It Works</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
            <div style="background:{CB}22;border:1px solid {CB};border-radius:10px;padding:12px 8px;text-align:center">
                <div style="font-family:Orbitron,monospace;font-size:.75rem;font-weight:700;color:{A};width:26px;height:26px;background:{A}22;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 8px">1</div>
                <div style="font-size:.72rem;font-weight:700;color:{TEXT};font-family:Exo 2,sans-serif;margin-bottom:3px">Enter Details</div>
                <div style="font-size:.68rem;color:{TEXT2};font-family:Exo 2,sans-serif;line-height:1.4">Fill in student session info</div>
            </div>
            <div style="background:{CB}22;border:1px solid {CB};border-radius:10px;padding:12px 8px;text-align:center">
                <div style="font-family:Orbitron,monospace;font-size:.75rem;font-weight:700;color:{A};width:26px;height:26px;background:{A}22;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 8px">2</div>
                <div style="font-size:.72rem;font-weight:700;color:{TEXT};font-family:Exo 2,sans-serif;margin-bottom:3px">33 Signals</div>
                <div style="font-size:.68rem;color:{TEXT2};font-family:Exo 2,sans-serif;line-height:1.4">AI reads click patterns & scores</div>
            </div>
            <div style="background:{CB}22;border:1px solid {CB};border-radius:10px;padding:12px 8px;text-align:center">
                <div style="font-family:Orbitron,monospace;font-size:.75rem;font-weight:700;color:{A};width:26px;height:26px;background:{A}22;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 8px">3</div>
                <div style="font-size:.72rem;font-weight:700;color:{TEXT};font-family:Exo 2,sans-serif;margin-bottom:3px">XGBoost</div>
                <div style="font-size:.68rem;color:{TEXT2};font-family:Exo 2,sans-serif;line-height:1.4">Model trained on 1.8M sessions</div>
            </div>
            <div style="background:{CB}22;border:1px solid {CB};border-radius:10px;padding:12px 8px;text-align:center">
                <div style="font-family:Orbitron,monospace;font-size:.75rem;font-weight:700;color:{A};width:26px;height:26px;background:{A}22;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 8px">4</div>
                <div style="font-size:.72rem;font-weight:700;color:{TEXT};font-family:Exo 2,sans-serif;margin-bottom:3px">Result</div>
                <div style="font-size:.68rem;color:{TEXT2};font-family:Exo 2,sans-serif;line-height:1.4">Prediction + advice</div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

        # Prediction result
        st.markdown(f'<div class="card"><div class="card-title">◈ Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            inp = {"week_of_module":week_of_module,"day_of_week":day_of_week,"days_to_deadline":days_to_deadline,"is_pre_deadline":is_pre_deadline,"module_progress":module_progress,"latest_score":latest_score,"rolling_score":rolling_score,"score_trend":score_trend,"avg_submission_lag":avg_submission_lag,"n_assessments":n_assessments,"intensity_ratio":intensity_ratio,"session_number":session_number,"mean_click_rate":mean_click_rate,**act_values}
            df  = pd.DataFrame([inp])[features]
            prob= float(model.predict_proba(scaler.transform(df))[0][1])
            pred= int(prob>=0.5)

            if pred==1:
                st.markdown(f"""<div class="v-fat">
                    <h2>⚠ DIGITAL FATIGUE DETECTED</h2>
                    <p>This student is showing signs of digital fatigue with <strong style="color:{FBD}">{prob*100:.1f}% confidence</strong>.</p>
                    <p>Early intervention is recommended.</p>
                </div>""", unsafe_allow_html=True)
                if prob>0.85:   sev,sc="🔴 High Risk — Immediate check-in recommended","rgba(239,68,68,.12);border:1px solid #ef444440"
                elif prob>0.65: sev,sc="🟠 Moderate Risk — Monitor closely","rgba(245,158,11,.12);border:1px solid #f59e0b40"
                else:           sev,sc="🟡 Low-Moderate — Keep an eye on engagement","rgba(250,204,21,.12);border:1px solid #facc1540"
                st.markdown(f'<div style="background:{sc};border-radius:9px;padding:9px 14px;margin-top:8px;font-size:.82rem;font-weight:600;color:{TEXT};font-family:Exo 2,sans-serif">{sev}</div>', unsafe_allow_html=True)
                pills = '<div style="margin-top:12px"><p style="font-family:JetBrains Mono,monospace;font-size:.68rem;color:{};letter-spacing:.1em;margin:0 0 6px">◈ RECOMMENDED ACTIONS</p>'.format(TEXT2)
                for action in ["Reach out to check how the student is feeling","Consider extending upcoming deadlines","Encourage breaks — Pomodoro technique","Suggest offline study to reduce screen time"]:
                    pills += f'<div class="pill pr">⚠ {action}</div><br>'
                pills += '</div>'
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="v-ok">
                    <h2>✓ NO FATIGUE DETECTED</h2>
                    <p>Student is engaging normally with <strong style="color:{OBD}">{(1-prob)*100:.1f}% confidence</strong>.</p>
                    <p>No immediate intervention required.</p>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<div style="margin-top:10px"><div class="pill pg">✓ Continue monitoring near deadlines</div><br><div class="pill pg">✓ Encourage healthy study schedule</div></div>', unsafe_allow_html=True)

            # Confidence bars
            st.markdown(f'<div style="margin-top:16px"><p style="font-family:JetBrains Mono,monospace;font-size:.68rem;color:{TEXT2};letter-spacing:.12em;margin:0 0 10px">◈ CONFIDENCE BREAKDOWN</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="cbar"><span class="cbar-lbl">🟢 Not Fatigued</span><div class="cbar-bg"><div class="cbar-g" style="width:{(1-prob)*100:.0f}%"></div></div><span class="cbar-pct" style="color:#34d399">{(1-prob)*100:.1f}%</span></div>
            <div class="cbar"><span class="cbar-lbl">🔴 Fatigued</span><div class="cbar-bg"><div class="cbar-r" style="width:{prob*100:.0f}%"></div></div><span class="cbar-pct" style="color:#f87171">{prob*100:.1f}%</span></div>
            </div>""", unsafe_allow_html=True)

            # Top features
            st.markdown(f'<p style="font-family:JetBrains Mono,monospace;font-size:.68rem;color:{TEXT2};letter-spacing:.12em;margin:16px 0 8px">◈ KEY FACTORS</p>', unsafe_allow_html=True)
            FLABELS={"act_subpage":"Subpage clicks","act_homepage":"Homepage revisits","act_oucontent":"Content views","act_resource":"Resource downloads","act_quiz":"Quiz attempts","act_forumng":"Forum activity","intensity_ratio":"Session intensity","mean_click_rate":"Click speed","days_to_deadline":"Days to deadline","week_of_module":"Course week","session_number":"Session count","module_progress":"Progress","latest_score":"Latest score","rolling_score":"Avg score","score_trend":"Score trend","avg_submission_lag":"Sub. lateness"}
            fi=pd.DataFrame({"Signal":[FLABELS.get(feat,feat.replace("act_","").title()) for feat in features],"Importance":[f"{v*100:.1f}%" for v in model.feature_importances_],"Value":[round(inp[feat],1) for feat in features]}).sort_values("Importance",ascending=False).head(6).reset_index(drop=True)
            st.dataframe(fi,use_container_width=True,hide_index=True)

        else:
            st.markdown("""<div class="await">
                <div class="aw-icon">🧠</div>
                <div class="aw-t">Awaiting Analysis</div>
                <div class="aw-s">Fill in the student details on the left,<br>then click <strong>Analyse Fatigue Level</strong>.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="disc">
        ⚠ DISCLAIMER: This is a research prototype for MSc dissertation purposes only.
        Predictions should support, not replace, professional human judgement.
        Not for real clinical or academic decision-making.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">◈ Model Comparison — All Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><h3>Best Accuracy</h3><p>93.8%</p></div>
        <div class="kpi"><h3>Best AUC</h3><p>0.976</p></div>
        <div class="kpi"><h3>Best F1</h3><p>0.872</p></div>
        <div class="kpi"><h3>FP Reduction</h3><p>63%</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">◈ Full Results Table — All 4 Models</div>', unsafe_allow_html=True)
    data = {
        "Model":    ["Logistic Regression","Random Forest","⭐ XGBoost","LSTM (KT4)"],
        "Accuracy": ["90.1%","91.6%","93.8%","65.4%"],
        "AUC-ROC":  [0.949,0.964,0.976,0.681],
        "F1 Score": [0.814,0.836,0.872,0.445],
        "Precision":[0.768,0.819,0.898,0.350],
        "Recall":   [0.866,0.853,0.848,0.600],
        "Dataset":  ["OULAD","OULAD","OULAD","EdNet KT4"],
    }
    df_res = pd.DataFrame(data)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="card">
    <div class="card-title">◈ Key Findings</div>
    <div style="margin-top:10px">
        <div class="pill pg">✓ XGBoost best across all 5 metrics on OULAD</div><br>
        <div class="pill pg">✓ XGBoost = 63% fewer false positives vs Logistic Regression (6,533 vs 17,750)</div><br>
        <div class="pill pa">◈ LSTM limited by 500-user training sample — full KT4 (297K users) expected to improve</div><br>
        <div class="pill pr">⚠ All OULAD models benefit from 33 engineered features vs LSTM's 5 raw features</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">◈ Fairness Evaluation — XGBoost</div>', unsafe_allow_html=True)
    fair_data = {
        "Demographic Group": ["Gender (M vs F)","Age Band (0-35 vs 55+)","Disability (Y vs N)","IMD Band (all)"],
        "AUC Gap":  [0.002,0.009,0.001,0.004],
        "Recall Gap":[0.022,0.060,0.003,0.012],
        "Status":   ["✅ Fair","⚠️ Flagged","✅ Fair","✅ Fair"],
    }
    st.dataframe(pd.DataFrame(fair_data), use_container_width=True, hide_index=True)
    st.markdown(f'<p style="font-family:Exo 2,sans-serif;font-size:.8rem;color:{TEXT2};margin-top:8px">Note: Age-band recall gap of 0.060 is marginally above 0.05 threshold — attributable to small 55+ sample size (3,086 vs 175,962) rather than genuine bias.</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="card">
        <div class="card-title">◈ About This Study</div>
        <p class="ab"><strong>Research Question:</strong> Can digital fatigue in remote learners be reliably predicted from behavioural, temporal, and interaction-level log data using machine learning?</p>
        <p class="ab"><strong>Answer:</strong> Yes — XGBoost achieves 93.8% accuracy and AUC 0.9763, with near-perfect discrimination between fatigued and non-fatigued sessions.</p>
        <p class="ab"><strong>Novel Contribution:</strong> Proxy label methodology using 6 behavioural signals, multi-model comparison with SHAP explainability, fairness evaluation across 4 demographic dimensions, and live deployment.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card">
        <div class="card-title">◈ The 6 Proxy Signals</div>
        <div style="margin-top:10px">
            <div class="pill pr">⚡ Signal 1: total_clicks > 75th percentile (> 26)</div><br>
            <div class="pill pr">🔀 Signal 2: task_switch_ratio > 0.90</div><br>
            <div class="pill pr">🌐 Signal 3: unique_sites > 75th percentile (> 6)</div><br>
            <div class="pill pr">🔢 Signal 4: n_interactions > 75th percentile (> 8)</div><br>
            <div class="pill pa">⏳ Signal 5: login_gap > 3 days</div><br>
            <div class="pill pa">📈 Signal 6: load_trend > 0 (increasing load)</div><br>
            <div class="pill pg">◈ Score ≥ 0.5 (3+ signals fired) → FATIGUED</div>
        </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="card">
        <div class="card-title">◈ Datasets</div>
        <p class="ab"><strong>OULAD</strong> (Open University Learning Analytics Dataset) — 10.6M VLE interactions, 32,593 students, 7 modules. Used for LR, RF, XGBoost training.</p>
        <p class="ab"><strong>EdNet KT4</strong> — 131M interactions, 784K students. 500 users sampled for LSTM due to RAM constraints. Provides sequential timestamped logs.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card">
        <div class="card-title">◈ SHAP Top Features</div>
        <p class="ab" style="margin-bottom:12px">What drives fatigue predictions in XGBoost:</p>
        <div class="pill pr">🥇 act_oucontent — SHAP 1.41</div><br>
        <div class="pill pr">🥈 act_subpage — SHAP 1.19</div><br>
        <div class="pill pr">🥉 act_homepage — SHAP 0.98</div><br>
        <div class="pill pa">◈ intensity_ratio — SHAP 0.92</div><br>
        <div class="pill pa">◈ act_forumng — SHAP 0.78</div><br>
        <p class="ab" style="margin-top:10px">Key insight: <strong>WHAT students engage with</strong> matters more than WHEN or HOW MUCH.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card">
        <div class="card-title">◈ Limitations & Future Work</div>
        <div class="pill pr">⚠ Fatigue label is a proxy — no ground-truth available</div><br>
        <div class="pill pr">⚠ LSTM trained on 500 users only (RAM constraint)</div><br>
        <div class="pill pr">⚠ Single institution — generalisability unconfirmed</div><br>
        <div class="pill pg" style="margin-top:6px">✓ Future: Karolinska Sleepiness Scale direct measurement</div><br>
        <div class="pill pg">✓ Future: Full KT4 (297K users) LSTM training</div><br>
        <div class="pill pg">✓ Future: Transformer-based sequential models</div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:28px 0 12px;font-family:'JetBrains Mono',monospace;font-size:.62rem;color:{TEXT2};letter-spacing:.16em">
◈ FATIGUESENSE · DS7010 DISSERTATION · MSc DATA SCIENCE ·
XGBOOST · OULAD DATASET · AUC 0.9763 · 93.8% ACCURACY ◈
</div>
""", unsafe_allow_html=True)
