import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS (inline for Streamlit Cloud compatibility)
# For local dev: reads from styles.css
# For cloud: CSS is embedded below
try:
    with open("styles.css") as f:
        css = f.read()
except FileNotFoundError:
    # Fallback CSS for Streamlit Cloud deployment
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Sora', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 50%, #fff5f0 100%); background-attachment: fixed; }
    #MainMenu, header, footer, .stDeployButton { visibility: hidden; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; max-width: 1100px !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.1rem !important; }
    section[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
    .gradient-text { background: linear-gradient(135deg, #4f46e5 0%, #ec4899 50%, #f97316 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .home-hero { text-align: center; padding: 2.5rem 1rem 2rem; position: relative; }
    .home-hero-img { font-size: 5rem; margin-bottom: 1rem; display: inline-block; animation: float 4s ease-in-out infinite; }
    @keyframes float { 0%, 100% { transform: translateY(0px) rotate(0deg); } 25% { transform: translateY(-12px) rotate(2deg); } 75% { transform: translateY(-8px) rotate(-2deg); } }
    .home-badge { display: inline-flex; align-items: center; gap: 6px; background: linear-gradient(135deg,#ede9fe,#ddd6fe); color: #5b21b6; border-radius: 999px; padding: 6px 18px; font-size: 0.75rem; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 1rem; }
    .home-title { font-family: 'Sora', sans-serif; font-size: 2.6rem; font-weight: 800; color: #1e1b4b; margin: 0 0 0.5rem; letter-spacing: -1.5px; line-height: 1.15; }
    .home-sub { color: #6b7280; font-size: 1rem; font-weight: 400; margin: 0 auto 2.5rem; max-width: 500px; line-height: 1.6; }
    .side-decoration { position: fixed; top: 50%; transform: translateY(-50%); font-size: 8rem; opacity: 0.08; pointer-events: none; z-index: 0; animation: sideFloat 6s ease-in-out infinite; }
    .side-decoration.left { left: 2%; } .side-decoration.right { right: 2%; }
    @keyframes sideFloat { 0%, 100% { transform: translateY(-50%) translateX(0); } 50% { transform: translateY(-50%) translateX(10px); } }
    .particle { position: fixed; border-radius: 50%; pointer-events: none; opacity: 0.15; animation: particleFloat 15s ease-in-out infinite; }
    @keyframes particleFloat { 0%, 100% { transform: translateY(0) translateX(0); } 25% { transform: translateY(-30px) translateX(20px); } 50% { transform: translateY(-60px) translateX(-10px); } 75% { transform: translateY(-30px) translateX(15px); } }
    .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 2rem 0; }
    .feature-card { background: rgba(255, 255, 255, 0.95); border-radius: 18px; padding: 1.5rem 1.2rem; text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); transition: transform 0.3s ease, box-shadow 0.3s ease; }
    .feature-card:hover { transform: translateY(-5px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .feature-title { font-family: 'Sora', sans-serif; font-size: 0.9rem; font-weight: 700; color: #1e1b4b; margin-bottom: 0.4rem; }
    .feature-desc { font-size: 0.8rem; color: #6b7280; line-height: 1.5; }
    .input-header { text-align: center; padding: 1rem 0 0.8rem; }
    .input-header h2 { font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 800; color: #1e1b4b; margin: 0 0 0.3rem; }
    .input-header p { color: #6b7280; font-size: 0.85rem; margin: 0; }
    .form-section { display: flex; align-items: center; gap: 10px; padding: 1rem 0 0.8rem; border-bottom: 2px solid #ede9fe; margin-bottom: 1.2rem; }
    .form-section-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .form-section-label { font-family: 'Sora', sans-serif; font-size: 0.85rem; font-weight: 700; color: #2d1b69; text-transform: uppercase; letter-spacing: 1.5px; }
    .form-section-emoji { font-size: 1.2rem; margin-right: 4px; }
    .stSelectbox label, .stNumberInput label, .stTextInput label { color: #5b4a7a !important; font-size: 0.72rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px !important; }
    .stSelectbox > div > div { background: #ffffff !important; border: 2px solid #e2e8f0 !important; border-radius: 14px !important; color: #1f1535 !important; min-height: 48px !important; font-weight: 500 !important; font-size: 0.95rem !important; box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important; transition: all 0.2s ease !important; }
    .stSelectbox > div > div:hover { border-color: #c4b5fd !important; }
    .stSelectbox > div > div:focus-within { border-color: #7c3aed !important; box-shadow: 0 0 0 4px rgba(124,58,237,0.1) !important; }
    .stNumberInput > div > div > input, .stTextInput > div > div > input { background: #ffffff !important; border: 2px solid #e2e8f0 !important; border-radius: 14px !important; color: #1f1535 !important; font-size: 0.95rem !important; font-weight: 500 !important; height: 48px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important; transition: all 0.2s ease !important; }
    .stNumberInput > div > div > input:hover, .stTextInput > div > div > input:hover { border-color: #c4b5fd !important; }
    .stNumberInput > div > div > input:focus, .stTextInput > div > div > input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 4px rgba(124,58,237,0.1) !important; background: white !important; outline: none !important; }
    .stNumberInput button { background: #ede9fe !important; border: none !important; color: #7c3aed !important; border-radius: 8px !important; height: 22px !important; }
    .stFormSubmitButton > button { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%) !important; color: white !important; border: none !important; border-radius: 16px !important; padding: 1rem 2rem !important; font-size: 1.05rem !important; font-weight: 700 !important; font-family: 'Sora', sans-serif !important; width: 100% !important; box-shadow: 0 8px 32px rgba(79, 70, 229, 0.3) !important; transition: all 0.3s ease !important; margin-top: 1rem !important; }
    .stFormSubmitButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 40px rgba(79, 70, 229, 0.4) !important; }
    .stButton > button[kind="secondary"] { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 1.4rem !important; font-size: 0.9rem !important; font-weight: 600 !important; transition: all 0.2s !important; box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25) !important; }
    .stButton > button[kind="secondary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important; }
    .result-heading { text-align: center; padding: 1rem 0 0.5rem; font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 800; color: #1e1b4b; }
    .result-hero { border-radius: 24px; padding: 2.5rem 2rem 2rem; text-align: center; margin-bottom: 0.8rem; animation: popIn 0.6s ease; }
    .result-hero.placed { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%); border: 2px solid #34d399; box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15); }
    .result-hero.not-placed { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 50%, #fecaca 100%); border: 2px solid #f87171; box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15); }
    .rh-emoji { font-size: 3rem; margin-bottom: 0.6rem; line-height: 1; }
    .rh-verdict { font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
    .rh-verdict.placed { color: #059669; } .rh-verdict.not-placed { color: #dc2626; }
    .rh-pct { font-family: 'Sora', sans-serif; font-size: 3.5rem; font-weight: 800; color: #1e1b4b; line-height: 1.1; margin: 0.3rem 0; background: linear-gradient(135deg, #1e1b4b 0%, #4f46e5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .rh-sub { font-size: 0.85rem; color: #6b7280; font-weight: 500; margin-top: 0.4rem; }
    @keyframes popIn { from { opacity: 0; transform: scale(0.9) translateY(20px); } to { opacity: 1; transform: scale(1) translateY(0); } }
    @keyframes fireworks { 0% { transform: translateY(0) scale(1); opacity: 1; } 100% { transform: translateY(-100px) scale(0); opacity: 0; } }
    .firework { position: absolute; width: 6px; height: 6px; border-radius: 50%; animation: fireworks 1s ease-out forwards; }
    @keyframes rainDrop { 0% { transform: translateY(-20px); opacity: 1; } 100% { transform: translateY(100px); opacity: 0; } }
    .rain-drop { position: absolute; width: 2px; height: 10px; background: #93c5fd; border-radius: 2px; animation: rainDrop 1.5s ease-in infinite; }
    .celebration-container { position: relative; overflow: visible; }
    .profile-score-wrap { background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 1.5rem 1.8rem; border: 1.5px solid #ede9fe; box-shadow: 0 2px 16px rgba(109,40,217,0.04); margin-bottom: 0.8rem; }
    .ps-title { font-family: 'Sora', sans-serif; font-size: 0.85rem; font-weight: 700; color: #2d1b69; text-transform: uppercase; letter-spacing: 1.3px; margin-bottom: 1.2rem; }
    .ps-row { display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem; }
    .ps-label { font-size: 0.85rem; font-weight: 600; color: #4b5563; width: 120px; flex-shrink: 0; }
    .ps-bar-bg { flex: 1; background: #f0ecff; border-radius: 999px; height: 10px; overflow: hidden; }
    .ps-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
    .ps-pct { font-size: 0.8rem; font-weight: 700; width: 42px; text-align: right; flex-shrink: 0; }
    .rec-section-title { font-family: 'Sora', sans-serif; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.3px; padding: 0.6rem 0 0.6rem; margin-top: 0.4rem; color: #2d1b69; }
    .rec-tier-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 0.8rem 0 0.4rem; display: flex; align-items: center; gap: 6px; }
    .rec-card { display: flex; align-items: flex-start; gap: 12px; background: rgba(255, 255, 255, 0.95); border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.4rem; border: 1.5px solid #f0ecff; box-shadow: 0 1px 6px rgba(109,40,217,0.04); transition: transform 0.2s ease; }
    .rec-card:hover { transform: translateX(4px); }
    .rec-card.urgent { border-left: 4px solid #f43f5e; border-color: #ffe4e6; border-left-color: #f43f5e; }
    .rec-card.improve { border-left: 4px solid #f97316; border-color: #fff7ed; border-left-color: #f97316; }
    .rec-card.good { border-left: 4px solid #10b981; border-color: #f0fdf4; border-left-color: #10b981; }
    .rec-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
    .rec-title { font-weight: 700; font-size: 0.9rem; color: #1f1535; margin-bottom: 3px; }
    .rec-desc { font-size: 0.85rem; color: #6b7280; line-height: 1.5; }
    .foot { text-align: center; color: #c4b8ea; font-size: 0.75rem; padding: 1.5rem 0 0.5rem; font-weight: 500; }
    .info-bar { background: linear-gradient(90deg, #f0f4ff, #fff5f0); border-radius: 14px; padding: 0.8rem 1.2rem; margin-top: 1rem; font-size: 0.75rem; color: #6b7280; font-weight: 500; text-align: center; border: 1px solid rgba(0,0,0,0.04); }
    @keyframes confetti-fall { 0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; } 100% { transform: translateY(100vh) rotate(720deg); opacity: 0; } }
    .confetti { position: fixed; width: 10px; height: 10px; top: -10px; animation: confetti-fall 3s ease-out forwards; z-index: 9999; }
    @keyframes pulse-glow { 0%, 100% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15); } 50% { box-shadow: 0 8px 48px rgba(16, 185, 129, 0.3); } }
    .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
    """

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─── Load Model Artifacts ────
@st.cache_resource(show_spinner="Loading model…")
def load_models():
    rf        = joblib.load('models/random_forest.pkl')
    scaler    = joblib.load('models/scaler.pkl')
    encoders  = joblib.load('models/label_encoders.pkl')
    feat_cols = joblib.load('models/feature_columns.pkl')
    ohe_cats  = joblib.load('models/ohe_categories.pkl')
    metrics   = joblib.load('models/model_metrics.pkl')
    metrics.setdefault('dataset_size', 3000)
    metrics.setdefault('placed_rate', 73.0)
    return rf, scaler, encoders, feat_cols, ohe_cats, metrics

rf_model, scaler, encoders, feature_cols, ohe_categories, metrics = load_models()


# ─── Build Input DataFrame for Prediction ───
def build_input(fv: dict) -> pd.DataFrame:
    row = fv.copy()
    norm_enc = {
        k.lower().replace(" ","").replace("_","").replace("-",""): k
        for k in encoders.keys()
    }
    def enc_key(field):
        n = field.lower().replace(" ","").replace("_","").replace("-","")
        if n in norm_enc: return norm_enc[n]
        for nk, ak in norm_enc.items():
            if n in nk or nk in n: return ak
        return None

    for col in ['Gender','Internships(Y/N)','Training(Y/N)','Any Backlogs?',
                'Innovative Project(Y/N)','Technical skills(Y/N)']:
        val = row[col]
        ek = enc_key(col)
        if ek and ek in encoders:
            row[col] = encoders[ek].transform([val])[0] if val in encoders[ek].classes_ else 0
        else:
            row[col] = 1 if val in ['Yes','Male','True','1'] else 0

    for col in ['10th board','12th board','Stream']:
        val = row.pop(col)
        for cat in ohe_categories[col][1:]:
            row[f"{col}_{cat}"] = 1 if val == cat else 0

    df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[feature_cols]


# ─── Generate Recommendations Based on Profile ────
def get_recommendations(fv: dict, prediction: int) -> list:
    cgpa   = float(fv['Cgpa'])
    blogs  = int(fv['_backlogs'])
    ints   = int(fv['_internships'])
    projs  = int(fv['_projects'])
    trains = int(fv['_trainings'])
    comm   = int(fv['Communication level'])
    tech   = fv['Technical skills(Y/N)']
    t10    = float(fv['10th marks'])
    t12    = float(fv['12th marks'])

    tips = []

    if blogs > 0:
        tips.append(dict(icon='⛔', title='Clear your backlogs immediately',
            desc=f'You have {blogs} pending backlog(s). 80%+ of companies auto-reject candidates with active backlogs — this is your #1 priority before anything else.',
            urgency='urgent'))
    if cgpa < 6.0:
        tips.append(dict(icon='📉', title=f'CGPA {cgpa:.2f} is critically low',
            desc='Most companies filter at 6.5 minimum during resume screening. Below 6.0, you may not clear the eligibility round at all. Prioritise your grades this semester.',
            urgency='urgent'))
    if t10 < 55 or t12 < 55:
        tips.append(dict(icon='📋', title='10th / 12th marks below threshold',
            desc='Many companies require 60%+ in both 10th and 12th for basic eligibility. Review criteria for each company you target.',
            urgency='urgent'))

    if 6.0 <= cgpa < 7.5:
        tips.append(dict(icon='📚', title='Target a CGPA above 7.5',
            desc='A CGPA of 7.5+ is the standard filter for most campus drives. Even a 0.3-point improvement this semester meaningfully expands your options.',
            urgency='improve'))
    elif 7.5 <= cgpa < 8.5:
        tips.append(dict(icon='📚', title='Push towards 8.5+ CGPA',
            desc='Top-tier companies commonly filter at 8.0–8.5. Consistent performance over the next semester can get you there.',
            urgency='improve'))

    if ints == 0:
        tips.append(dict(icon='🎯', title='Get at least one internship',
            desc='An internship — even 4 weeks, even virtual — signals real-world exposure to recruiters. Apply on Internshala, LinkedIn, or company career portals now.',
            urgency='improve'))
    if projs == 0:
        tips.append(dict(icon='🔬', title='Build 2–3 projects',
            desc='Interviewers routinely ask "what have you built?" GitHub projects in your domain are among the strongest differentiators in tech interviews.',
            urgency='improve'))
    if tech == 'No':
        tips.append(dict(icon='💻', title='Develop your technical skills',
            desc='Certify in at least one area: DSA on LeetCode, Python, SQL, or a cloud platform (AWS Free Tier). Technical aptitude tests appear in 90%+ of placement rounds.',
            urgency='improve'))
    if comm <= 2:
        tips.append(dict(icon='🗣️', title='Improve communication skills',
            desc='Weak communication is a frequent reason for rejections even after clearing technical rounds. Join mock GD/PI groups or practice with peers regularly.',
            urgency='improve'))
    elif comm == 3:
        tips.append(dict(icon='🗣️', title='Level up communication to 4+',
            desc='Average communication often costs candidates their offer after the HR round. Practice structured answers (STAR method) and mock interviews.',
            urgency='improve'))
    if trains == 0 and ints == 0:
        tips.append(dict(icon='🏅', title='Add certifications or training',
            desc='NPTEL, Coursera, or company-run certifications fill the experience gap. They signal initiative and add legitimacy to your resume when work experience is thin.',
            urgency='improve'))

    if cgpa >= 8.0 and blogs == 0:
        tips.append(dict(icon='✅', title=f'Strong academic record ({cgpa:.2f} CGPA)',
            desc='You clear the academic filter for the vast majority of campus drives. Keep consistency through your final semester.',
            urgency='good'))
    if ints >= 1:
        tips.append(dict(icon='✅', title=f'{ints} internship(s) — great signal',
            desc='Internship experience is one of the strongest signals on a student resume. Prepare to speak fluently about the work, impact, and learnings from each role.',
            urgency='good'))
    if projs >= 2:
        tips.append(dict(icon='✅', title=f'{projs} projects on your profile',
            desc='A strong project portfolio shows initiative and practical skills. Make sure each project is documented clearly on GitHub with a README.',
            urgency='good'))

    has_gaps = any(t['urgency'] in ('urgent','improve') for t in tips)
    if prediction == 1 and not has_gaps:
        tips.append(dict(icon='🚀', title='You are in the competitive zone',
            desc='Your profile clears most filters. Now focus: master DSA + system design, polish a 1-page ATS-friendly resume with quantified results, and practice mock interviews until confident.',
            urgency='good'))

    if not tips:
        tips.append(dict(icon='📌', title='Start building your profile now',
            desc='No single factor disqualifies you, but internships + projects + skills compound into a strong placement profile. Start with one project this month.',
            urgency='improve'))

    return tips


# ─── Animation Generators ──────
def generate_confetti_html():
    colors = ['#4f46e5', '#7c3aed', '#ec4899', '#f97316', '#10b981', '#06b6d4', '#f43f5e', '#8b5cf6']
    html = ""
    for i in range(50):
        left = np.random.randint(0, 100)
        delay = np.random.uniform(0, 2)
        duration = np.random.uniform(2, 4)
        color = colors[i % len(colors)]
        html += f'<div class="confetti" style="left:{left}%;background:{color};animation-delay:{delay}s;animation-duration:{duration}s;"></div>'
    return html


def generate_fireworks_html():
    colors = ['#4f46e5', '#ec4899', '#f97316', '#10b981', '#06b6d4']
    html = ""
    for i in range(20):
        left = np.random.randint(20, 80)
        top = np.random.randint(10, 40)
        delay = np.random.uniform(0, 1.5)
        color = colors[i % len(colors)]
        html += f'<div class="firework" style="left:{left}%;top:{top}%;background:{color};animation-delay:{delay}s;"></div>'
    return html


def generate_rain_html():
    html = ""
    for i in range(30):
        left = np.random.randint(0, 100)
        delay = np.random.uniform(0, 2)
        duration = np.random.uniform(1, 2)
        html += f'<div class="rain-drop" style="left:{left}%;animation-delay:{delay}s;animation-duration:{duration}s;"></div>'
    return html


# ─── Session State ─────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'result_data' not in st.session_state:
    st.session_state.result_data = None


# PAGE 1 — HOMEPAGE
if st.session_state.page == 'home':

    st.markdown("""
    <div class="side-decoration left">🎓</div>
    <div class="side-decoration right">🎯</div>
    """, unsafe_allow_html=True)

    particles_html = ""
    colors = ['#4f46e5', '#ec4899', '#f97316', '#10b981']
    for i in range(8):
        size = np.random.randint(8, 20)
        left = np.random.randint(5, 95)
        top = np.random.randint(5, 90)
        delay = np.random.uniform(0, 5)
        color = colors[i % len(colors)]
        particles_html += f'<div class="particle" style="width:{size}px;height:{size}px;left:{left}%;top:{top}%;background:{color};animation-delay:{delay}s;"></div>'
    st.markdown(particles_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-hero">
        <div class="home-hero-img">🎓</div>
        <div class="home-badge">✦ AI-Powered · Instant Results</div>
        <div class="home-title"><span class="gradient-text">Campus Placement</span><br><span class="gradient-text">Predictor</span></div>
        <div class="home-sub">Predict your placement chances with our advanced AI model. Get honest feedback and personalised improvement tips tailored to your profile.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Prediction</div>
            <div class="feature-desc">Random Forest model trained on 3,000+ student records for accurate placement forecasting.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Profile Breakdown</div>
            <div class="feature-desc">Visual score analysis across CGPA, skills, internships, projects, and communication.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Smart Tips</div>
            <div class="feature-desc">Personalised recommendations prioritised by urgency — fix critical gaps first.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀  Get Started — Predict My Placement", type="primary", width='stretch'):
            st.session_state.page = 'input'
            st.rerun()

    st.markdown('<div class="foot">Developed by Bhavana Chennu · Powered by Random Forest AI</div>',
                unsafe_allow_html=True)


#  PAGE 2 — INPUT FORM

elif st.session_state.page == 'input':

    st.markdown("""
    <div class="input-header">
        <h2>🧑‍💻 <span class="gradient-text">Enter Your Details Correctly</span></h2>
        <p>Fill in your academic and experience information for an honest AI prediction</p>
    </div>
    """, unsafe_allow_html=True)

    cb, _ = st.columns([1, 4])
    with cb:
        if st.button("← Back to Home", type="secondary"):
            st.session_state.page = 'home'
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    with st.form("pf", clear_on_submit=False, enter_to_submit=False):

        st.markdown("""
        <div class="form-section">
            <div class="form-section-dot" style="background:#7c3aed;"></div>
            <span class="form-section-emoji">📚</span>
            <span class="form-section-label">Academic Profile</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: gender = st.selectbox("Gender", ["Male", "Female"])
        with c2: stream = st.selectbox("Stream", ["CS", "ECE", "ME", "Civil", "EE", "IT", "Other"])
        with c3: cgpa   = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.5, 0.01, "%.2f")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4: tenth_board  = st.selectbox("10th Board", ["CBSE", "State", "ICSE", "Other"])
        with c5: tenth_marks  = st.number_input("10th Marks (%)", 0.0, 100.0, 75.0, 0.1, "%.1f")
        with c6: twelfth_board = st.selectbox("12th Board", ["CBSE", "State", "ICSE", "Other"])

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        c7, c8, c9 = st.columns(3)
        with c7: twelfth_marks = st.number_input("12th Marks (%)", 0.0, 100.0, 72.0, 0.1, "%.1f")
        with c8: communication = st.selectbox("Communication Level", [1, 2, 3, 4, 5], index=2,
                                               help="1 = Poor  |  3 = Average  |  5 = Excellent")
        with c9: technical = st.selectbox("Technical Skills", ["No", "Yes"])

        st.markdown("""
        <div class="form-section" style="margin-top:1.5rem;">
            <div class="form-section-dot" style="background:#06b6d4;"></div>
            <span class="form-section-emoji">💼</span>
            <span class="form-section-label">Experience &amp; Activities</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        c10, c11, c12, c13 = st.columns(4)
        with c10: internships = st.number_input("Internships",          0, 20, 0, 1, help="Completed internships")
        with c11: trainings   = st.number_input("Trainings / Courses",  0, 20, 0, 1, help="Certified training programs")
        with c12: projects    = st.number_input("Projects",             0, 20, 0, 1, help="Academic or personal projects")
        with c13: backlogs    = st.number_input("Active Backlogs",      0, 20, 0, 1, help="Pending backlogs right now")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮  Predict My Placement Chances", width='stretch')

    if submitted:
        fv = {
            'Gender': gender, '10th board': tenth_board, '10th marks': tenth_marks,
            '12th board': twelfth_board, '12th marks': twelfth_marks, 'Stream': stream,
            'Cgpa': cgpa,
            'Internships(Y/N)':       "Yes" if internships > 0 else "No",
            'Training(Y/N)':          "Yes" if trainings   > 0 else "No",
            'Any Backlogs?':          "Yes" if backlogs    > 0 else "No",
            'Innovative Project(Y/N)':"Yes" if projects    > 0 else "No",
            'Communication level': communication,
            'Technical skills(Y/N)': technical,
            '_backlogs': backlogs, '_internships': internships,
            '_projects': projects, '_trainings': trainings,
        }
        input_df   = build_input({k: v for k, v in fv.items() if not k.startswith('_')})
        prediction = rf_model.predict(input_df)[0]
        proba      = rf_model.predict_proba(input_df)[0]
        placed_pct = round(proba[1] * 100, 1)
        recs       = get_recommendations(fv, prediction)

        st.session_state.result_data = {
            'prediction': prediction, 'placed_pct': placed_pct, 'recs': recs,
            'cgpa': cgpa, 'backlogs': backlogs, 'internships': internships,
            'projects': projects, 'communication': communication, 'technical': technical,
        }
        st.session_state.page = 'result'
        st.rerun()

    st.markdown('<div class="foot">Developed by Bhavana Chennu · Powered by Random Forest AI</div>',
                unsafe_allow_html=True)


#  PAGE 3 — RESULTS

elif st.session_state.page == 'result':
    d          = st.session_state.result_data
    prediction = d['prediction']
    placed_pct = d['placed_pct']
    recs       = d['recs']
    cgpa       = d['cgpa']
    backlogs   = d['backlogs']
    internships= d['internships']
    projects   = d['projects']
    comm       = d['communication']
    tech       = d['technical']

    # Auto-scroll to top so heading and buttons are visible first
    st.markdown("""
    <script>
        window.scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-heading">
        🎯 <span class="gradient-text">Here is Your Honest Result</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    cb1, cb2, _ = st.columns([1, 1, 3])
    with cb1:
        if st.button("← Back to Home", type="secondary"):
            st.session_state.page = 'home'
            st.rerun()
    with cb2:
        if st.button("✏️ Edit Details", type="secondary"):
            st.session_state.page = 'input'
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(generate_confetti_html(), unsafe_allow_html=True)
        st.markdown("""
        <div class="celebration-container">
        """ + generate_fireworks_html() + """
        <div class="result-hero placed pulse-glow">
            <div class="rh-emoji">🎉</div>
            <div class="rh-verdict placed">Likely to be Placed!</div>
            <div class="rh-pct">""" + str(placed_pct) + """%</div>
            <div class="rh-sub">placement probability · Random Forest AI model</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="position:relative;overflow:hidden;">
        """ + generate_rain_html() + """
        <div class="result-hero not-placed">
            <div class="rh-emoji">💪</div>
            <div class="rh-verdict not-placed">Needs Improvement</div>
            <div class="rh-pct">""" + str(placed_pct) + """%</div>
            <div class="rh-sub">placement probability · let's work on this together</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    profile_items = [
        ("CGPA",          min(cgpa / 10.0, 1.0),             "#7c3aed"),
        ("Communication", (comm - 1) / 4.0,                  "#06b6d4"),
        ("Internships",   min(internships / 3.0, 1.0),        "#10b981"),
        ("Projects",      min(projects / 5.0, 1.0),           "#f97316"),
        ("No Backlogs",   1.0 if backlogs == 0
                          else max(0.0, 1.0 - backlogs*0.33), "#f43f5e"),
    ]

    bars_html = '<div class="profile-score-wrap"><div class="ps-title">Your Profile Breakdown</div>'
    for label, score, color in profile_items:
        pct = int(score * 100)
        bars_html += (
            '<div class="ps-row">'
            '<div class="ps-label">' + label + '</div>'
            '<div class="ps-bar-bg">'
            '<div class="ps-bar-fill" style="width:' + str(pct) + '%;background:' + color + ';"></div>'
            '</div>'
            '<div class="ps-pct" style="color:' + color + ';">' + str(pct) + '%</div>'
            '</div>'
        )
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    urgent  = [r for r in recs if r['urgency'] == 'urgent']
    improve = [r for r in recs if r['urgency'] == 'improve']
    good    = [r for r in recs if r['urgency'] == 'good']

    st.markdown('<div class="rec-section-title" style="color:#2d1b69;margin-top:1rem;">📋 Personalised Recommendations</div>',
                unsafe_allow_html=True)

    def render_cards(items):
        out = ""
        for r in items:
            out += (
                '<div class="rec-card ' + r["urgency"] + '">'
                '<div class="rec-icon">' + r["icon"] + '</div>'
                '<div>'
                '<div class="rec-title">' + r["title"] + '</div>'
                '<div class="rec-desc">' + r["desc"] + '</div>'
                '</div></div>'
            )
        return out

    if urgent:
        st.markdown('<div class="rec-tier-label" style="color:#f43f5e;">🚨 Critical — Fix These First</div>', unsafe_allow_html=True)
        st.markdown(render_cards(urgent), unsafe_allow_html=True)
    if improve:
        st.markdown('<div class="rec-tier-label" style="color:#f97316;">📈 Areas to Improve</div>', unsafe_allow_html=True)
        st.markdown(render_cards(improve), unsafe_allow_html=True)
    if good:
        st.markdown('<div class="rec-tier-label" style="color:#10b981;">✨ Strengths</div>', unsafe_allow_html=True)
        st.markdown(render_cards(good), unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-bar">'
        f'🤖 Prediction by Random Forest model trained on {metrics["dataset_size"]:,} student records. '
        f'CGPA = primary academic gate · backlogs = hard filter · skills + communication = interview success.</div>',
        unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Predict for Another Student", width='stretch'):
        st.session_state.page = 'input'
        st.rerun()

    st.markdown('<div class="foot">Developed by Bhavana Chennu · Powered by Random Forest AI</div>',
                unsafe_allow_html=True)