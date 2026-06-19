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
try:
    with open("styles.css") as f:
        css = f.read()
except FileNotFoundError:
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
    metrics.setdefault('dataset_size', 5000)
    metrics.setdefault('placed_rate', 68.0)
    return rf, scaler, encoders, feat_cols, ohe_cats, metrics

rf_model, scaler, encoders, feature_cols, ohe_categories, metrics = load_models()


# ─── Build Input DataFrame for Prediction ───
def build_input(fv: dict) -> pd.DataFrame:
    row = fv.copy()

    # Encode binary categorical columns
    for col in ['Gender', 'Technical skills(Y/N)']:
        val = row[col]
        if col in encoders:
            row[col] = encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
        else:
            row[col] = 1 if val == 'Male' else 0

    # One-hot encode multi-class categorical columns
    for col in ['10th board', '12th board', 'Stream']:
        val = row.pop(col)
        for cat in ohe_categories[col][1:]:
            row[f"{col}_{cat}"] = 1 if val == cat else 0

    # Build DataFrame with exact feature column order
    df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[feature_cols]


# ─── Generate Recommendations Based on Profile ────
def get_recommendations(fv: dict, prediction: int) -> list:
    cgpa   = float(fv['Cgpa'])
    blogs  = int(fv['Backlogs'])
    ints   = int(fv['Internships'])
    projs  = int(fv['Projects'])
    trains = int(fv['Trainings'])
    comm   = int(fv['Communication level'])
    tech   = fv['Technical skills(Y/N)']
    t10    = float(fv['10th marks'])
    t12    = float(fv['12th marks'])
    coding = int(fv['Coding Score'])
    apt    = int(fv['Aptitude Score'])
    hacks  = int(fv['Hackathons Count'])
    certs  = int(fv['Certifications Count'])

    tips = []

    # ─── URGENT ───
    if blogs > 0:
        tips.append(dict(icon='⛔', title=f'Clear your {blogs} backlog{"s" if blogs > 1 else ""} immediately',
            desc=f'You have {blogs} pending backlog{"s" if blogs > 1 else ""}. Most companies auto-reject candidates with active backlogs — this is your #1 priority.',
            urgency='urgent'))
    if cgpa < 5.5:
        tips.append(dict(icon='📉', title=f'CGPA {cgpa:.2f} is critically low',
            desc='Most companies filter at 6.0+ minimum during resume screening. Below 5.5, you are likely to be filtered out before interviews even begin.',
            urgency='urgent'))
    if t10 < 50 or t12 < 50:
        tips.append(dict(icon='📋', title='10th / 12th marks below 50%',
            desc='Many companies require 60%+ in both 10th and 12th for basic eligibility. Some relax to 50% but options become very limited.',
            urgency='urgent'))
    if coding < 30 and fv['Stream'] in ['Computer Science', 'Information Technology']:
        tips.append(dict(icon='💻', title='Coding score critically low for CS/IT',
            desc=f'Your coding score ({coding}) is far below the average for CS/IT students. DSA and problem-solving are non-negotiable for tech placements.',
            urgency='urgent'))

    # ─── IMPROVE ───
    if 5.5 <= cgpa < 6.5:
        tips.append(dict(icon='📚', title='Push CGPA above 6.5',
            desc='A CGPA of 6.5+ is the minimum filter for most campus drives. Focus on scoring well in upcoming exams.',
            urgency='improve'))
    elif 6.5 <= cgpa < 7.5:
        tips.append(dict(icon='📚', title='Target 7.5+ CGPA for better opportunities',
            desc='Top-tier companies commonly filter at 7.0–7.5. Consistent effort this semester can unlock better companies.',
            urgency='improve'))
    elif 7.5 <= cgpa < 8.5:
        tips.append(dict(icon='📚', title='Aim for 8.5+ CGPA',
            desc='Premium companies and core roles often filter at 8.0+. You are close — push for that extra edge.',
            urgency='improve'))

    if ints == 0:
        tips.append(dict(icon='🎯', title='Get at least one internship',
            desc='Zero internships is a red flag for most recruiters. Even a 4-week virtual internship adds credibility. Apply on Internshala, LinkedIn, or company portals.',
            urgency='improve'))
    elif ints == 1:
        tips.append(dict(icon='🎯', title='Consider a second internship',
            desc='One internship is good, but two+ shows sustained industry exposure. It significantly improves your resume weight.',
            urgency='improve'))

    if projs == 0:
        tips.append(dict(icon='🔬', title='Build 2–3 quality projects',
            desc='Interviewers always ask "what have you built?" Start with one project in your domain and host it on GitHub with a clean README.',
            urgency='improve'))
    elif projs == 1:
        tips.append(dict(icon='🔬', title='Add one more project',
            desc='A single project is thin. Build a second one that demonstrates a different skill or technology stack.',
            urgency='improve'))

    if tech == 'No':
        tips.append(dict(icon='⚡', title='Develop technical skills',
            desc='Technical skills are essential for almost every placement round. Pick one: Python, Java, SQL, or a cloud platform (AWS/GCP free tier).',
            urgency='improve'))

    if coding < 50:
        tips.append(dict(icon='💻', title=f'Improve coding score (currently {coding})',
            desc='Practice on LeetCode, HackerRank, or CodeChef. Aim for 60+ to clear most technical aptitude rounds comfortably.',
            urgency='improve'))
    elif 50 <= coding < 70:
        tips.append(dict(icon='💻', title=f'Push coding score from {coding} to 70+',
            desc='You are in the average zone. 70+ puts you in the top tier for coding rounds. Focus on arrays, strings, and basic DSA.',
            urgency='improve'))

    if apt < 50:
        tips.append(dict(icon='🧮', title=f'Improve aptitude score (currently {apt})',
            desc='Aptitude tests are the first gate in 90% of companies. Practice quantitative, logical, and verbal sections daily.',
            urgency='improve'))

    if comm <= 2:
        tips.append(dict(icon='🗣️', title='Improve communication skills',
            desc='Weak communication is a top reason for HR round rejections. Join mock GD/PI groups or practice with peers regularly.',
            urgency='improve'))
    elif comm == 3:
        tips.append(dict(icon='🗣️', title='Level up communication to 4+',
            desc='Average communication often costs candidates their offer after the HR round. Practice structured answers using the STAR method.',
            urgency='improve'))

    if hacks == 0:
        tips.append(dict(icon='🏆', title='Participate in a hackathon',
            desc='Hackathons demonstrate problem-solving under pressure and teamwork. Even participation without winning adds value to your profile.',
            urgency='improve'))

    if certs == 0 and ints == 0:
        tips.append(dict(icon='🏅', title='Add certifications or training',
            desc='NPTEL, Coursera, or AWS Free Tier certifications signal initiative. They help when work experience is limited.',
            urgency='improve'))
    elif certs == 0 and ints >= 1:
        tips.append(dict(icon='🏅', title='Add a certification',
            desc='You have internship experience — add a certification to round out your profile and show continuous learning.',
            urgency='improve'))

    if trains == 0:
        tips.append(dict(icon='📖', title='Pursue a training or workshop',
            desc='Formal training shows structured learning. Look for college workshops, NPTEL courses, or company-sponsored programs.',
            urgency='improve'))

    # ─── GOOD ───
    if cgpa >= 8.0 and blogs == 0:
        tips.append(dict(icon='✅', title=f'Strong academic record ({cgpa:.2f} CGPA)',
            desc='You clear the academic filter for most campus drives. Maintain consistency through your final semester.',
            urgency='good'))
    if ints >= 2:
        tips.append(dict(icon='✅', title=f'{ints} internships — excellent signal',
            desc='Multiple internships show sustained industry engagement. Be ready to articulate what you learned and delivered in each role.',
            urgency='good'))
    elif ints == 1 and cgpa >= 7.0:
        tips.append(dict(icon='✅', title='Balanced profile: internship + academics',
            desc='You have both internship exposure and a solid CGPA. This is a well-rounded profile for most companies.',
            urgency='good'))
    if projs >= 3:
        tips.append(dict(icon='✅', title=f'{projs} projects — strong portfolio',
            desc='A multi-project portfolio shows depth and versatility. Ensure each has clear documentation and a live demo or GitHub link.',
            urgency='good'))
    if coding >= 75:
        tips.append(dict(icon='✅', title=f'Excellent coding score ({coding})',
            desc='Top-tier coding scores give you a significant edge in technical rounds. Keep practicing to maintain this advantage.',
            urgency='good'))
    if apt >= 75:
        tips.append(dict(icon='✅', title=f'Strong aptitude ({apt})',
            desc='You are well-prepared for aptitude screening rounds. This is a reliable strength across all company types.',
            urgency='good'))
    if hacks >= 2:
        tips.append(dict(icon='✅', title=f'{hacks} hackathon{"s" if hacks > 1 else ""} — great initiative',
            desc='Hackathon participation shows competitive spirit and rapid problem-solving. Highlight specific challenges you solved.',
            urgency='good'))
    if certs >= 2:
        tips.append(dict(icon='✅', title=f'{certs} certifications — shows commitment',
            desc='Multiple certifications demonstrate continuous learning. Make sure they are relevant to your target roles.',
            urgency='good'))
    if comm >= 4:
        tips.append(dict(icon='✅', title='Strong communication skills',
            desc='Good communication is a major differentiator in HR and managerial rounds. Use this strength to your advantage.',
            urgency='good'))

    # Default tip if profile is neutral
    if not tips:
        tips.append(dict(icon='📌', title='Build a stronger profile step by step',
            desc='Focus on one area at a time: add an internship, build a project, or improve your coding score. Small consistent efforts compound.',
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
def init_session_state():
    defaults = {
        'page': 'home',
        'result_data': None,
        'form_gender': None, 'form_stream': None, 'form_cgpa': None,
        'form_tenth_board': None, 'form_tenth_marks': None,
        'form_twelfth_board': None, 'form_twelfth_marks': None,
        'form_communication': None, 'form_technical': None,
        'form_internships': None, 'form_trainings': None,
        'form_projects': None, 'form_backlogs': None,
        'form_coding_score': None, 'form_aptitude_score': None,
        'form_hackathons': None, 'form_certifications': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


def save_form_values(gender, stream, cgpa, tenth_board, tenth_marks, twelfth_board,
                     twelfth_marks, communication, technical, internships, trainings,
                     projects, backlogs, coding_score, aptitude_score, hackathons,
                     certifications):
    st.session_state.form_gender = gender
    st.session_state.form_stream = stream
    st.session_state.form_cgpa = cgpa
    st.session_state.form_tenth_board = tenth_board
    st.session_state.form_tenth_marks = tenth_marks
    st.session_state.form_twelfth_board = twelfth_board
    st.session_state.form_twelfth_marks = twelfth_marks
    st.session_state.form_communication = communication
    st.session_state.form_technical = technical
    st.session_state.form_internships = internships
    st.session_state.form_trainings = trainings
    st.session_state.form_projects = projects
    st.session_state.form_backlogs = backlogs
    st.session_state.form_coding_score = coding_score
    st.session_state.form_aptitude_score = aptitude_score
    st.session_state.form_hackathons = hackathons
    st.session_state.form_certifications = certifications


def clear_form_values():
    keys = ['form_gender', 'form_stream', 'form_cgpa', 'form_tenth_board',
            'form_tenth_marks', 'form_twelfth_board', 'form_twelfth_marks',
            'form_communication', 'form_technical', 'form_internships',
            'form_trainings', 'form_projects', 'form_backlogs',
            'form_coding_score', 'form_aptitude_score', 'form_hackathons',
            'form_certifications']
    for k in keys:
        st.session_state[k] = None


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
            <div class="feature-desc">Random Forest model trained on 5,000+ student records for accurate placement forecasting.</div>
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

    s = st.session_state

    with st.form("pf", clear_on_submit=False, enter_to_submit=False):

        st.markdown("""
        <div class="form-section">
            <div class="form-section-dot" style="background:#7c3aed;"></div>
            <span class="form-section-emoji">📚</span>
            <span class="form-section-label">Academic Profile</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female", "Others"],
                                  index=None if s.form_gender is None else ["Male", "Female", "Others"].index(s.form_gender),
                                  placeholder="Select gender...")
        with c2:
            stream = st.selectbox("Stream", [
                "Computer Science", "Information Technology", "Electronics",
                "Mechanical", "Civil", "Electrical", "Other"
            ], index=None if s.form_stream is None else [
                "Computer Science", "Information Technology", "Electronics",
                "Mechanical", "Civil", "Electrical", "Other"
            ].index(s.form_stream), placeholder="Select stream...")
        with c3:
            cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, s.form_cgpa, 0.01, "%.2f")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            tenth_board = st.selectbox("10th Board", ["CBSE", "State", "ICSE", "Other"],
                                       index=None if s.form_tenth_board is None else ["CBSE", "State", "ICSE", "Other"].index(s.form_tenth_board),
                                       placeholder="Select board...")
        with c5:
            tenth_marks = st.number_input("10th Marks (%)", 0.0, 100.0, s.form_tenth_marks, 0.1, "%.1f")
        with c6:
            twelfth_board = st.selectbox("12th Board", ["CBSE", "State", "ICSE", "Other"],
                                         index=None if s.form_twelfth_board is None else ["CBSE", "State", "ICSE", "Other"].index(s.form_twelfth_board),
                                         placeholder="Select board...")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        c7, c8, c9 = st.columns(3)
        with c7:
            twelfth_marks = st.number_input("12th Marks (%)", 0.0, 100.0, s.form_twelfth_marks, 0.1, "%.1f")
        with c8:
            communication = st.selectbox("Communication Level", [1, 2, 3, 4, 5],
                                         index=None if s.form_communication is None else [1, 2, 3, 4, 5].index(s.form_communication),
                                         placeholder="Select level...")
        with c9:
            technical = st.selectbox("Technical Skills", ["No", "Yes"],
                                       index=None if s.form_technical is None else ["No", "Yes"].index(s.form_technical),
                                       placeholder="Select...")

        st.markdown("""
        <div class="form-section" style="margin-top:1.5rem;">
            <div class="form-section-dot" style="background:#06b6d4;"></div>
            <span class="form-section-emoji">💼</span>
            <span class="form-section-label">Experience &amp; Activities</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        c10, c11, c12, c13 = st.columns(4)
        with c10:
            internships = st.number_input("Internships", 0, 20, s.form_internships, 1)
        with c11:
            trainings = st.number_input("Trainings / Courses", 0, 20, s.form_trainings, 1)
        with c12:
            projects = st.number_input("Projects", 0, 20, s.form_projects, 1)
        with c13:
            backlogs = st.number_input("Active Backlogs", 0, 20, s.form_backlogs, 1)

        st.markdown("""
        <div class="form-section" style="margin-top:1.5rem;">
            <div class="form-section-dot" style="background:#f59e0b;"></div>
            <span class="form-section-emoji">🧠</span>
            <span class="form-section-label">Skills &amp; Assessments</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        c14, c15 = st.columns(2)
        with c14:
            coding_score = st.number_input("Coding Score (0-100)", 0, 100, s.form_coding_score, 1)
        with c15:
            aptitude_score = st.number_input("Aptitude Score (0-100)", 0, 100, s.form_aptitude_score, 1)

        c16, c17 = st.columns(2)
        with c16:
            hackathons_count = st.number_input("Hackathons Count", 0, 20, s.form_hackathons, 1)
        with c17:
            certifications_count = st.number_input("Certifications Count", 0, 20, s.form_certifications, 1)

        submitted = st.form_submit_button("🔮  Predict My Placement Chances", width='stretch')

    if submitted:
        required_fields = {
            'Gender': gender, 'Stream': stream, 'Cgpa': cgpa,
            '10th board': tenth_board, '10th marks': tenth_marks,
            '12th board': twelfth_board, '12th marks': twelfth_marks,
            'Communication level': communication, 'Technical skills(Y/N)': technical,
            'Internships': internships, 'Trainings': trainings,
            'Projects': projects, 'Backlogs': backlogs,
            'Coding Score': coding_score, 'Aptitude Score': aptitude_score,
            'Hackathons Count': hackathons_count, 'Certifications Count': certifications_count
        }

        empty_fields = [k for k, v in required_fields.items() if v is None]
        if empty_fields:
            save_form_values(gender, stream, cgpa, tenth_board, tenth_marks, twelfth_board,
                             twelfth_marks, communication, technical, internships, trainings,
                             projects, backlogs, coding_score, aptitude_score, hackathons_count,
                             certifications_count)
            st.error(f"⚠️ Please fill all fields. Missing: {', '.join(empty_fields)}")
        else:
            save_form_values(gender, stream, cgpa, tenth_board, tenth_marks, twelfth_board,
                             twelfth_marks, communication, technical, internships, trainings,
                             projects, backlogs, coding_score, aptitude_score, hackathons_count,
                             certifications_count)

            fv = {
                'Gender': gender, '10th board': tenth_board, '10th marks': tenth_marks,
                '12th board': twelfth_board, '12th marks': twelfth_marks, 'Stream': stream,
                'Cgpa': cgpa, 'Communication level': communication,
                'Technical skills(Y/N)': technical, 'Internships': internships,
                'Trainings': trainings, 'Projects': projects, 'Backlogs': backlogs,
                'Coding Score': coding_score, 'Aptitude Score': aptitude_score,
                'Hackathons Count': hackathons_count, 'Certifications Count': certifications_count,
            }

            input_df = build_input(fv)
            prediction = rf_model.predict(input_df)[0]
            proba = rf_model.predict_proba(input_df)[0]
            placed_pct = round(proba[1] * 100, 1)
            recs = get_recommendations(fv, prediction)

            st.session_state.result_data = {
                'prediction': prediction, 'placed_pct': placed_pct, 'recs': recs,
                'cgpa': cgpa, 'backlogs': backlogs, 'internships': internships,
                'projects': projects, 'communication': communication, 'technical': technical,
                'coding_score': coding_score, 'aptitude_score': aptitude_score,
                'hackathons': hackathons_count, 'trainings': trainings,
                'certifications': certifications_count,
            }
            st.session_state.page = 'result'
            st.rerun()

    st.markdown('<div class="foot">Developed by Bhavana Chennu · Powered by Random Forest AI</div>',
                unsafe_allow_html=True)


#  PAGE 3 — RESULTS
elif st.session_state.page == 'result':
    d = st.session_state.result_data
    prediction = d['prediction']
    placed_pct = d['placed_pct']
    recs = d['recs']
    cgpa = d['cgpa']
    backlogs = d['backlogs']
    internships = d['internships']
    projects = d['projects']
    comm = d['communication']
    tech = d['technical']
    coding = d['coding_score']
    apt = d['aptitude_score']
    hacks = d['hackathons']
    trains = d['trainings']
    certs = d['certifications']

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
        ("CGPA",          min(cgpa / 10.0, 1.0),                              "#7c3aed"),
        ("Coding",        min(coding / 100.0, 1.0),                           "#06b6d4"),
        ("Aptitude",      min(apt / 100.0, 1.0),                              "#f59e0b"),
        ("Communication", (comm - 1) / 4.0,                                   "#10b981"),
        ("Internships",   min(internships / 3.0, 1.0),                        "#ec4899"),
        ("Projects",      min(projects / 5.0, 1.0),                           "#f97316"),
        ("No Backlogs",   1.0 if backlogs == 0 else max(0.0, 1.0 - backlogs*0.25), "#f43f5e"),
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

    stats_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin:1rem 0;">'
    stats = [
        (f"🎯 {internships}", "Internship" + ("s" if internships != 1 else ""), "#ec4899"),
        (f"🔬 {projects}", "Project" + ("s" if projects != 1 else ""), "#f97316"),
        (f"📖 {trains}", "Training" + ("s" if trains != 1 else ""), "#8b5cf6"),
        (f"🏆 {hacks}", "Hackathon" + ("s" if hacks != 1 else ""), "#06b6d4"),
        (f"🏅 {certs}", "Certification" + ("s" if certs != 1 else ""), "#10b981"),
    ]
    for text, label, color in stats:
        stats_html += (
            f'<div style="background:rgba(255,255,255,0.95);border-radius:12px;padding:10px 16px;'
            f'border:1.5px solid {color}30;text-align:center;min-width:90px;">'
            f'<div style="font-size:1.1rem;font-weight:800;color:{color};">{text}</div>'
            f'<div style="font-size:0.7rem;color:#6b7280;font-weight:600;">{label}</div></div>'
        )
    stats_html += '</div>'
    st.markdown(stats_html, unsafe_allow_html=True)

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
        clear_form_values()
        st.session_state.page = 'input'
        st.rerun()

    st.markdown('<div class="foot">Developed by Bhavana Chennu · Powered by Random Forest AI</div>',
                unsafe_allow_html=True)