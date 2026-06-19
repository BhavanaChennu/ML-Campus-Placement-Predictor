# 🎓 Campus Placement Predictor v2.0

**AI-powered web application that predicts campus placement chances using 15 real-world factors across academics, skills, and experience.**

Built with **Streamlit** + **Random Forest ML** — enter your complete profile, get an instant prediction with actionable improvement tips.

---

## 🪟 Project Overview

This application analyzes **15 placement factors** across 3 dimensions to predict placement probability:

### 📚 Academic (6 factors)
- Gender, Stream, B.Tech CGPA, 10th Board & Marks, 12th Board & Marks, Active Backlogs

### 🧠 Skills (4 factors)
- Coding Score, Aptitude Score, Communication Level, Technical Skills (Yes/No)

### 💼 Experience (5 factors)
- Internship Count, Project Count, Certifications Count, Hackathons Count, Trainings Attended

### What it does:
- **Predicts placement probability** using Random Forest trained on **5,000+ student records**
- **Visual profile breakdown** — score bars for all 15 factors
- **Personalised recommendations** — sorted by urgency (Critical → Improve → Strengths)
- **Interactive animations** — confetti/fireworks for placed, rain drops for needs improvement
- **3-page navigation** — Home → Profile Input → Results
- **Edit Details** — go back and modify your inputs without re-entering everything

---

## 📂 Project Structure

```
Campus-Placement-Predictor/
├── app.py                  # Streamlit web app (3-page UI)
├── styles.css              # Custom CSS styling, animations, gradients
├── train_model.py          # Model training script (Random Forest)
├── requirements.txt        # Python dependencies
├── Dataset/
│   └── Placements_data.csv # 5,000 student records with 17 features
├── models/                 # Saved model artifacts (auto-generated)
│   ├── random_forest.pkl       # Trained classifier
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoders.pkl      # Categorical encoders
│   ├── feature_columns.pkl     # Feature order
│   ├── ohe_categories.pkl      # One-hot encoding categories
│   ├── model_metrics.pkl       # Accuracy, AUC, CV scores
│   └── confusion_matrix.png    # Visualization
└── README.md
```

---

## ⚓ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Streamlit** | Web framework for interactive UI |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Random Forest classifier, preprocessing, metrics |
| **Joblib** | Model serialization (`.pkl` files) |
| **Matplotlib** | Confusion matrix visualization |
| **CSS3** | Custom styling, animations, gradients |

---

## 🏃🏻‍♀️‍➡️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/campus-placement-predictor.git
cd campus-placement-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (first time only)
```bash
python train_model.py
```

This will:
- Load and preprocess `Dataset/Placements_data.csv`
- Train a Random Forest classifier with **5-fold cross-validation**
- Save all model artifacts to `models/`
- Generate a confusion matrix plot

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare your GitHub repository

Commit these files:
```
app.py
requirements.txt
styles.css
Dataset/Placements_data.csv
models/ (all .pkl files — REQUIRED for cloud)
```

**Note:** `models/` must be committed even though `.pkl` files are binary. Streamlit Cloud cannot run `train_model.py` during deployment.

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Main file path: `app.py`
6. Click **Deploy**

You'll get a public URL like:
```
https://yourusername-campus-placement-predictor-app-abc123.streamlit.app
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🤖 **15-Factor AI Prediction** | Random Forest on 5,000 records with academic + skills + experience |
| 📊 **3-Section Profile Breakdown** | Visual score bars across all 15 factors |
| 💡 **Smart Recommendations** | Prioritised by urgency: Critical → Improve → Strengths |
| 🎉 **Celebration Animations** | Confetti + fireworks for placed, rain drops for needs improvement |
| 🎨 **Modern UI** | Gradient headings, floating particles, smooth transitions |
| 📱 **Responsive Layout** | Wide layout with decorative side elements |
| ✏️ **Edit Details** | Modify inputs after seeing results — values preserved in session state |

---

## 📝 Input Features (15 Factors)

### 📚 Academic
| Feature | Type | Range |
|---------|------|-------|
| Gender | Categorical | Male / Female / Others |
| Stream | Categorical | Computer Science, IT, Electronics, Mechanical, Civil, Electrical, Other |
| B.Tech CGPA | Numeric | 0.0 – 10.0 |
| 10th Board | Categorical | CBSE / State / ICSE / Other |
| 10th Marks | Numeric | 0 – 100% |
| 12th Board | Categorical | CBSE / State / ICSE / Other |
| 12th Marks | Numeric | 0 – 100% |
| Active Backlogs | Numeric | 0 – 20 |

### 🧠 Skills
| Feature | Type | Range |
|---------|------|-------|
| Coding Score | Numeric | 0 – 100 |
| Aptitude Score | Numeric | 0 – 100 |
| Communication Level | Numeric | 1 – 5 |
| Technical Skills | Categorical | Yes / No |

### 💼 Experience
| Feature | Type | Range |
|---------|------|-------|
| Internship Count | Numeric | 0 – 20 |
| Project Count | Numeric | 0 – 20 |
| Certifications Count | Numeric | 0 – 20 |
| Hackathons Count | Numeric | 0 – 20 |
| Trainings / Courses | Numeric | 0 – 20 |

---

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier (1,000 estimators)
- **Test Accuracy**: **94%**
- **Cross-Validation**: 5-fold stratified — **93.4% ± 0.8%**
- **ROC-AUC**: **0.967**
- **Class Balancing**: `class_weight='balanced'`
- **Dataset Size**: 5,000 student records
- **Placement Rate**: ~81%

### Top Feature Importances
| Feature | Importance |
|---------|------------|
| CGPA | Highest |
| Backlogs | Very High |
| Coding Score | High |
| Internships | High |
| Projects | High |
| Aptitude Score | Moderate |
| Communication Level | Moderate |
| Technical Skills | Moderate |

---

## 📌 About the `.pkl` Files

The `models/` folder contains **binary serialized files** — standard for ML deployment:

| File | Purpose |
|------|---------|
| `random_forest.pkl` | Trained model weights |
| `scaler.pkl` | Feature scaling parameters |
| `label_encoders.pkl` | Categorical → numeric mappings |
| `feature_columns.pkl` | Exact column order for inference |
| `ohe_categories.pkl` | One-hot encoding category mappings |
| `model_metrics.pkl` | Accuracy, AUC, CV scores, dataset size |

You **cannot open them in a text editor** — they are loaded by `joblib.load()`. For Streamlit Cloud, commit these files to GitHub.

---

## 📝 Notes

- This project is for **educational and demonstration purposes**.
- The app runs **locally** — no cloud deployment or API calls required.
- Model artifacts are **auto-generated** — run `train_model.py` first if `models/` is missing.
- Dataset is **synthetic but realistic** — correlations mirror real placement patterns.
- **Count-based features** (Internships, Projects, Backlogs, etc.) are passed directly to the model for maximum predictive power.
- **Technical Skills** is kept as Yes/No as per design choice.

---

## 🔩 Potential Improvements

- Add real college placement data for even higher accuracy
- Implement XGBoost / LightGBM comparison
- Add salary package prediction (regression)
- Export results as PDF reports
- Add user authentication for multi-user access
- Deploy to Hugging Face Spaces or AWS

---

## 👩‍💻 Author

**Bhavana Chennu**

Built to help students prepare smarter for campus placements.

---

## 📄 License

This project is open-source. Feel free to use, modify, and share!
