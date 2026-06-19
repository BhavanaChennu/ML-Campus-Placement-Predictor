import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CAMPUS PLACEMENT PREDICTOR v2.0 — MODEL TRAINING")
print("=" * 60)

os.makedirs('models', exist_ok=True)
os.makedirs('Dataset', exist_ok=True)

np.random.seed(42)
n = 5000

# ─── GENERATE DATASET (run once, then remove this section) ───
gender = np.random.choice(['Male', 'Female', 'Others'], n, p=[0.54, 0.44, 0.02])
stream = np.random.choice([
    'Computer Science', 'Information Technology', 'Electronics',
    'Mechanical', 'Civil', 'Electrical', 'Other'
], n, p=[0.30, 0.19, 0.11, 0.09, 0.09, 0.15, 0.07])

tenth_board = np.random.choice(['CBSE', 'State', 'ICSE', 'Other'], n, p=[0.42, 0.33, 0.15, 0.10])
twelfth_board = np.random.choice(['CBSE', 'State', 'ICSE', 'Other'], n, p=[0.40, 0.35, 0.15, 0.10])

base_score = np.random.normal(50, 25, n)
base_score = np.clip(base_score, 0, 100)

tenth_marks = np.clip(base_score * 0.7 + np.random.normal(20, 10, n), 35, 100).round(1)
twelfth_marks = np.clip(base_score * 0.65 + np.random.normal(18, 12, n), 35, 100).round(1)
cgpa = np.clip(base_score * 0.08 + np.random.normal(2, 0.8, n), 3.0, 10.0).round(2)

internships = np.random.choice([0, 1], n, p=[0.30, 0.70])
trainings = np.random.choice([0, 1], n, p=[0.28, 0.72])
backlogs = np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.25, 0.08, 0.02])
projects = np.random.choice([0, 1], n, p=[0.25, 0.75])

communication = np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.18, 0.32, 0.28, 0.14])
technical = np.random.choice(['No', 'Yes'], n, p=[0.22, 0.78])


coding_score = np.clip(base_score * 0.8 + np.random.normal(5, 12, n), 0, 100).round().astype(int)
aptitude_score = np.clip(base_score * 0.75 + np.random.normal(8, 10, n), 0, 100).round().astype(int)
hackathons_count = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.32, 0.26, 0.22, 0.13, 0.05, 0.02])
certifications_count = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.30, 0.28, 0.24, 0.13, 0.04, 0.01])

# Composite score
stream_val = np.array([{
    'Computer Science': 10, 'Information Technology': 9, 'Electronics': 7,
    'Mechanical': 5, 'Civil': 4, 'Electrical': 6, 'Other': 5
}[s] for s in stream])
gender_val = np.array([1 if g == 'Male' else (0 if g == 'Female' else 2) for g in gender])
tech_val = np.where(np.array(technical) == 'Yes', 1, 0)

composite = (
    gender_val * 0.5 +
    stream_val * 2.5 +
    tenth_marks * 0.15 +
    twelfth_marks * 0.15 +
    cgpa * 6.0 +
    internships * 12.0 +
    trainings * 8.0 +
    (1 - np.minimum(backlogs, 1)) * 15.0 +
    projects * 10.0 +
    communication * 6.0 +
    tech_val * 12.0 +
    coding_score * 0.15 +
    aptitude_score * 0.13 +
    hackathons_count * 5.0 +
    certifications_count * 4.0 +
    np.random.normal(0, 3, n)
)

threshold = np.percentile(composite, 27)
placement_status = np.where(composite > threshold, 'Placed', 'Not Placed')

df = pd.DataFrame({
    'Gender': gender, '10th board': tenth_board, '10th marks': tenth_marks,
    '12th board': twelfth_board, '12th marks': twelfth_marks, 'Stream': stream,
    'Cgpa': cgpa,
    'Internships(Y/N)': np.where(internships > 0, 'Yes', 'No'),
    'Training(Y/N)': np.where(trainings > 0, 'Yes', 'No'),
    'Any Backlogs?': np.where(backlogs > 0, 'Yes', 'No'),
    'Innovative Project(Y/N)': np.where(projects > 0, 'Yes', 'No'),
    'Communication level': communication,
    'Technical skills(Y/N)': technical,
    'Coding Score': coding_score,
    'Aptitude Score': aptitude_score,
    'Hackathons Count': hackathons_count,
    'Certifications Count': certifications_count,
    'Placement(Y/N)?': placement_status
})

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Placement rate: {(df['Placement(Y/N)?'] == 'Placed').mean() * 100:.1f}%")

df.to_csv('Dataset/Placements_data.csv', index=False)
print("✓ Dataset saved to Dataset/Placements_data.csv")

# ─── END OF DATASET GENERATION ───

# Load dataset
df = pd.read_csv('Dataset/Placements_data.csv')
print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Preprocessing
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

binary_cols = ['Gender', 'Internships(Y/N)', 'Training(Y/N)',
               'Any Backlogs?', 'Innovative Project(Y/N)', 'Technical skills(Y/N)']
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_le = LabelEncoder()
df['Placement(Y/N)?'] = target_le.fit_transform(df['Placement(Y/N)?'])

multi_cols = ['10th board', '12th board', 'Stream']
ohe_categories = {col: sorted(df[col].unique().tolist()) for col in multi_cols}
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Placement(Y/N)?', axis=1)
y = df_encoded['Placement(Y/N)?']
feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\n" + "-" * 50)
print("Training Random Forest...")
print("-" * 50)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

print(f"  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"  ROC-AUC:       {auc:.4f}")
print(f"  CV Accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols)
top10 = importances.sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances:")
for feat, imp in top10.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

# Save confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Placed', 'Placed'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Random Forest v2.0\nAccuracy: {acc*100:.1f}%', fontsize=13)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Save artifacts
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
joblib.dump(ohe_categories, 'models/ohe_categories.pkl')
joblib.dump({
    'rf_acc': round(acc * 100, 1),
    'rf_auc': round(auc, 4),
    'rf_cv': round(cv_scores.mean() * 100, 1),
    'dataset_size': len(df),
    'placed_rate': round((df['Placement(Y/N)?'] == 1).mean() * 100, 1),
    'top_features': top10.reset_index().rename(columns={'index': 'feature', 0: 'importance'}).to_dict(orient='records')
}, 'models/model_metrics.pkl')

print("\n✓ All artifacts saved!")
print("✓ Training complete!")