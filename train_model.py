import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


print("=" * 55)
print("        PLACEMENT PREDICTOR — MODEL TRAINING")
print("=" * 55)

# Load dataset
# Auto-detect CSV file in Dataset folder
import glob
csv_files = glob.glob('Dataset/*.csv')
if not csv_files:
    raise FileNotFoundError("No CSV file found in 'Dataset/' folder. Please place your dataset there.")
dataset_path = csv_files[0]  # Use the first CSV found
print(f"Using dataset: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop non-predictive columns
df = df.drop(['Email', 'Name'], axis=1)

# Check and fill nulls
print("Null values per column:")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.any() else "  → No nulls found")

for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Target distribution:")
print(df['Placement(Y/N)?'].value_counts())
placed_pct = (df['Placement(Y/N)?'] == 'Placed').mean() * 100
print(f"  → Placement rate: {placed_pct:.1f}%")

# Encode binary categorical columns
# Auto-detect binary columns (handles slight naming variations)
possible_binary_cols = [
    'Gender', 'Internships(Y/N)', 'Training(Y/N)',
    'Any Backlogs?', 'Innovative Project(Y/N)',
    'Technical Course(Y/N)', 'Technical skills(Y/N)', 'Technical Skills(Y/N)'
]
binary_feature_cols = [col for col in possible_binary_cols if col in df.columns]
print(f"\nDetected binary columns: {binary_feature_cols}")
# Auto-detect target column
possible_targets = ['Placement(Y/N)?', 'Placement', 'Placed', 'Status']
target_col = None
for pt in possible_targets:
    if pt in df.columns:
        target_col = pt
        break
if target_col is None:
    raise ValueError(f"Could not find target column. Available columns: {list(df.columns)}")
print(f"Target column: {target_col}")

label_encoders = {}
for col in binary_feature_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode target
target_le = LabelEncoder()
df[target_col] = target_le.fit_transform(df[target_col])
print(f"Target encoding: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

# One-hot encode multi-category columns
multi_cols = ['10th board', '12th board', 'Stream']

ohe_categories = {}
for col in multi_cols:
    ohe_categories[col] = sorted(df[col].unique().tolist())

df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print(f"After encoding: {df_encoded.shape[1]} total features")

# Split features and target
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

feature_cols = list(X.columns)
print(f"Features used for training ({len(feature_cols)}):")
for f in feature_cols:
    print(f"   • {f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("" + "-" * 40)
print("Random Forest")
print("-" * 40)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"  Test Accuracy   : {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"  ROC-AUC         : {rf_auc:.4f}")
print(f"  CV Accuracy     : {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(classification_report(y_test, rf_pred, target_names=['Not Placed', 'Placed']))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols)
top10 = importances.sort_values(ascending=False).head(10)

print("-" * 40)
print("Top 10 Feature Importances (Random Forest):")
for feat, imp in top10.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

# Save confusion matrix plot
os.makedirs('models', exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Placed', 'Placed'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Random Forest\nAccuracy: {accuracy_score(y_test, rf_pred)*100:.1f}%', fontsize=13)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved to models/confusion_matrix.png")

# Save all artifacts
joblib.dump(rf,             'models/random_forest.pkl')
joblib.dump(scaler,         'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols,   'models/feature_columns.pkl')
joblib.dump(ohe_categories, 'models/ohe_categories.pkl')
joblib.dump(
    {
        'rf_acc': round(rf_acc * 100, 1),
        'rf_auc': round(rf_auc, 4),
        'rf_cv': round(rf_cv_scores.mean() * 100, 1),
        'dataset_size': len(df),
        'placed_rate': round(placed_pct, 1),
        'top_features': top10.reset_index().rename(
            columns={'index': 'feature', 0: 'importance'}
        ).to_dict(orient='records')
    },
    'models/model_metrics.pkl'
)

print("All artifacts saved in 'models/' folder:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}')
    print(f"   {f:<35} {size:>8} bytes")

print("Training complete!")