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
print("  CAMPUS PLACEMENT PREDICTOR — MODEL TRAINING")
print("=" * 60)

os.makedirs('models', exist_ok=True)
os.makedirs('Dataset', exist_ok=True)

# Load dataset from CSV
df = pd.read_csv('Dataset/Placements_data.csv')
print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Preprocessing
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Label Encoding for binary columns
binary_cols = ['Gender', 'Technical skills(Y/N)']
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# Target encoding
target_le = LabelEncoder()
df['Placement(Y/N)?'] = target_le.fit_transform(df['Placement(Y/N)?'])
print(f"  Target classes: {list(target_le.classes_)}")

# One-hot encoding for multi-class categorical columns
multi_cols = ['10th board', '12th board', 'Stream']
ohe_categories = {col: sorted(df[col].unique().tolist()) for col in multi_cols}
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Features and target
X = df_encoded.drop('Placement(Y/N)?', axis=1)
y = df_encoded['Placement(Y/N)?']
feature_cols = list(X.columns)

print(f"\nFinal features ({len(feature_cols)}):")
for i, f in enumerate(feature_cols):
    print(f"  {i+1:2d}. {f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── Train Optimized Random Forest ───
print("\n" + "-" * 50)
print("Training Optimized Random Forest...")
print("-" * 50)

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    max_features='sqrt'
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

print(f"\n  Test Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"  ROC-AUC:        {auc:.4f}")
print(f"  CV Accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols)
top10 = importances.sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances:")
for feat, imp in top10.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

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
    'top_features': top10.reset_index().rename(
        columns={'index': 'feature', 0: 'importance'}
    ).to_dict(orient='records')
}, 'models/model_metrics.pkl')

print("\n✓ All artifacts saved to models/")
print("✓ Training complete!")