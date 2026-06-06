import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/placement_data.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# ── 2. EDA ────────────────────────────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

# Placement distribution
plt.figure(figsize=(5, 4))
df['Placed'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.title('Placed vs Not Placed')
plt.xticks([0, 1], ['Not Placed', 'Placed'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/placement_distribution.png', dpi=100)
plt.close()

# CGPA vs Placement
plt.figure(figsize=(6, 4))
sns.boxplot(x='Placed', y='CGPA', data=df, palette=['#e74c3c', '#2ecc71'])
plt.title('CGPA vs Placement')
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.tight_layout()
plt.savefig('plots/cgpa_vs_placement.png', dpi=100)
plt.close()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=100)
plt.close()

print("\nPlots saved.")

# ── 3. Prepare Features ───────────────────────────────────────────────────────
X = df.drop('Placed', axis=1)
y = df['Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train Models ───────────────────────────────────────────────────────────

# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"\nLogistic Regression Accuracy: {lr_acc*100:.2f}%")
print(f"Random Forest Accuracy:       {rf_acc*100:.2f}%")

# ── 5. Best Model Evaluation ──────────────────────────────────────────────────
best_model = rf if rf_acc >= lr_acc else lr
best_preds = rf_preds if rf_acc >= lr_acc else lr_preds
best_name  = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"

print(f"\nBest Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, best_preds, target_names=['Not Placed', 'Placed']))

# Confusion matrix plot
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=100)
plt.close()

# Feature importance (Random Forest)
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 4))
feat_imp.plot(kind='bar', color='#3498db', edgecolor='black')
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=100)
plt.close()

# ── 6. Save Model & Scaler ────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
pickle.dump(rf, open('model/rf_model.pkl', 'wb'))
pickle.dump(lr, open('model/lr_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
print("\nModels saved to /model/")
print("All plots saved to /plots/")
