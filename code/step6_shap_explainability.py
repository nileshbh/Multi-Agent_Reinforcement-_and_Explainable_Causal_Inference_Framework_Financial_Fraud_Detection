# step6_shap_explainability_final.py
import pandas as pd, numpy as np, shap, time, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ---------- 1. LOAD ENGINEERED DATA ----------
df = pd.read_csv("engineered_data.csv")
features = ['Amount_log','AvgAmount_t','Freq_t','Lag_Time']
X, y = df[features], df['Class']
print(f"\n Original dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ---------- 2. SMOTE BALANCING ----------
sm = SMOTE(random_state=42, sampling_strategy=0.2)
X_res, y_res = sm.fit_resample(X, y)
print("Class balance after SMOTE:\n", y_res.value_counts())

# ---------- 3. TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
)

# ---------- 4. MODEL TRAINING ----------
model = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
start = time.time()
model.fit(X_train, y_train)
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)[:,1]
latency = time.time() - start

# ---------- 5. PERFORMANCE METRICS ----------
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
pr, rc, _ = precision_recall_curve(y_test, pred_proba)
pr_auc = auc(rc, pr)
print(f"\n Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | PR-AUC={pr_auc:.4f}")
print(classification_report(y_test, pred))

# ---------- 6. CONFUSION MATRIX ----------
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title("Confusion Matrix", fontsize=12)
plt.xlabel("Predicted Label"); plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=300)
plt.show()

# ---------- 7. CROSS-VALIDATION ACCURACY (simulate accuracy/loss curves) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')
plt.figure(figsize=(6,3))
plt.plot(range(1,6), cv_scores, marker='o', color="teal")
plt.title("Cross-Validation Accuracy per Fold", fontsize=12)
plt.xlabel("Fold Number"); plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Accuracy_CV.png", dpi=300)
plt.show()
print(f"CV Mean Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------- 8. SHAP EXPLAINABILITY (FAST SUBSET + IMPROVED LAYOUT) ----------
sample_X = X_test.sample(2000, random_state=42)
explainer = shap.Explainer(model, sample_X)
shap_values = explainer(sample_X)
print(" SHAP values computed ")

# --- Plot 1: Feature Importance (Bar) ---
plt.figure(figsize=(7,4))
shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=12, pad=15)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("SHAP_Feature_Importance.png", dpi=300)
plt.close()

# --- Plot 2: Feature Impact (Summary) ---
plt.figure(figsize=(7,4))
shap.summary_plot(shap_values, sample_X, show=False)
plt.title("SHAP Summary Plot – Feature Impact on Fraud Prediction", fontsize=12, pad=15)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("SHAP_Feature_Impact.png", dpi=300)
plt.close()

# ---------- 9. SAVE TRAINED MODEL FOR KAFKA ----------
joblib.dump(model, "fraud_detection_model.pkl")
print("\n Model saved successfully → fraud_detection_model.pkl")

# ---------- 10. FINAL METRIC SUMMARY ----------
print("\n Final Evaluation Summary")
print(f"Latency: {latency:.2f}s")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")
print(f"Cross-Val Accuracy: {cv_scores.mean():.3f}")

print("\nPlots saved as:")
print("  • Confusion_Matrix.png")
print("  • Accuracy_CV.png")
print("  • SHAP_Feature_Importance.png")
print("  • SHAP_Feature_Impact.png")
print(" step6_shap_explainability_final.py completed successfully.")
