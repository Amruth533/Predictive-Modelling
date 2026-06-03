# predictive_maintenance_balanced.py

# ============================================
# IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ============================================
# STEP 1: GENERATE REALISTIC DATA
# ============================================
np.random.seed(42)
rows = 1000

time = np.arange(rows)

vibration = 1.0 + (time * 0.0008) + np.random.normal(0, 0.15, rows)
temperature = 40 + (time * 0.01) + np.random.normal(0, 5, rows)
pressure = 90 - (time * 0.005) + np.random.normal(0, 8, rows)
usage_hours = np.clip(time * 0.5 + np.random.randint(0, 50, rows), 0, 500)

df = pd.DataFrame({
    "vibration": vibration,
    "temperature": temperature,
    "pressure": pressure,
    "usage_hours": usage_hours
})


# ============================================
# STEP 2: HIDDEN FAILURE LOGIC (NO LEAKAGE)
# ============================================
true_risk = (
    0.4 * vibration +
    0.3 * (temperature / 50) +
    0.2 * (usage_hours / 500)
)

noise = np.random.normal(0, 0.2, rows)

failure_score = true_risk + noise

# Convert to probability
failure_prob = 1 / (1 + np.exp(-3 * failure_score))


# ============================================
# STEP 3: BALANCED LABEL CREATION
# ============================================
# Dynamic threshold → ensures realistic class balance
threshold_label = np.percentile(failure_prob, 70)

df["needs_maintenance"] = (failure_prob > threshold_label).astype(int)

print("\nClass Distribution:\n", df["needs_maintenance"].value_counts())


# ============================================
# STEP 4: FEATURE ENGINEERING
# ============================================
df["temp_vibration_ratio"] = df["temperature"] / df["vibration"]
df["high_usage_flag"] = (df["usage_hours"] > 350).astype(int)

df["vibration_lag1"] = df["vibration"].shift(1)
df["temperature_lag1"] = df["temperature"].shift(1)

df["vibration_trend"] = df["vibration"].rolling(5).mean()
df["temp_trend"] = df["temperature"].rolling(5).mean()

df = df.bfill().ffill()


# ============================================
# STEP 5: PREPARE DATA
# ============================================
X = df.drop("needs_maintenance", axis=1)
y = df["needs_maintenance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# STEP 6: TRAIN MODEL
# ============================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)


# ============================================
# STEP 7: PREDICTIONS WITH TUNING
# ============================================
y_prob = rf.predict_proba(X_test)[:, 1]

# Tune threshold for business needs
prediction_threshold = 0.4
y_pred = (y_prob > prediction_threshold).astype(int)


# ============================================
# STEP 8: EVALUATION
# ============================================
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))


# ============================================
# STEP 9: FEATURE IMPORTANCE
# ============================================
importances = pd.Series(rf.feature_importances_, index=X.columns)

print("\nFeature Importance:\n",
      importances.sort_values(ascending=False))


# ============================================
# STEP 10: BUSINESS OUTPUT
# ============================================
results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["Probability"] = y_prob

def get_action(prob):
    if prob > 0.75:
        return "Immediate Maintenance"
    elif prob > 0.5:
        return "Schedule Maintenance"
    else:
        return "Monitor"

results["Action"] = results["Probability"].apply(get_action)

print("\nSample Predictions:\n")
print(results.head(10))


# ============================================
# STEP 11: BUSINESS INSIGHT
# ============================================
print("\nBusiness Insight:")
print("Model balances detection of failures while minimizing unnecessary maintenance.")


# ============================================
# OPTIONAL: SHAP
# ============================================
try:
    import shap

    print("\nGenerating SHAP explanations...")

    explainer = shap.Explainer(rf, X_train)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values[:, :, 1])

except ImportError:
    print("\nInstall SHAP for explainability: pip install shap")

    #confusion matrix plot

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# Create subplots (ALL plots in one window)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            ax=axes[0])

axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ------------------------------------------
# 2. ROC Curve
# ------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
axes[1].plot([0,1], [0,1], linestyle="--")

axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# ------------------------------------------
# 3. Feature Importance
# ------------------------------------------
importances = rf.feature_importances_
features = X.columns

axes[2].barh(features, importances)
axes[2].set_title("Feature Importance")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.show()