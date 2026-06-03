# 🔧Predictive Maintenance for MechArms - Mechanical Robotic Parts for the Aviation Industry (Robotic Parts)

## 🚀 Overview

This project presents a predictive maintenance prototype for Robotic parts inspired from TempleAllen Insutries Airpressure operated Mechanical Arm Robotic parts, leveraging machine learning to detect early signs of equipment degradation and recommend proactive servicing.

The goal is to move from reactive maintenance to intelligent, data-driven maintenance, improving reliability and enabling a scalable service model.

---

## 🎯 Objective

- Predict potential maintenance needs using operational signals  
- Reduce unexpected downtime  
- Improve lifecycle performance of EMMA & SAM units  
- Lay foundation for a SmartCare subscription service  

---

## 📂 Project Workflow

### 1. Data Simulation

Since real sensor data is unavailable, the model uses synthetic operational data simulating:

- Temperature  
- Vibration  
- Pressure  
- Usage Hours  

---

### 2. Data Processing

- Cleaned and structured input features  
- Engineered additional features:
  - Vibration trends  
  - Temperature trends  
  - Lag features  
  - Usage-based indicators  
- Generated labeled dataset (`needs_maintenance`)  

---

### 3. Model Development

- Model: **Random Forest Classifier**  
- Train-test split for validation  
- Addressed class imbalance for realistic predictions  
- Learned patterns indicating abnormal machine behavior  

---

### 4. Prediction Output

Each unit is classified as:

- ✅ Monitor (Working Fine)  
- ⚠️ Schedule Maintenance  
- 🚨 Immediate Maintenance  

---

## 📊 Model Performance

- **Accuracy:** ~77%  
- **ROC-AUC Score:** ~0.79  

### Confusion Matrix



### Key Insight

- Balanced detection of failures vs false alarms  
- Slightly higher false positives are acceptable to avoid critical breakdowns  

---

## 📈 Feature Importance

Top contributing features:

- Vibration  
- Vibration Trend  
- Usage Hours  
- Temperature  
- Pressure  

👉 These align with real-world mechanical degradation patterns  

---

## 📉 ROC Curve

![ROC Curve]<img width="1800" height="500" alt="Figure_5" src="https://github.com/user-attachments/assets/77b7ef44-de5a-4a53-8b6c-31895d6f49b9" />


- Demonstrates good model separability  
- Confirms reliable predictive capability  

---

## 🧠 Explainability (SHAP)

![SHAP Summary]<img width="800" height="550" alt="Figure_4" src="https://github.com/user-attachments/assets/73d2033e-0c2a-4368-a54d-86f02eba26e4" />


SHAP values were used to interpret model predictions.

### Key Observations:

- High vibration significantly increases failure probability  
- Trend-based features are strong indicators of degradation  
- Usage intensity contributes to wear and failure  

👉 Ensures model decisions are transparent and trustworthy  

---

## 💻 Core Code Snippet

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
