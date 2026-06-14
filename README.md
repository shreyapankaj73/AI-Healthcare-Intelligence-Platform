
# 🛢️ IOCL Employee Health Risk Prediction System

## Internship Project — Indian Oil Corporation Limited  
**Domain:** Healthcare AI / Occupational Health & Safety  
**Tech Stack:** Python · Scikit-learn · Streamlit · Plotly  

---

## 📌 Project Overview

This project predicts the **health risk level (Low / Medium / High)** of IOCL employees
based on their clinical vitals and occupational work conditions using a Machine Learning model.

**Problem Statement:**  
Petroleum industry workers face unique occupational health risks — chemical exposure, heat stress,
noise, irregular shifts. Early identification of at-risk employees allows preventive intervention.

---

## 🗂️ Project Structure

```
iocl_health/
├── app.py                  ← Streamlit web application
├── train_model_v2.py       ← ML model training script
├── requirements.txt        ← Python dependencies
├── README.md
└── model_artifacts/        ← Generated after training
    ├── health_risk_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_names.pkl
    └── training_data.csv
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model_v2.py
```

### 3. Launch the web app
```bash
streamlit run app.py
```

---

## 📊 Features Used (20 variables)

### Clinical Vitals
| Feature | Description |
|---|---|
| Age | Employee age (years) |
| BMI | Body Mass Index |
| Systolic / Diastolic BP | Blood pressure readings |
| Heart Rate | Resting heart rate (bpm) |
| SpO₂ | Blood oxygen saturation (%) |
| Cholesterol | Total cholesterol (mg/dL) |
| Blood Glucose | Fasting blood glucose (mg/dL) |

### Occupational Factors (IOCL-specific)
| Feature | Description |
|---|---|
| Heat Exposure (hrs/day) | Daily hours in high-heat zones |
| Chemical Exposure Score | 0–10 scale (H₂S, benzene, etc.) |
| Noise Exposure (dB) | Average noise levels |
| Night Shifts/Month | Irregular shift count |
| Physical Demand Score | Job physical intensity (1–10) |
| Stress Score | Psychological stress (1–10) |
| Years of Service | Total work experience |

### Lifestyle
| Feature | Description |
|---|---|
| Smoking | Binary (Yes/No) |
| Alcohol Use | Binary (Yes/No) |
| Exercise (days/week) | Physical activity frequency |
| Sleep Hours | Average sleep duration |

---

## 🤖 ML Model

- **Algorithm:** Random Forest Classifier (300 trees)
- **Class Imbalance Handling:** SMOTE oversampling
- **Accuracy:** ~79%
- **Dataset:** 3,000 synthetic employee records
- **Output Classes:** Low Risk · Medium Risk · High Risk

---

## 📈 Top Risk Factors (by importance)
1. BMI
2. Age  
3. Systolic Blood Pressure
4. Heat Exposure Hours
5. Noise Exposure
6. Sleep Hours
7. Smoking Status
8. Chemical Exposure Score
9. Blood Glucose
10. Diastolic BP

---

## ⚠️ Disclaimer
This system is built for educational/internship purposes using synthetic data.
All predictions must be reviewed by a qualified medical professional.

---

*Developed as part of IOCL Summer Internship 2026
=======
# AI-Healthcare-Intelligence-Platform
AI-powered occupational health monitoring platform that analyzes medical reports using Gemma 3 Vision, predicts worker health risks with Random Forest, and provides workforce analytics through an interactive Streamlit dashboard.

