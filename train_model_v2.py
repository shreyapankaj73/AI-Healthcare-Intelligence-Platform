"""
IOCL Employee Health Risk Prediction - Model Training (with SMOTE)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

np.random.seed(42)
N = 3000

def generate_dataset(n=N):
    data = {}
    data['age'] = np.random.randint(22, 60, n)
    data['bmi'] = np.round(np.random.normal(25.5, 4.5, n).clip(16, 42), 1)
    data['systolic_bp'] = np.random.normal(122, 15, n).clip(90, 200).astype(int)
    data['diastolic_bp'] = np.random.normal(80, 10, n).clip(60, 120).astype(int)
    data['heart_rate'] = np.random.normal(78, 12, n).clip(55, 130).astype(int)
    data['spo2'] = np.round(np.random.normal(97.5, 1.5, n).clip(88, 100), 1)
    data['cholesterol'] = np.random.normal(190, 38, n).clip(120, 320).astype(int)
    data['blood_glucose'] = np.random.normal(105, 28, n).clip(70, 300).astype(int)
    data['years_of_service'] = np.random.randint(1, 38, n)
    data['heat_exposure_hrs'] = np.round(np.random.uniform(0, 10, n), 1)
    data['chemical_exposure_score'] = np.random.randint(0, 10, n)
    data['noise_exposure_db'] = np.random.normal(72, 18, n).clip(40, 130).astype(int)
    data['night_shifts_per_month'] = np.random.randint(0, 16, n)
    data['physical_demand_score'] = np.random.randint(1, 10, n)
    data['stress_score'] = np.random.randint(1, 10, n)
    data['smoking'] = np.random.choice([0, 1], n, p=[0.68, 0.32])
    data['alcohol_use'] = np.random.choice([0, 1], n, p=[0.72, 0.28])
    data['exercise_days_per_week'] = np.random.randint(0, 7, n)
    data['sleep_hours'] = np.round(np.random.normal(6.5, 1.2, n).clip(3, 10), 1)

    df = pd.DataFrame(data)

    risk_score = np.zeros(n)
    risk_score += (df['age'] > 45) * 2.5
    risk_score += (df['bmi'] > 30) * 2
    risk_score += (df['bmi'] > 35) * 2
    risk_score += (df['systolic_bp'] > 140) * 2.5
    risk_score += (df['systolic_bp'] > 160) * 2
    risk_score += (df['diastolic_bp'] > 90) * 1.5
    risk_score += (df['cholesterol'] > 240) * 2
    risk_score += (df['blood_glucose'] > 126) * 2.5
    risk_score += (df['spo2'] < 94) * 3
    risk_score += (df['heart_rate'] > 100) * 1.5
    risk_score += (df['heat_exposure_hrs'] > 6) * 2
    risk_score += (df['chemical_exposure_score'] > 6) * 2
    risk_score += (df['noise_exposure_db'] > 85) * 1.5
    risk_score += (df['night_shifts_per_month'] > 10) * 1.5
    risk_score += (df['physical_demand_score'] > 7) * 1.5
    risk_score += (df['stress_score'] > 7) * 2
    risk_score += (df['years_of_service'] > 20) * 1
    risk_score += df['smoking'] * 3
    risk_score += df['alcohol_use'] * 1.5
    risk_score += (df['exercise_days_per_week'] < 2) * 1.5
    risk_score += (df['sleep_hours'] < 5.5) * 2
    risk_score += np.random.normal(0, 1.5, n)

    df['risk_level'] = pd.cut(risk_score,
                               bins=[-np.inf, 9, 16, np.inf],
                               labels=['Low', 'Medium', 'High'])
    return df

df = generate_dataset()
print("Dataset shape:", df.shape)
print("\nRisk Level Distribution:")
print(df['risk_level'].value_counts())

X = df.drop('risk_level', axis=1)
y = df['risk_level']

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Apply SMOTE to training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE - Training samples: {X_train_res.shape[0]}")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(model, 'model_artifacts/health_risk_model.pkl')
joblib.dump(scaler, 'model_artifacts/scaler.pkl')
joblib.dump(le, 'model_artifacts/label_encoder.pkl')
joblib.dump(list(X.columns), 'model_artifacts/feature_names.pkl')
df.to_csv('model_artifacts/training_data.csv', index=False)

print("\n✅ Model saved successfully!")
print(f"Classes: {le.classes_}")
