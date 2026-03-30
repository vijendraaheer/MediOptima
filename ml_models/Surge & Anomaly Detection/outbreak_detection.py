import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# Ensure folder exists
os.makedirs("Surge & Anomaly Detection", exist_ok=True)

# Load data
df = pd.read_csv("data/outputs/Prediction.csv")

# Features to detect anomalies
features = df[['Predicted_Patients', 'Predicted_Emergency_Cases', 'Predicted_ICU_Admissions', 'Expected_discharge']]

# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Train Isolation Forest
model = IsolationForest(contamination=0.07, random_state=42)
df['Outbreak'] = model.fit_predict(scaled)

# Convert to readable alert
df['Outbreak_Alert'] = df['Outbreak'].apply(lambda x: "YES" if x == -1 else "NO")

# Save results
df.to_csv("ml_models/Surge & Anomaly Detection/outbreak_alert.csv", index=False)

print(" Outbreak detection completed. CSV saved at 'Surge & Anomaly Detection/outbreak_alert.csv'")