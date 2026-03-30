import pandas as pd 
import numpy as np 

# Load data
df = pd.read_csv('data/outputs/Prediction.csv')

# Calculate Z-score for emergency cases
mean = df['Predicted_Emergency_Cases'].mean()
std = df['Predicted_Emergency_Cases'].std()

df["Z_score"] = (df['Predicted_Emergency_Cases']- mean)/ std

# Flag surge
df['Surge_Alert'] = df['Z_score'].apply(
    lambda x: "YES" if abs(x) > 2.5 else "NO"
    )

df.to_csv("ml_models/Surge & Anomaly Detection\surge_alert.csv", index=False)

print("Surge detection completed")
