import pandas as pd
import numpy as np

df1 = pd.read_csv("ml_models/Surge & Anomaly Detection/surge_alert.csv")
df2 = pd.read_csv("ml_models/Surge & Anomaly Detection/outbreak_alert.csv")

df = pd.DataFrame({"Date": df1['Date'],
                   "Predicted_Patients": np.ceil(df1['Predicted_Patients']),
                   "Z_score": np.ceil(df1['Z_score']),
                   "Surge_Alert": df1['Surge_Alert'],
                   "Outbreak": np.ceil(df2['Outbreak']),
                   "Outbreak_Alert": df2['Outbreak_Alert']})

df.to_csv("data\outputs\Surge_Outbreak_Alerts.csv", index=False)
print("successfully...")