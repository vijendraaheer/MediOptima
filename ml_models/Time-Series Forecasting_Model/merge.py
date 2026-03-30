import pandas as pd
import numpy as np

df1 = pd.read_csv("ml_models\Time-Series Forecasting Model/prediced_patients.csv")
df2 = pd.read_csv("ml_models\Time-Series Forecasting Model/Expected_discharge.csv")
df3 = pd.read_csv("ml_models\Time-Series Forecasting Model/Predicted_Emergency_Cases.csv")
df4 = pd.read_csv("ml_models\Time-Series Forecasting Model/Predicted_ICU_Admissions.csv")

df = pd.DataFrame({"Date": df1['Date'],
                   "Predicted_Patients": np.ceil(df1['Predicted_Patients']),
                   "Expected_discharge": np.ceil(df2['Expected_discharge']),
                   "Predicted_Emergency_Cases": np.ceil(df3['Predicted_Emergency_Cases']),
                   "Predicted_ICU_Admissions": np.ceil(df4['Predicted_ICU_Admissions'])})

df.to_csv("data\outputs\Prediction.csv", index=False)
print("successfully...")