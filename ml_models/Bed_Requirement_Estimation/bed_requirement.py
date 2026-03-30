import pandas as pd  
import numpy as np

# -------------------- CONFIGURABLE PARAMETERS --------------------
EMERGENCY_MARGIN_PCT = 0.12    # Emergency buffer (12%)
ICU_RATIO = 0.15               # ICU beds as % of emergency_margin
MAX_ICU_BEDS = 50              # Maximum ICU beds
MAX_TOTAL_BEDS = 500           # Total hospital bed capacity

# -------------------- LOAD DATA --------------------
df = pd.read_csv("data/outputs/Prediction.csv")
df['Date'] = pd.to_datetime(df['Date'])  # Ensure date column is datetime

# -------------------- VALIDATE REQUIRED COLUMNS --------------------
required_cols = [
    'Date', 
    'Predicted_Patients', 
    'Expected_discharge', 
    'Predicted_Emergency_Cases', 
    'Predicted_ICU_Admissions'
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {missing_cols}")

# -------------------- BED CALCULATION --------------------
Required_Beds = pd.DataFrame()
Required_Beds['Date'] = df['Date']

# Step 1: Base required beds (Predicted - Discharged)
Required_Beds['Required_Beds'] = np.ceil((df['Predicted_Patients'] - df['Expected_discharge']).clip(lower=0))

# Step 2: Add predicted emergency cases
Required_Beds['Required_Beds'] += df['Predicted_Emergency_Cases']

# Step 3: Add emergency margin
Required_Beds['emergency_margin'] = np.ceil(Required_Beds['Required_Beds'] * (1 + EMERGENCY_MARGIN_PCT))

# Step 4: ICU allocation (max of predicted ICU or % of beds)
Required_Beds['ICU_Beds'] = np.ceil(np.maximum(df['Predicted_ICU_Admissions'], Required_Beds['emergency_margin'] * ICU_RATIO))
Required_Beds['ICU_Beds'] = Required_Beds['ICU_Beds'].clip(upper=MAX_ICU_BEDS)

# Step 5: General Ward beds
Required_Beds['General_Ward_Beds'] = (Required_Beds['emergency_margin'] - Required_Beds['ICU_Beds']).clip(lower=0)

# Step 6: Bed shortage alert
Required_Beds['Bed_Shortage_Alert'] = (Required_Beds['ICU_Beds'] + Required_Beds['General_Ward_Beds']) > MAX_TOTAL_BEDS

# Optional: Number of beds exceeding capacity
Required_Beds['Beds_Over_Capacity'] = ((Required_Beds['ICU_Beds'] + Required_Beds['General_Ward_Beds']) - MAX_TOTAL_BEDS).clip(lower=0)

# -------------------- LOGGING --------------------
print("Bed requirement calculation completed successfully!")
print("Max ICU Beds Needed:", Required_Beds['ICU_Beds'].max())
print("Max Total Beds Required:", (Required_Beds['ICU_Beds'] + Required_Beds['General_Ward_Beds']).max())
print("Days with Bed Shortage:", Required_Beds['Bed_Shortage_Alert'].sum())
print("Sample data:\n", Required_Beds.head())

# -------------------- SAVE TO CSV --------------------
Required_Beds.to_csv("data/outputs/Bed_Requirement.csv", index=False)  # CSV name same as original
print("Saved Bed_Requirement.csv successfully!")