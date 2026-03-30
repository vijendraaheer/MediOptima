import pandas as pd

# LOAD ALL MODULE OUTPUTS
old = pd.read_csv(r'D:\MY PROJECTS\MediOptima\data\processed\clean_hospital_data.csv')
pred = pd.read_csv(r'D:\MY PROJECTS\MediOptima\data\outputs\Prediction.csv')
bed = pd.read_csv(r'D:\MY PROJECTS\MediOptima\data\outputs\Bed_Requirement.csv')
staff = pd.read_csv(r'D:\MY PROJECTS\MediOptima\data\outputs\Optimized_Staff.csv')
risk = pd.read_csv(r'D:\MY PROJECTS\MediOptima\data\outputs\Surge_Outbreak_Alerts.csv')

# Convert Date columns
old['Date'] = pd.to_datetime(old['Date'])
for df in [pred, bed, staff, risk]:
    df['Date'] = pd.to_datetime(df['Date'])

# Merge all
df = pred.merge(bed, on="Date").merge(staff, on="Date").merge(risk, on="Date")

def generate_insight(old, df):

    TOTAL_BEDS = 500
    ICU_CAPACITY = 50

    last_week = old.tail(7)
    previous_week = old.tail(14).head(7)
    next_week = df.head(7)

    # ---------------- CALCULATIONS ----------------

    patient_growth = ((next_week['Predicted_Patients_x'].mean() - 
                      last_week['Patients'].mean()) 
                      / last_week['Patients'].mean()) * 100

    bed_occupancy = (next_week['Required_Beds'].mean() / TOTAL_BEDS) * 100
    icu_usage = (next_week['Predicted_ICU_Admissions'].max() / ICU_CAPACITY) * 100

    doctors_required = next_week['Doctors_Required'].mean()
    nurses_required = (
        next_week['General_Nurses_Required'].mean() +
        next_week['ICU_Nurses_Required'].mean()
    )

    # ---------------- DISCHARGE TREND ----------------

    last_week_discharge = last_week['Discharge Count'].mean()
    previous_week_discharge = previous_week['Discharge Count'].mean()

    discharge_growth = ((last_week_discharge - previous_week_discharge)
                        / previous_week_discharge) * 100

    # ---------------- INSIGHT BUILD ----------------

    insight = ""

    # Patient Growth
    if abs(patient_growth) < 1:
        insight += "🟢 Patient inflow stable.\n\n"
    elif patient_growth > 0:
        insight += f"🔺 Patient inflow increasing by {round(patient_growth,1)}%.\n\n"
    else:
        insight += f"🔻 Patient inflow decreasing by {abs(round(patient_growth,1))}%.\n\n"

    # Bed Risk
    if bed_occupancy > 100:
        bed_status = "🔴 Critical – Capacity exceeded!"
    elif bed_occupancy > 85:
        bed_status = "🔴 High Risk"
    elif bed_occupancy > 65:
        bed_status = "🟡 Moderate Load"
    else:
        bed_status = "🟢 Low Risk"

    insight += f"🛏 Bed Occupancy: {round(bed_occupancy,1)}% → {bed_status}\n\n"

    # ICU Risk
    if icu_usage > 100:
        icu_status = "🔴 Critical – ICU overflow expected!"
    elif icu_usage > 85:
        icu_status = "🔴 High ICU Risk"
    elif icu_usage > 65:
        icu_status = "🟡 Elevated Demand"
    else:
        icu_status = "🟢 Manageable"

    insight += f"🏥 ICU Usage: {round(icu_usage,1)}% → {icu_status}\n\n"

    # Discharge Trend
    if discharge_growth > 2:
        discharge_status = "🔺 Discharge rate improving."
    elif discharge_growth < -2:
        discharge_status = "🔻 Discharge rate declining."
    else:
        discharge_status = "🟢 Discharge rate stable."

    insight += f"🚑 Avg Daily Discharge: {round(last_week_discharge)} patients\n"
    insight += f"{discharge_status}\n\n"

    # Staff Recommendation
    if bed_occupancy >= 85 or icu_usage >= 85:
        staff_needed = "🔴 Additional staffing recommended."
    else:
        staff_needed = "🟢 Current staffing sufficient."

    insight += f"👨‍⚕ Doctors Required (avg): {round(doctors_required)}\n"
    insight += f"👩‍⚕ Nurses Required (avg): {round(nurses_required)}\n"
    insight += f"{staff_needed}\n\n"

    # Overall System Status
    overall_risk = max(bed_occupancy, icu_usage)

    if overall_risk > 90:
        overall_status = "🔴 SYSTEM UNDER STRAIN"
    elif overall_risk > 70:
        overall_status = "🟡 MODERATE PRESSURE"
    else:
        overall_status = "🟢 STABLE OPERATIONS"

    insight += f"⚠ Overall Operational Status: {overall_status}\n"

    return insight

print("\n📊 AI OPERATIONAL INTELLIGENCE REPORT\n")
print("========================================\n\n")
print(generate_insight(old, df))

# print(df.head())
# print(old.tail())
# print(df.info())
# print(old.info()