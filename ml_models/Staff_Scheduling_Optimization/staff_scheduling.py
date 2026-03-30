import pandas as pd
import pulp as lp
import math

# Load data
data = pd.read_csv("data/outputs/Bed_Requirement.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Monthly Salaries
doctor_monthly_salary = 150000
general_nurse_monthly_salary = 35000
icu_nurse_monthly_salary = 45000

# Convert to daily salary
doctor_cost = doctor_monthly_salary / 30
nurse_cost = general_nurse_monthly_salary / 30
icu_nurse_cost = icu_nurse_monthly_salary / 30

# Ratios
doctor_patient_ratio = 20
icu_nurse_ratio = 2
general_nurse_ratio = 6

# Shift logic
hospital_hours = 24
doctor_shift_hours = 12
nurse_shift_hours = 8

doctor_shifts = math.ceil(hospital_hours / doctor_shift_hours)
nurse_shifts = math.ceil(hospital_hours / nurse_shift_hours)

results = []

for index, row in data.iterrows():

    total_patients = row["Required_Beds"] + (0.2 * row["emergency_margin"])
    icu_beds = row["ICU_Beds"]
    general_beds = row["General_Ward_Beds"]

    # Define LP problem
    model = lp.LpProblem(f"Staff_Planning_{index}", lp.LpMinimize)

    # Decision variables (per shift)
    Doctors_shift = lp.LpVariable("Doctors_per_shift", lowBound=0, cat='Integer')
    General_Nurses_shift = lp.LpVariable("General_Nurses_per_shift", lowBound=0, cat='Integer')
    ICU_Nurses_shift = lp.LpVariable("ICU_Nurses_per_shift", lowBound=0, cat='Integer')

    # Cost calculation (objective)
    model += (doctor_cost * Doctors_shift * doctor_shifts +
              nurse_cost * General_Nurses_shift * nurse_shifts +
              icu_nurse_cost * ICU_Nurses_shift * nurse_shifts)

    # Constraints
    model += Doctors_shift >= total_patients / doctor_patient_ratio
    model += ICU_Nurses_shift >= icu_beds / icu_nurse_ratio
    model += General_Nurses_shift >= general_beds / general_nurse_ratio

    # Solve LP
    model.solve()

    # Compute totals
    total_doctors = Doctors_shift.varValue * doctor_shifts
    total_general_nurses = General_Nurses_shift.varValue * nurse_shifts
    total_icu_nurses = ICU_Nurses_shift.varValue * nurse_shifts

    results.append({
        "Date": row["Date"],

        "Doctors_Required": math.ceil(total_doctors),
        "General_Nurses_Required": math.ceil(total_general_nurses),
        "ICU_Nurses_Required": math.ceil(total_icu_nurses),

        "Doctor_Cost": round(doctor_cost * total_doctors, 2),
        "General_Nurse_Cost": round(nurse_cost * total_general_nurses, 2),
        "ICU_Nurse_Cost": round(icu_nurse_cost * total_icu_nurses, 2),

        "Total_Minimum_Daily_Cost": round(lp.value(model.objective), 2)
    })

# Output CSV
output = pd.DataFrame(results)
output.to_csv("data\outputs\Optimized_Staff.csv", index=False)

print(output.head())
print("Staff optimization completed ✔")