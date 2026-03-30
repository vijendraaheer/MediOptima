import numpy as np
import pandas as pd
import holidays
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/hospital_25_year_dataset.csv")

# Change datatypes
df['Date'] = pd.to_datetime(df['Date'])

# maping in holiday
df['Public Holidays'] = df['Public Holidays'].map({'Yes': 1, 'No': 0})

# Check missing values and remove
print(f'Missing values : {df.isnull().sum()}')

# check duplicates and fill
print(f'Duplicate values : {df.duplicated().sum()}')


# check outliers
Q1 = df['Patients'].quantile(0.25)
Q3 = df['Patients'].quantile(0.75)
IQR = Q3 - Q1

Upper = Q3 + 1.5 * IQR
Lower = Q1 - 1.5 * IQR
outlier = (df['Patients'] > Upper) | (df['Patients'] < Lower)
print(f'sum of outliers : {outlier.sum()}')

print(df.head())
# print(df.info())
# print(df.describe())

df.to_csv("data\processed\clean_hospital_data.csv", index=False)
print("successfully......")