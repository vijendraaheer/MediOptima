import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
df = pd.read_csv("data/processed/clean_hospital_data.csv")  # Make sure it has "Date" and "Emergency Cases"
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ==============================
# FEATURE ENGINEERING
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Lag features
for lag in [1,3,7,14,21,30]:
    df[f"lag{lag}"] = df["Emergency Cases"].shift(lag)

# Rolling features
for window in [3,7,14,30]:
    df[f"rolling{window}"] = df["Emergency Cases"].rolling(window).mean()

df = df.dropna()

# Prepare dataset for Prophet
data = df[[
    "Date","Emergency Cases","Patients","Discharge Count",
    "day_of_week","month","week_of_year",
    "lag1","lag3","lag7","lag14","lag21","lag30",
    "rolling3","rolling7","rolling14","rolling30",
    "weekend"
]]
data.columns = [
    "ds","y",
    "day_of_week","month","week_of_year","Patients","Discharge Count",
    "lag1","lag3","lag7","lag14","lag21","lag30",
    "rolling3","rolling7","rolling14","rolling30",
    "weekend"
]

# ==============================
# TRAIN-TEST SPLIT
train_size = int(len(data)*0.8)
train = data[:train_size]
test = data[train_size:]

# ==============================
# PROPHET MODEL
model = Prophet(
    changepoint_prior_scale=0.2,
    seasonality_prior_scale=25,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="multiplicative"
)
for col in [
    "day_of_week","month","week_of_year","Patients","Discharge Count",
    "lag1","lag3","lag7","lag14","lag21","lag30",
    "rolling3","rolling7","rolling14","rolling30",
    "weekend"
]:
    model.add_regressor(col)

model.fit(train)

future = model.make_future_dataframe(periods=len(test))
future = future.merge(
    data[[
        "ds","day_of_week","month","week_of_year","Patients","Discharge Count",
        "lag1","lag3","lag7","lag14","lag21","lag30",
        "rolling3","rolling7","rolling14","rolling30",
        "weekend"
    ]], on="ds", how="left").ffill()

prophet_pred = model.predict(future)["yhat"].tail(len(test)).values

# ==============================
# LSTM MODEL
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Emergency Cases"]])
sequence = 14

X, y = [], []
for i in range(sequence, len(scaled)):
    X.append(scaled[i-sequence:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

lstm_pred = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred).flatten()

# ==============================
# HYBRID MODEL
min_len = min(len(prophet_pred), len(lstm_pred))
prophet_pred = prophet_pred[:min_len]
lstm_pred = lstm_pred[:min_len]
actual = test["y"].values[:min_len]

hybrid_pred = 0.4*prophet_pred + 0.6*lstm_pred
hybrid_pred = np.maximum(hybrid_pred, 0)  # Clip negatives

mae = mean_absolute_error(actual, hybrid_pred)
rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
mape = np.mean(np.abs((actual - hybrid_pred)/actual))*100

print("\nHYBRID MODEL PERFORMANCE")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("MAPE:", round(mape,2), "%")

# ==============================
# Load Patients and Discharge CSVs
patient_data = pd.read_csv("ml_models\Time-Series Forecasting Model/prediced_patients.csv")
discharge_data = pd.read_csv("ml_models\Time-Series Forecasting Model/Expected_discharge.csv")  # ✅ Correct CSV

patient_data["Date"] = pd.to_datetime(patient_data["Date"])
discharge_data["Date"] = pd.to_datetime(discharge_data["Date"])
discharge_data.rename(columns={"Expected_discharge": "Discharge Count"}, inplace=True)
patient_data.rename(columns={"Predicted_Patients": "Patients"}, inplace=True)

# Merge on Date
merged_df = pd.merge(patient_data, discharge_data, on="Date", how="inner")

# Prepare future dataframe
future_df = merged_df.rename(columns={"Date":"ds"})
future_df["day_of_week"] = future_df["ds"].dt.dayofweek
future_df["month"] = future_df["ds"].dt.month
future_df["week_of_year"] = future_df["ds"].dt.isocalendar().week.astype(int)
future_df["weekend"] = future_df["day_of_week"].apply(lambda x: 1 if x>=5 else 0)

last_row = data.iloc[-1]
for lag in [1,3,7,14,21,30]:
    future_df[f"lag{lag}"] = last_row[f"lag{lag}"]
for window in [3,7,14,30]:
    future_df[f"rolling{window}"] = last_row[f"rolling{window}"]

# Predict Emergency Cases
forecast_jan = model.predict(future_df)
df_forecast = forecast_jan[["ds","yhat"]]
df_forecast.columns = ["Date","Predicted_Emergency_Cases"]
df_forecast["Predicted_Emergency_Cases"] = df_forecast["Predicted_Emergency_Cases"].clip(lower=0)

df_forecast.to_csv("ml_models\Time-Series Forecasting Model/Predicted_Emergency_Cases.csv", index=False)
print("\nJanuary 2025 Emergency Cases Forecast")
print(df_forecast.head())

# ==============================
# PLOT
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual Emergency Cases")
plt.plot(prophet_pred, label="Prophet")
plt.plot(lstm_pred, label="LSTM")
plt.plot(hybrid_pred, label="Hybrid", linewidth=3)
plt.legend()
plt.title("Hybrid Emergency Cases Forecast")
plt.show()