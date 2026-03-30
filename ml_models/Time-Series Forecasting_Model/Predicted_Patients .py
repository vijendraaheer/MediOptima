import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LOAD DATA
df = pd.read_csv("data/processed/clean_hospital_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")


# OUTLIER HANDLING USING SCALING
scaler_outlier = StandardScaler()
df["Patients_scaled"] = scaler_outlier.fit_transform(df[["Patients"]])

# FEATURE ENGINEERING
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

# Lag features
df["lag1"] = df["Patients"].shift(1)
df["lag3"] = df["Patients"].shift(3)
df["lag7"] = df["Patients"].shift(7)
df["lag14"] = df["Patients"].shift(14)
df["lag21"] = df["Patients"].shift(21)
df["lag30"] = df["Patients"].shift(30)


# Rolling features
df["rolling3"] = df["Patients"].rolling(3).mean()
df["rolling7"] = df["Patients"].rolling(7).mean()
df["rolling14"] = df["Patients"].rolling(14).mean()
df["rolling30"] = df["Patients"].rolling(30).mean()

# Weekend
df["weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

df = df.dropna()

# PREPARE DATA
data = df[[
"Date","Patients",
"day_of_week","month","week_of_year",
"lag1","lag3","lag7","lag14",'lag21','lag30',
"rolling7","rolling14","rolling3","rolling30",
"weekend"
]]

data.columns = [
"ds","y",
"day_of_week","month","week_of_year",
"lag1","lag3","lag7","lag14",'lag21','lag30',
"rolling7","rolling14","rolling3","rolling30",
"weekend"
]

# ==============================
# TRAIN TEST SPLIT
train_size = int(len(data)*0.8)
train = data[:train_size]
test = data[train_size:]

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
"day_of_week",
"month","week_of_year",
"lag1","lag3","lag7","lag14",'lag21','lag30',
"rolling7","rolling14","rolling3","rolling30",
"weekend"
]:
    model.add_regressor(col)

model.fit(train)

future = model.make_future_dataframe(periods=len(test))

future = future.merge(
data[[
"ds",
"day_of_week","month","week_of_year",
"lag1","lag3","lag7","lag14",'lag21','lag30',
"rolling7","rolling14","rolling3","rolling30",
"weekend"
]],
on="ds",
how="left"
)

future = future.ffill()

forecast = model.predict(future)

prophet_pred = forecast["yhat"].tail(len(test)).values

# LSTM MODEL
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Patients"]])
sequence = 14

X = []
y = []

for i in range(sequence, len(scaled)):
    X.append(scaled[i-sequence:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model_lstm = Sequential()

model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

lstm_pred = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
lstm_pred = lstm_pred.flatten()

# HYBRID MODEL
min_len = min(len(prophet_pred), len(lstm_pred))

prophet_pred = prophet_pred[:min_len]
lstm_pred = lstm_pred[:min_len]

actual = test["y"].values[:min_len]
hybrid_pred = (0.4 * prophet_pred) + (0.6 * lstm_pred)

# PERFORMANCE
mae = mean_absolute_error(actual, hybrid_pred)
rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
mape = np.mean(np.abs((actual - hybrid_pred) / actual)) * 100

print("\nHYBRID MODEL PERFORMANCE")

print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("MAPE:", round(mape,2), "%")

# GRAPH
plt.figure(figsize=(10,5))

plt.plot(actual, label="Actual Patients")
plt.plot(prophet_pred, label="Prophet")
plt.plot(lstm_pred, label="LSTM")
plt.plot(hybrid_pred, label="Hybrid", linewidth=3)
plt.legend()
plt.title("Hybrid Patient Forecast")
plt.show()


# FUTURE FORECAST

# Create future dates for January 2025
future_dates = pd.date_range(start= df["Date"].max(), end="2025-01-31")

future_df = pd.DataFrame({"ds": future_dates})

# Create required time features
future_df["day_of_week"] = future_df["ds"].dt.dayofweek
future_df["month"] = future_df["ds"].dt.month
future_df["week_of_year"] = future_df["ds"].dt.isocalendar().week.astype(int)

future_df["weekend"] = future_df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Use last known lag & rolling values
last_row = data.iloc[-1]

future_df["lag1"] = last_row["lag1"]
future_df["lag3"] = last_row["lag3"]
future_df["lag7"] = last_row["lag7"]
future_df["lag14"] = last_row["lag14"]
future_df["lag21"] = last_row["lag21"]
future_df["lag30"] = last_row["lag30"]

future_df["rolling3"] = last_row["rolling3"]
future_df["rolling7"] = last_row["rolling7"]
future_df["rolling14"] = last_row["rolling14"]
future_df["rolling30"] = last_row["rolling30"]

# Predict
forecast_jan2025 = model.predict(future_df)
df1 = forecast_jan2025[["ds","yhat"]]
df1.columns = ["Date","Predicted_Patients"]
df1.to_csv("ml_models\Time-Series Forecasting Model\prediced_patients.csv",index=False)

print("\nJanuary 2025 Patient Forecast")
