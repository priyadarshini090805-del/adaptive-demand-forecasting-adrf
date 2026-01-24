import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load and prepare data
# -----------------------------
data = pd.read_csv("data.csv", encoding="ISO-8859-1")
data = data.dropna()
data = data[data["Quantity"] > 0]
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

# Select one popular product
top_product = data["StockCode"].value_counts().idxmax()
product_data = data[data["StockCode"] == top_product]

# Daily aggregation
product_data["Date"] = product_data["InvoiceDate"].dt.date
daily = product_data.groupby("Date")["Quantity"].sum().reset_index()
daily["Date"] = pd.to_datetime(daily["Date"])
daily = daily.sort_values("Date")

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler()
scaled_qty = scaler.fit_transform(daily[["Quantity"]])

# -----------------------------
# Create sequences
# -----------------------------
WINDOW = 21  # longer memory = more advanced

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_qty, WINDOW)

# -----------------------------
# Train-test split (time-based)
# -----------------------------
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Build ADVANCED LSTM model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# -----------------------------
# Evaluate on test data
# -----------------------------
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
print("Advanced LSTM Product-level MAE:", round(mae, 2))

# -----------------------------
# Future forecasting (next 30 days)
# -----------------------------
future_steps = 30
last_window = scaled_qty[-WINDOW:]

future_predictions = []

for _ in range(future_steps):
    pred = model.predict(last_window.reshape(1, WINDOW, 1))
    future_predictions.append(pred[0][0])
    last_window = np.append(last_window[1:], pred, axis=0)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

future_dates = pd.date_range(
    start=daily["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=future_steps
)

# -----------------------------
# Plot results
# -----------------------------
plt.figure()
plt.plot(daily["Date"], daily["Quantity"], label="Actual Demand")
plt.plot(future_dates, future_predictions, label="LSTM Forecast")
plt.title("Advanced LSTM Demand Forecast (Product Level)")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.show()
# -----------------------------
# BUSINESS PARAMETERS
# -----------------------------
current_stock = 1500     # units currently in warehouse
lead_time_days = 7       # supplier lead time
safety_factor = 0.20     # 20% safety buffer
# Expected demand during lead time
expected_lead_demand = future_predictions[:lead_time_days].sum()

# Safety stock
safety_stock = expected_lead_demand * safety_factor

# Reorder point
reorder_point = expected_lead_demand + safety_stock
if current_stock < reorder_point:
    reorder_qty = reorder_point - current_stock
else:
    reorder_qty = 0
print("\n--- BUSINESS INSIGHTS ---")
print("Expected demand (next", lead_time_days, "days):", round(expected_lead_demand, 2))
print("Safety stock:", round(safety_stock, 2))
print("Reorder point:", round(reorder_point, 2))
print("Current stock:", current_stock)

if reorder_qty > 0:
    print("⚠️ STOCK ALERT: Reorder required")
    print("Recommended reorder quantity:", round(reorder_qty))
else:
    print("✅ Stock level is sufficient. No reorder needed.")
# Overstock check (30-day horizon)
monthly_demand = future_predictions.sum()

if current_stock > monthly_demand * 1.5:
    print("⚠️ OVERSTOCK ALERT: Inventory exceeds expected 30-day demand")
# -----------------------------
# DASHBOARD TABLES
# -----------------------------
forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Demand": future_predictions.flatten()
})

print("\n--- FORECAST TABLE (Next 10 Days) ---")
print(forecast_table.head(10))
inventory_table = pd.DataFrame({
    "Metric": [
        "Current Stock",
        "Expected Demand (Lead Time)",
        "Safety Stock",
        "Reorder Point",
        "Recommended Reorder Qty"
    ],
    "Value": [
        current_stock,
        round(expected_lead_demand, 2),
        round(safety_stock, 2),
        round(reorder_point, 2),
        round(reorder_qty, 2)
    ]
})

print("\n--- INVENTORY DECISION SUMMARY ---")
print(inventory_table)
# -----------------------------
# DASHBOARD VISUALIZATION
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(daily["Date"], daily["Quantity"], label="Historical Demand")
plt.plot(future_dates, future_predictions, label="Forecasted Demand")

plt.axhline(
    y=reorder_point,
    linestyle="--",
    label="Reorder Point"
)

plt.title("Demand Forecasting Dashboard")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.show()
low_stock_days = forecast_table[
    forecast_table["Forecasted_Demand"] > (current_stock / lead_time_days)
]

if not low_stock_days.empty:
    print("\n⚠️ High-demand days detected:")
    print(low_stock_days.head())
