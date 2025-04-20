import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


train_df = pd.read_csv("data/train_df.csv")

features = ['Store', 'Dept', 'IsHoliday', 'week_number', 'Normalized_Size']
target = 'Weekly_Sales'

scaler_y = MinMaxScaler()
scaler_X = MinMaxScaler()

X_scaled = scaler_X.fit_transform(train_df[features])
y_scaled = scaler_y.fit_transform(train_df[[target]])

data_combined = np.hstack((y_scaled, X_scaled))  # y + features

def create_sequences(data, seq_length=8):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 1:])  # only features
        y.append(data[i+seq_length, 0])     # target is the first column
    return np.array(X), np.array(y)

seq_len = 8
X, y = create_sequences(data_combined, seq_len)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

y_pred = model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"\nMultivariate LSTM RMSE: {rmse:.2f}")
print(f"Multivariate LSTM R² Score: {r2:.4f}")

plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='Actual Sales')
plt.plot(y_pred_inv, label='Predicted Sales', linestyle='--')
plt.title('Multivariate LSTM: Actual vs Predicted Weekly Sales')
plt.xlabel('Time')
plt.ylabel('Weekly Sales')
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()