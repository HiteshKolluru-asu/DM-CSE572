import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


train_df = pd.read_csv("data/train_df.csv")
train_df = train_df.sort_values('Date')

sales_data = train_df.groupby('Date')['Weekly_Sales'].sum().reset_index()

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales_data[['Weekly_Sales']])

def create_sequences(data, seq_length=4):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 4
X, y = create_sequences(sales_scaled, sequence_length)

X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

y_test_flat = y_test_inv.flatten()
y_pred_flat = y_pred_inv.flatten()

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

r2 = r2_score(y_test_flat, y_pred_flat)

print(f"LSTM RMSE: {rmse:.2f}")
print(f"LSTM R² Score: {r2:.4f}")

plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='Actual Sales')
plt.plot(y_pred_inv, label='Predicted Sales', linestyle='--')
plt.title('LSTM Forecasting: Weekly Sales')
plt.xlabel('Weeks')
plt.ylabel('Sales')
plt.legend()
plt.show()
