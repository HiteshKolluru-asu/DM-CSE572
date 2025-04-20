import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

train_df = pd.read_csv("data/train_df.csv")


train_df.fillna(0, inplace=True)

train_df['CPI'] = pd.to_numeric(train_df['CPI'], errors='coerce')
train_df['Unemployment'] = pd.to_numeric(train_df['Unemployment'], errors='coerce')
train_df_full = train_df.copy()

X = train_df.drop(columns=['Weekly_Sales'])
y = train_df['Weekly_Sales']

features = ['Store', 'Dept', 'IsHoliday', 'week_number', 'Normalized_Size']
X = train_df[features]
y = train_df['Weekly_Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

print(X_train.info())

print(f"Training Set Size: {X_train.shape[0]} rows")
print(f"Testing Set Size: {X_test.shape[0]} rows")
print("Feature Names:", X_train.columns.tolist())
print("Target Variable Name: Weekly_Sales")
print("First few rows of the training set:")
print(X_train.head())


dt_model = DecisionTreeRegressor(random_state=101)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree Regressor Mean Squared Error:", mse_dt)
print("Decision Tree Regressor R-squared:", r2_dt)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5, label='Decision Tree Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('Decision Tree: Actual vs Predicted Weekly Sales')
plt.legend()
plt.show()

rf_model = RandomForestRegressor(n_estimators=100, random_state=101)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)
print("Random Forest Regressor Mean Squared Error:", mse_rf)
print("Random Forest Regressor R-squared:", r2_rf)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.show()

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=101)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)
print("XGBoost Regressor Mean Squared Error:", xgb_mse)
print("XGBoost Regressor R-squared:", xgb_r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label='XGBoost Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('XGBoost: Actual vs Predicted Weekly Sales')
plt.legend()
plt.show()

# ---------------------------
# Model Building: LightGBM Regressor
# ---------------------------
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=101)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

lgbm_mse = mean_squared_error(y_test, y_pred_lgbm)
lgbm_r2 = r2_score(y_test, y_pred_lgbm)
print("LightGBM Regressor Mean Squared Error:", lgbm_mse)
print("LightGBM Regressor R-squared:", lgbm_r2)

# Visualization: Actual vs. Predicted for LightGBM
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.5, label='LightGBM Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('LightGBM: Actual vs Predicted Weekly Sales')
plt.legend()
plt.show()

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM'],
    'MSE': [mse_rf, mse_dt, xgb_mse, lgbm_mse],
    'R-squared': [r2_rf, r2_dt, xgb_r2, lgbm_r2]
})

print("\nModel Performance Summary:")
print(results_df)


dt_model = DecisionTreeRegressor(
    max_depth=10,             # limiting depth -> avoid overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=101
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree (Tuned) MSE:", mse_dt)
print("Decision Tree (Tuned) R²:", r2_dt)


rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=101,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest (Tuned) MSE:", mse_rf)
print("Random Forest (Tuned) R²:", r2_rf)

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=101,
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost (Tuned) MSE:", mse_xgb)
print("XGBoost (Tuned) R²:", r2_xgb)

lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=101,
    n_jobs=-1
)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
print("LightGBM (Tuned) MSE:", mse_lgbm)
print("LightGBM (Tuned) R²:", r2_lgbm)

results_df = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'Decision Tree (Tuned)', 'XGBoost (Tuned)', 'LightGBM (Tuned)'],
    'MSE': [mse_rf, mse_dt, mse_xgb, mse_lgbm],
    'R-squared': [r2_rf, r2_dt, r2_xgb, r2_lgbm]
})
print("\nModel Performance Summary (Tuned):")
print(results_df)
