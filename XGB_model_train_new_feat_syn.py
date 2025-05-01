import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Calculates RSI (Relative Strength Index)
def compute_rsi(series, window=14):
    delta = series.diff() # Price changes
    gain = delta.clip(lower=0) # Gains (positive)
    loss = -delta.clip(upper=0) # Losses (negative)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # RSI Formula
    rs = avg_gain / (avg_loss + 1e-10)  # 1e-10 avoids division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Loads Synthetic BTC/USD data
df = pd.read_csv("synthetic_btc_2year.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime")
df.rename(columns={"Close": "price", "Volume": "volume"}, inplace=True)

# Feature engineering
df['price_t'] = df['price']                                 # Current Price
df['price_t-1'] = df['price'].shift(1)                      # Price -1 time step
df['price_t-2'] = df['price'].shift(2)                      # Price -2 time step
df['price_t-3'] = df['price'].shift(3)                      # Price -3 time step
df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()  # Exponential Moving Average (5 time steps)
df['momentum'] = df['price'] - df['price'].shift(3)         # Momentum over 3 time steps
df['volatility_5'] = df['price'].rolling(5).std()           # Volatility over 5 time steps (Standard Deviation)
df['volume_change'] = df['volume'].diff()                   # Change in volume compared to previous time step
df['price_diff'] = df['price'].diff()                       # First-order Price Difference
df['price_diff2'] = df['price'].diff().diff()               # Second-order Price Difference
df['rsi_14'] = compute_rsi(df['price'], window=14)          # RSI over 14 time steps
df['target_price'] = df['price'].shift(-1)                  # Target Price for model to predict - the next real price
df.dropna(inplace=True)                                     # Drops rows with NaN values

# Defines the features and the target
features = ['price_t', 'price_t-1', 'price_t-2', 'price_t-3', 'ema_5', 'momentum', 'volatility_5', 'volume_change',
            'price_diff', 'price_diff2', 'rsi_14' ]
X = df[features]
y = df['target_price']

# Splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Hyperparameter tuning using Randomised Search CV
param_dist = {
    'max_depth': [6, 8, 10],                # Depth of trees
    'learning_rate': [0.15, 0.2, 0.25],     # Learning rate
    'n_estimators': [600, 800, 1000],       # Number of boosting rounds
    'subsample': [0.6, 0.8, 1.0],           # Fraction of data used per tree
    'colsample_bytree': [0.6, 0.8, 1.0],    # Fraction of features used per tree
    'gamma': [0, 0.1, 0.3],                 # Minimum loss reduction to make a split
    'reg_alpha': [0, 0.1, 1, 5],            # L1 regularisation
    'reg_lambda': [0.1, 1, 10]              # L2 regularisation
}

# Initialises the Regression Model with objective of reducing MSE
model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Uses Randomised Search to tune hyperparameters
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,                          # tries 30 random combinations
    scoring='neg_mean_squared_error',   # Aims to minimise MSE
    cv=3,                               # Divides dataset into 3 equal parts
    verbose=1,                          # Print
    random_state=42,                    # Makes results reproducible
    n_jobs=-1                           # Uses all cores
)

# Fits the model
random_search.fit(X_train, y_train)

# Finds the best model for minimising MSE
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Best Model Performance:")
print("Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Test MSE: {mse:.4f}")

# Saves the best model
best_model.save_model("xgb_trader_tuned_new_syn.ubj")



