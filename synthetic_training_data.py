import numpy as np
import pandas as pd

def generate_synthetic_btc(
    filename="synthetic_btc_2year.csv",
    start_price=70000, # Initial BTC price
    mu=0.00001, # Drift
    sigma=0.001, # Volatility
    lam=0.001, # Jump probability at each time step
    jump_mu=0.0001, # Size of jump (Average)
    jump_sigma=0.0003, # Volatility of jump
    n_days=730, # number of days of synthetic data - 2 years
    interval_minutes=5, # Time interval for price
    seed=42 # seed makes results reproducible
):
    np.random.seed(seed)

    # Calculates total time steps and dt
    steps_per_day = int((24 * 60) / interval_minutes)
    n_steps = steps_per_day * n_days
    dt = 1 / steps_per_day  # time increment (in days)

    prices = [start_price]
    times = pd.date_range(start="2023-01-01", periods=n_steps, freq=f"{interval_minutes}min")

    for _ in range(1, n_steps):
        S_t = prices[-1] # Most recent price

        # Geometric Brownian Motion Component - simulates the continuous fluctuations
        dW = np.random.normal(0, np.sqrt(dt))
        gbm = mu * dt + sigma * dW

        # Jump component - adds occasional sudden large jumps
        jump = np.random.normal(jump_mu, jump_sigma) if np.random.rand() < lam else 0

        dS = gbm + jump # Log change per time step
        new_price = S_t * np.exp(dS) # Next price - converts log and multiplies by last price
        prices.append(new_price)

    # Create DataFrame
    df = pd.DataFrame({
        "Datetime": times,
        "price": prices
    })

    # Volume is correlated with the magnitude of the price change
    price_diff = np.abs(np.diff([start_price] + prices))
    volume_noise = np.random.normal(50, 10, size=n_steps)
    df["volume"] = np.abs(price_diff + volume_noise).astype(int)

    # Saves synthetic data to CSV file
    df.to_csv(filename, index=False)
    print(f"Synthetic BTC data (2 year, 5min intervals) saved to: {filename}")

    return df

df = generate_synthetic_btc()