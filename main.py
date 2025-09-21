import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load two stock series
tickers = ["MSFT", "AAPL"]
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")["Close"]
s1, s2 = data[tickers[0]], data[tickers[1]]

# Test for cointegration
coint_t, p_value, _ = sm.tsa.coint(s1, s2)
spread = s1 - s2

# Z-score
z = (spread - spread.mean()) / spread.std()

# Strategy: enter when |z| > 2, exit when |z| < 0.5
longs = z < -2
shorts = z > 2
exits = abs(z) < 0.5

positions = np.where(longs, 1, np.where(shorts, -1, 0))
positions = pd.Series(positions, index=spread.index).ffill().where(~exits)

# Daily returns
returns = positions.shift(1) * (spread.diff() / spread.shift(1))
cum_returns = (1 + returns.fillna(0)).cumprod()

# Metrics
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
drawdown = (cum_returns / cum_returns.cummax() - 1).min()

print(f"Cointegration p-value: {p_value:.4f}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {drawdown:.2%}")

cum_returns.plot(title="StatArb Backtest")
plt.show()
