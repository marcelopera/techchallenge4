import yfinance as yf
import pandas as pd

tickers = 'AAPL MSFT NVDA AMZN META BRK-B GOOGL AVGO TSLA GOOG'
df = yf.download(tickers.split(), period="1d", group_by='ticker')

df = df.stack(level=0).reset_index()

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

df.to_parquet(
    'dados',
    partition_cols=['year', 'month', 'day'],
)