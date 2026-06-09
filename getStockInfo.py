import yfinance as yf
ticker = yf.Ticker("AAPL")

df = ticker.history(period="1mo")

print(df[['Open', 'Close', 'Volume']].head())
print(df.head())
print(len(df))