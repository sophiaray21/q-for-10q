import yfinance as yf
import numpy as np
#ticker = yf.Ticker("AAPL")

#df = ticker.history(period="1mo")
#print(type(df))
#print(df[['Open', 'Close', 'Volume']].head())
#print(df.head())
#print(len(df))

def data_frame_with_ticker(period,ticker):
    ticker = yf.Ticker(ticker)
    df = ticker.history(period=period)
    return df[['Open', 'Close', 'Volume', 'Dividends']]

print(data_frame_with_ticker("1mo","AAPL"))