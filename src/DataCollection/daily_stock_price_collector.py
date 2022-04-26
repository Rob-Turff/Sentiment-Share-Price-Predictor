import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import config

path = "data/Daily-Stock-Prices/"
start_date = "2015-01-01"


def save_data(data, ticker, dates):
    os.makedirs(path, exist_ok=True)
    data.to_csv(path + ticker + ".csv", mode="a", index=False, header=False, encoding="utf-8")


def load_latest_date(ticker):
    file_path = Path(path + ticker + ".csv")
    if file_path.is_file():
        df = pd.read_csv(filepath_or_buffer=file_path, header=None, parse_dates=True)
        df = df.dropna()
        df.to_csv(file_path, mode="w", index=False, header=False, encoding="utf-8")
        df.iloc[:, 6] = pd.to_datetime(df.iloc[:, 6])
        date = df.tail(1).iloc[0, 6]
        date_time = date + timedelta(days=1)
        return date_time.strftime("%Y-%m-%d")
    else:
        return start_date


for ticker in config.tickers:
    start = load_latest_date(ticker)
    end = str(datetime.now().date())
    data = yf.download(tickers=ticker, start=start, end=end, interval="1d", threads=True)

    dates = []
    for d in data.axes[0]:
        dates.append(d.strftime("%Y-%m-%d"))

    data["Date"] = dates
    save_data(data, ticker, dates)
