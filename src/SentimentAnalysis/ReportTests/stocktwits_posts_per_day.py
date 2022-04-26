import os
import pandas as pd
import src.config as cf
from datetime import datetime, timedelta

DATA_PATH = "data/StockTwits"
PRICE_PATH = "data/Hourly-Stock-Prices"
date_format = "%Y-%m-%d"
columns = ["Twit", "Date", "Sentiment", "ID", "User", "User Likes", "User Posts"]

latest_date = datetime.now() - timedelta(days=30)

file_list = {}
for current_ticker in cf.tickers:
    print("file list for " + current_ticker)
    file_list[current_ticker] = []
    for date in os.listdir(DATA_PATH):
        file_date = datetime.strptime(date, date_format)
        if file_date > latest_date:
            for csv in os.listdir(DATA_PATH + "/" + date):
                if csv.split(".")[0] == current_ticker:
                    csv_path = DATA_PATH + "/" + date + "/" + csv
                    file_list[current_ticker].append(csv_path)

posts_list = {}
for ticker in cf.tickers:
    print("post list for " + ticker)
    posts_list[ticker] = 0
    for file in file_list[ticker]:
        df = pd.read_csv(filepath_or_buffer=file, names=columns)
        posts_list[ticker] += len(df.index)
    posts_list[ticker] = posts_list[ticker] / len(file_list[ticker])

print(posts_list)