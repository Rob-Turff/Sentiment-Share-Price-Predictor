import os

import yfinance as yf

from src import config

start_date = "2021-04-20"
end_date = "2021-07-13"


def save_data(data, ticker, dates):
    last_date = date = dates[0].split(" ")[0]
    start_index = 0
    end_index = 0
    for index, row in enumerate(data.iterrows()):
        date = dates[index].split(" ")[0]
        if date == last_date:
            end_index = index + 1
        else:
            last_date, start_index = save_file(data, date, end_index, last_date, start_index, ticker)
    save_file(data, date, end_index, last_date, start_index, ticker)


def save_file(data, date, end_index, last_date, start_index, ticker):
    path = "data/Hourly-Stock-Prices/" + last_date
    os.makedirs(path, exist_ok=True)
    data_to_save = data.iloc[start_index:end_index]
    data_to_save.to_csv(path + "/" + ticker + ".csv", mode="w", index=False, header=False, encoding="utf-8")
    start_index = end_index
    last_date = date
    return last_date, start_index


for ticker in config.tickers:

    data = yf.download(tickers=ticker, start=start_date, end=end_date, interval="30m", threads=True)

    dates = []
    for d in data.axes[0]:
        dates.append(d.strftime("%Y-%m-%d %H:%M"))

    data["Date"] = dates
    save_data(data, ticker, dates)
