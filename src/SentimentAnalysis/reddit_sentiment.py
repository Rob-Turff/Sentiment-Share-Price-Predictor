import os
import re

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/reddit-sorted"

file_list = []
date_list = []
for date in os.listdir(DATA_PATH)[10:]:
    for csv in os.listdir(DATA_PATH + "/" + date):
        date_list.append(date + ":" + csv.split(".")[0])
        csv_path = DATA_PATH + "/" + date + "/" + csv
        file_list.append(csv_path)

ticker_market_cap = {"GME": 10, "NOK": 22, "TSLA": 543, "PLTR": 40, "BB": 5, "NIO": 54}
ticker_data_raw = {}
ticker_data_weighted = {}
for ticker in ticker_market_cap:
    ticker_data_raw[ticker] = []
    ticker_data_weighted[ticker] = []

for file_num, path in enumerate(file_list):
    for key in ticker_market_cap:
        ticker_data_raw[key].append(0)
        ticker_data_weighted[key].append(0)
    df = pd.read_csv(filepath_or_buffer=path)
    df = df[df.Body.notnull()]
    text = df["Body"].tolist()
    for index, comment in enumerate(text):
        matches = re.findall("(?!\s)\$?([A-Za-z]{2,4})(?=\s)", comment)
        for match in matches:
            if match in ticker_market_cap:
                ticker_data_raw[match][file_num] += (1 * df["Score"].tolist()[index])
                ticker_data_weighted[match][file_num] += (1 * df["Score"].tolist()[index]) / ticker_market_cap[match]

plt.figure(figsize=(20, 10))
plt.title('Mentions vs Hour')
plt.ylabel('Mentions')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.xticks(rotation="vertical")

for ticker in ticker_market_cap:
    if ticker != "PLACEHOLDER":
        plt.plot(date_list, ticker_data_raw[ticker], label=ticker)
plt.legend(loc="upper left")
plt.show()
