import os
import re
from pathlib import Path

import pandas as pd


def read_twitter_id_from_file(ticker):
    path = "var/twitter-id-" + ticker + ".txt"
    file = Path(path)
    if file.is_file():
        file = open(path, mode="r")
        twitter_id = int(file.readline())
        file.close()
    else:
        twitter_id = 0
    return twitter_id


def generate_ids(ticker, size):
    current_id = read_twitter_id_from_file(ticker)
    id_list = []
    for i in range(size):
        id_list.append(current_id)
        current_id += 1
    save_twitter_id(current_id, ticker)
    id_list.reverse()
    return id_list


def save_twitter_id(current_id, ticker):
    file = open("var/twitter-id-" + ticker + ".txt", mode="w+")
    file.write(str(current_id))
    file.close()


def rename_files():
    raw_path = "data/Twitter/scraped/filtered"
    for date in os.listdir(raw_path):
        for csv in os.listdir(raw_path + "/" + date):
            csv_path = raw_path + "/" + date + "/" + csv
            result = re.sub("\$", "", csv_path)
            if result != csv_path:
                os.rename(csv_path, result)


def get_file_list(ticker):
    raw_path = "data/Twitter/scraped/filtered"
    file_list = []
    for date in os.listdir(raw_path):
        for csv in os.listdir(raw_path + "/" + date):
            if csv.split(".")[0] == ticker:
                csv_path = raw_path + "/" + date + "/" + csv
                file_list.append(csv_path)
    return file_list


rename_files()

for ticker in ["TSLA", "AMZN", "FB", "MSFT", "AAPL", "AMD", "NVDA"]:
    file_list = get_file_list(ticker)
    for file in file_list:
        df = pd.read_csv(filepath_or_buffer=file)
        id_list = generate_ids(ticker, len(df.index))
        df["ID"] = id_list
        df.to_csv(file, index=False)
