import os
from pathlib import Path

import pandas as pd
import twint

from src import config as cf


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


datetime_list = pd.date_range(start="2021-04-02", end="2021-04-29")
date_list = []
for index, date in enumerate(datetime_list):
    time1 = str(date).split(" ")
    time2 = str(date + pd.DateOffset(1)).split(" ")
    date_list.append((time1[0], time2[0]))

config = twint.Config()
config.Pandas = True
config.Hide_output = True
for ticker in cf.tickers:
    config.Search = "$" + ticker
    for date in date_list:
        config.Since = date[0]
        config.Until = date[1]
        twint.run.Search(config)
        df = twint.storage.panda.Tweets_df

        tweets = df["tweet"]
        cashtags = df["cashtags"]
        dates = df["date"]
        likes = df["nlikes"]

        size = len(likes.index)
        id_list = generate_ids(ticker, size)

        frame = {'tweet': tweets, 'date': dates, 'likes': likes, 'tickers': cashtags, 'ID': id_list}
        data_frame = pd.DataFrame(frame)
        path = "data/Twitter/scraped/raw/" + date[0]
        os.makedirs(path, exist_ok=True)
        data_frame.to_csv(path + "/" + ticker + ".csv", index=False, encoding="utf-8")

        for index, list in enumerate(cashtags):
            if len(list) > 2:
                data_frame = data_frame.drop(index=index)
            elif len(list) == 0:
                data_frame = data_frame.drop(index=index)

        path = "data/Twitter/scraped/filtered/" + date[0]
        os.makedirs(path, exist_ok=True)
        data_frame.to_csv(path + "/" + ticker + ".csv", index=False, encoding="utf-8")
        print(ticker + ":" + date[0] + " completed\n")
