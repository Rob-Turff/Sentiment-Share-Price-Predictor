import os
import random
from datetime import datetime

import pandas as pd

from src.DataProcessing import pre_processing

NUM_POSTS = 50000
# NUM_POSTS = 250000
current_ticker = "AMZN"
DATA_PATH = "data/StockTwits"
PRICE_PATH = "data/Hourly-Stock-Prices"
SAVE_PATH = "data/TrainingData/stocktwits_tagged_sentiment" + current_ticker + ".csv"
TESTING_SAVE_PATH = "data/TrainingData/testing_stocktwits_tagged_sentiment" + current_ticker + ".csv"
date_format = "%Y-%m-%d"
columns = ["Twit", "Date", "Sentiment", "ID", "User", "User Likes", "User Posts"]
test_set = True

if test_set:
    save_path = TESTING_SAVE_PATH
else:
    save_path = SAVE_PATH


def get_latest_date():
    latest_date = datetime.now()
    for date in os.listdir(PRICE_PATH):
        file_date = datetime.strptime(date, date_format)
        if file_date < latest_date:
            latest_date = file_date
    return latest_date


def parse_file(file):
    temp = file.split("/")
    ticker = temp[len(temp) - 1].split(".")[0]
    df = pd.read_csv(filepath_or_buffer=file, names=columns)
    df = df[df.Sentiment.notnull()]
    df = df.drop(["Date", "ID", "User", "User Likes", "User Posts"], 1)
    df.loc[df["Sentiment"] == "Bearish", "Sentiment"] = 0
    df.loc[df["Sentiment"] == "Bullish", "Sentiment"] = 1
    for index, row in df.iterrows():
        new_text = pre_processing.stocktwit_cleanup(row["Twit"], ticker)
        if new_text is not None:
            df.loc[index]["Twit"] = new_text
        else:
            df = df.drop(axis=0, index=index)
    df = df.reset_index(drop=True)
    return (df, len(df.index))


def delete_old_data():
    if os.path.exists(save_path):
        print("Deleting old data")
        os.remove(save_path)


latest_date = get_latest_date()
# Creates list of all stocktwits data files
file_list = []
for date in os.listdir(DATA_PATH):
    file_date = datetime.strptime(date, date_format)
    if test_set:
        if file_date > latest_date:
            for csv in os.listdir(DATA_PATH + "/" + date):
                if csv.split(".")[0] == current_ticker:
                    csv_path = DATA_PATH + "/" + date + "/" + csv
                    file_list.append(csv_path)
    else:
        if file_date < latest_date:
            for csv in os.listdir(DATA_PATH + "/" + date):
                if csv.split(".")[0] == current_ticker:
                    csv_path = DATA_PATH + "/" + date + "/" + csv
                    file_list.append(csv_path)

training_data = pd.DataFrame(columns=columns)

cur_len = 0

delete_old_data()
for i in range(len(file_list)):
    ran_int = random.randint(0, len(file_list))
    df, size = parse_file(file_list.pop(ran_int - 1))
    print(len(file_list))
    if (cur_len + size) >= NUM_POSTS:
        df = df.truncate(after=(NUM_POSTS - cur_len) - 1)
        df.to_csv(save_path, mode="a", index=False, header=False, encoding="utf-8")
        cur_len += len(df.index)
        break
    else:
        cur_len += size
        df.to_csv(save_path, mode="a", index=False, header=False, encoding="utf-8")

print("Wrote " + str(cur_len) + " posts to the file!")
