import os

import pandas as pd


def sort():
    DATA_PATH = "data/reddit"

    file_list = []
    for date in os.listdir(DATA_PATH):
        for csv in os.listdir(DATA_PATH + "/" + date):
            csv_path = DATA_PATH + "/" + date + "/" + csv
            file_list.append(csv_path)

    dict = {}
    for path in file_list:
        df = pd.read_csv(filepath_or_buffer=path)
        df = df[df.Body.notnull()]
        for index, row in df.iterrows():
            row_date = row["Date"].split(" ")
            new_date = row_date[0] + " " + row_date[1].split(":")[0]
            if new_date in dict:
                dict[new_date] = dict[new_date].append(row)
            else:
                dict[new_date] = pd.DataFrame(row).transpose()

    for key in dict:
        key_date = key.split(" ")
        path = "data/reddit-sorted" + "/" + key_date[0]
        os.makedirs(path, exist_ok=True)
        full_path = path + "/" + key_date[1] + ".csv"
        if os.path.isfile(full_path):
            df = pd.read_csv(filepath_or_buffer=full_path)
            new_df = df.append(dict[key], ignore_index=True)
            new_df.drop_duplicates(subset="ID", keep="first", inplace=True)
            new_df.to_csv(full_path, mode="w", index=False, header=True, encoding="utf-8")
        else:
            dict[key].drop_duplicates(subset="ID", keep="first", inplace=True)
            dict[key].to_csv(full_path, mode="w", index=False, header=True, encoding="utf-8")
        print(key + " written")

    for path in file_list:
        os.remove(path)
