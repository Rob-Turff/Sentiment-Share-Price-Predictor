import logging
import os
import sys

import pandas as pd
import requests

from src import config


def get_logger(
        LOG_FORMAT="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        LOG_NAME="StockTwitSaver",
        LOG_FILE_INFO="var/logs/StockTwitSaver.log",
        LOG_FILE_ERROR="var/logs/StockTwitSaver.err"):
    log = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='a')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_error = logging.FileHandler(LOG_FILE_ERROR, mode='a')
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.INFO)

    return log


logger = get_logger()


def save_data(results, date, ticker):
    path = "data/StockTwits/" + date
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path + "/" + ticker + ".csv", mode="a", index=False, header=False, encoding="utf-8")


def save_vars(max_id, since, var_file):
    file = open(var_file, mode="w+")
    file.write(str(max_id) + "\n" + str(since))
    file.close()


def check_vars(var_file, ticker):
    if not os.path.isfile(var_file):
        r = requests.get("https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?limit=1")
        json_data = r.json()
        start_since = json_data["cursor"]["since"]
        start_max = json_data["cursor"]["max"]
        save_vars(start_max, start_since, var_file)
        return start_max, start_since
    else:
        file = open(var_file, mode="r")
        max_id = file.readline()
        since = file.readline()
        file.close()
        return int(max_id), int(since)


def collect_data_loop(since, max_id, use_since, var_file, ticker):
    num_messages = 0
    while True:
        if use_since:
            api_string = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?since=" + str(since)
        else:
            api_string = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?max=" + str(max_id)

        r = requests.get(api_string)
        json_data = r.json()

        if json_data["response"]["status"] != 200:
            logger.info("Saved %d messages" % num_messages)
            if json_data["response"]["status"] == 429:
                logger.info("Rate limited exceeded for now")
                sys.exit(0)
            else:
                logger.error(json_data)
                sys.exit(1)
        elif json_data["cursor"]["since"] is None and json_data["cursor"]["max"] is None:
            break

        new_since = json_data["cursor"]["since"]
        if new_since > since:
            since = new_since

        new_max = json_data["cursor"]["max"]
        if new_max < max_id:
            max_id = new_max

        results = {"Twit": [], "Date": [], "Sentiment": [], "ID": [], "User": [], "User Likes": [], "User Posts": []}
        cur_date = None
        if use_since:
            msg_list = reversed(json_data["messages"])
        else:
            msg_list = json_data["messages"]
        for msg in msg_list:
            num_messages += 1
            date = msg["created_at"]
            d1 = date.split("T")
            d2 = d1[1].split(":")
            if cur_date is None:
                cur_date = d1[0]
            elif cur_date != d1[0]:
                save_data(results, cur_date, ticker)
                for key in results:
                    results[key].clear()

            results["Twit"].append(msg["body"])
            results["Date"].append(d1[0] + " " + d2[0] + ":" + d2[1])
            if msg["entities"]["sentiment"] is None:
                results["Sentiment"].append("N/A")
            else:
                results["Sentiment"].append(msg["entities"]["sentiment"]["basic"])
            results["ID"].append(msg["id"])
            results["User"].append(msg["user"]["username"])
            results["User Likes"].append(msg["user"]["like_count"])
            results["User Posts"].append(msg["user"]["ideas"])
        save_data(results, cur_date, ticker)
        save_vars(max_id, since, var_file)

        if not json_data["cursor"]["more"]:
            break


def start():
    for ticker in config.tickers:
        var_file = "var/stocktwitsAPI_" + ticker + ".txt"

        logger.info("Running for ticker " + ticker)

        max_id, since = check_vars(var_file, ticker)

        logger.info("Starting run with since: %s" % since)

        collect_data_loop(since, max_id, True, var_file, ticker)

    for ticker in config.tickers:
        var_file = "var/stocktwitsAPI_" + ticker + ".txt"

        logger.info("Running for ticker " + ticker)

        max_id, since = check_vars(var_file, ticker)

        logger.info("Starting run with max: %s" % max_id)

        collect_data_loop(since, max_id, False, var_file, ticker)


if __name__ == "__main__":
    try:
        start()
    except Exception as e:
        logger.exception("start crashed. Error %s", e)
