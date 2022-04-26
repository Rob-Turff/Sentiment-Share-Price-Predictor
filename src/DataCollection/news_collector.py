import logging
import os
import time

import pandas as pd
import requests

stocks = {"TSLA": "tesla", "AMZN": "amazon", "FB": "facebook", "MSFT": "microsoft", "AAPL": "apple", "AMD": "amd",
          "NVDA": "nvidia"}
news_sites = ["forbes.com", "seekingalpha.com", "marketwatch.com", "reuters.com", "nasdaq.com", "bloomberg.com",
              "businessinsider.com", "yahoo.com", "investopedia.com", "smarteranalyst.com", "wsj.com"]
subscription_key = "[ADD KEY HERE]"
assert subscription_key

search_url = "https://api.bing.microsoft.com/v7.0/news/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}


def get_logger(
        LOG_FORMAT="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        LOG_NAME="NewsCollector",
        LOG_FILE_INFO="../../var/logs/NewsCollector.log",
        LOG_FILE_ERROR="../../var/logs/NewsCollector.err"):
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


def save_data(results, date, ticker):
    path = "../../data/NewsHeadlines/" + date
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path + "/" + ticker + ".csv", mode="a", index=False, header=False, encoding="utf-8")


logger = get_logger()

for ticker in stocks:
    logger.info("Running for ticker " + ticker)
    news_data = {}
    for site in news_sites:
        time.sleep(0.5)
        search_term = stocks[ticker] + " (site:" + site + ")"
        params = {"q": search_term, "textDecorations": False, "textFormat": "HTML", "count": 100, "cc": "US",
                  "freshness": "Week"}
        response = requests.get(search_url, headers=headers, params=params)
        if response.status_code != 200:
            logger.error("Raised response code " + str(response.status_code))
            logger.error(response)
        response.raise_for_status()
        search_results = response.json()
        for article in search_results["value"]:
            date = article["datePublished"].split("T")
            d = date[1].split(":")
            fin_date = date[0] + " " + d[0] + ":" + d[1]
            if date[0] in news_data.keys():
                news_data[date[0]]["name"].append(article["name"])
                news_data[date[0]]["date"].append(fin_date)
                news_data[date[0]]["website"].append(site)
                news_data[date[0]]["url"].append(article["url"])
                news_data[date[0]]["desc"].append(article["description"])
            else:
                line = {}
                line["name"] = [article["name"]]
                line["date"] = [fin_date]
                line["website"] = [site]
                line["url"] = [article["url"]]
                line["desc"] = [article["description"]]
                news_data[date[0]] = line

    for key in news_data:
        save_data(news_data[key], key, ticker)
