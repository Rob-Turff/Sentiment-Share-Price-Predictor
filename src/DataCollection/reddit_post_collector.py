import logging
import os
from datetime import datetime

import pandas as pd
import praw

import src.DataProcessing.sort_reddit_by_hour as reddit_sort


def get_logger(
        LOG_FORMAT="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        LOG_NAME="reddit_post_collector",
        LOG_FILE_INFO="var/logs/reddit_post_collector.log",
        LOG_FILE_ERROR="var/logs/reddit_post_collector.err"):
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

reddit = praw.Reddit(
    client_id="uK8GixGo3kKyiw",
    client_secret="mvy1mHKGw5a4-9E1D763G8If_skBFA",
    user_agent="wsb-scraper:v0.0.1 (by u/-The_Law-)"
)

path = "data/reddit/2021-02-02"
os.makedirs(path, exist_ok=True)

wsb_subreddit = reddit.subreddit("wallstreetbets")
for post_num, post in enumerate(wsb_subreddit.hot(limit=25)):
    logger.info(post.title + " " + str(post.score))
    post.comments.replace_more(limit=None)
    data = {"Body": [], "Score": [], "Date": [], "ID": []}
    for count, comment in enumerate(post.comments):
        data["Body"].append(comment.body)
        data["Score"].append(comment.score)
        data["Date"].append((datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')))
        data["ID"].append(comment.id)
    df = pd.DataFrame(data)
    df.to_csv(path + "/wsb" + str(post_num) + ".csv", mode="w", index=False, header=True, encoding="utf-8")

reddit_sort.sort()
