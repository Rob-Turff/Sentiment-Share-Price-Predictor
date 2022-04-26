import os
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pytz

from src.DataProcessing import pre_processing
from src.SentimentAnalysis.roberta_config import RobertaConfig
from src.helper_functions import format_time, get_logger

columns = ["Twit", "Date", "Sentiment", "ID", "User", "User Likes", "User Posts"]
logger = get_logger(LOG_NAME="roberta_data")


class RobertaData:
    def __init__(self, config: RobertaConfig):
        self.config = config
        self.current_twitter_id = None

    def _time_operation(self, t0, i, total_len, operation_name):
        if i % 1000 == 0 and not i == 0:
            elapsed = format_time(time.time() - t0)
            eta = format_time(((time.time() - t0) / i) * (total_len - i))
            logger.info("%s - index %d of %d ----- Elapsed time: %s ETA: %s", operation_name, i, total_len, elapsed,
                        eta)

    '''
    Timezone conversions from UTC (stocktwits and twitter posts) to US eastern time (stock market)
    '''

    def _convert_date(self, date, is_stocktwits):
        if is_stocktwits:
            intraday_date_format = "%Y-%m-%d %H:%M"
        else:
            intraday_date_format = "%Y-%m-%d %H:%M:%S"
        date_time = datetime.strptime(date, intraday_date_format)
        us_timezone = pytz.timezone("US/Eastern")
        utc_timezone = pytz.timezone("UTC")

        localized_timestamp = utc_timezone.localize(date_time)
        date_time = localized_timestamp.astimezone(us_timezone)

        if date_time.minute < 30:
            time = (date_time.hour - 9) * 2 - 1
        else:
            time = (date_time.hour - 9) * 2
        return date_time.weekday(), time, str(date_time.date())

    '''
    Gets the dates the stock market is open from daily price data
    '''

    def _get_market_dates(self):
        df = pd.read_csv(filepath_or_buffer=self.config.price_path, header=None)
        date_list = df.iloc[:, 6].tolist()
        return date_list

    '''
    Gets first and last post id's from dataframe
    '''

    def _get_bounding_ids(self, df):
        row = df.tail(1)
        latest_id = row["last_id"][0]

        row = df.head(1)
        earliest_id = row["first_id"][0]
        return latest_id, earliest_id

    '''
    Gets first/last date from sentiment file
    '''

    def _get_date(self, file_path, get_tail, date_format):
        df = pd.read_csv(file_path, index_col=[0, 1, 2])
        df = df.dropna()
        if get_tail:
            date = df.tail(1).index[0][0]
        else:
            date = df.head(1).index[0][0]
        date = datetime.strptime(date, date_format)

        return date

    def _get_lower_limit_date(self):
        date_format = "%Y-%m-%d"
        if self.config.generate_daily_data:
            df = pd.read_csv(self.config.get_price_path())
            date = df.iloc[0][6]
            return datetime.strptime(date, date_format)
        else:
            return datetime.strptime("2020-09-25", date_format)

    '''
    Gets the dates that bound the currently stored sentiment data and the most recent date that stock price data exists for
    '''

    def _get_bounding_dates(self, is_stocktwits):
        date_format = "%Y-%m-%d"

        lower_limit = self._get_lower_limit_date()

        file_path_tagged = self.config.get_sentiment_file_path(is_stocktwits, is_tagged=True)
        file_path_untagged = self.config.get_sentiment_file_path(is_stocktwits, is_tagged=False)
        if os.path.exists(file_path_tagged) and os.path.exists(file_path_untagged):
            latest_date_tagged = self._get_date(file_path_tagged, True, date_format)
            latest_date_untagged = self._get_date(file_path_untagged, True, date_format)
            earliest_date_tagged = self._get_date(file_path_tagged, False, date_format)
            earliest_date_untagged = self._get_date(file_path_untagged, False, date_format)

            if latest_date_tagged < latest_date_untagged:
                latest_date = latest_date_tagged
            else:
                latest_date = latest_date_untagged

            if earliest_date_tagged > earliest_date_untagged:
                earliest_date = earliest_date_tagged
            else:
                earliest_date = earliest_date_untagged
        else:
            latest_date = lower_limit
            earliest_date = datetime.now()

        logger.info("latest_date: %s earliest_date: %s lower_limit: %s", latest_date, earliest_date, lower_limit)
        return latest_date, earliest_date, lower_limit

    '''
    Parses twitter and stocktwits posts with pre-processing
    '''

    def _multi_core_parse(self, df, ticker, is_stocktwits, thread):
        if is_stocktwits:
            row_name = "Twit"
        else:
            row_name = "tweet"

        t0 = time.time()
        total_length = len(df.index)
        name = "parse t-" + str(thread) + " "
        df = df.reset_index(drop=True)
        print(name + "starting... length: " + str(total_length))
        for index, row in df.iterrows():
            if is_stocktwits:
                new_text = pre_processing.stocktwit_cleanup(row[row_name], ticker)
            else:
                new_text = pre_processing.twitter_cleanup(row[row_name])

            if new_text is not None:
                df.at[index, row_name] = new_text
            else:
                df = df.drop(axis=0, index=index)
        elapsed = format_time(time.time() - t0)
        print(name + "ending... took: " + elapsed)
        return df

    '''
    Starts multiple threads to pre-process twitter and stocktwit posts
    '''

    def _parse_file(self, df, ticker, is_stocktwits):
        if is_stocktwits:
            df.loc[df["Sentiment"] == "Bearish", "Sentiment"] = -1
            df.loc[df["Sentiment"] == "Bullish", "Sentiment"] = 1
        dataframes = np.array_split(df, 100)
        with Pool(processes=10) as pool:
            parsed_frames = pool.starmap(self._multi_core_parse,
                                         [(dataframes[i], ticker, is_stocktwits, i) for i in range(len(dataframes))])
        df = pd.concat(parsed_frames, ignore_index=True)
        df = df.reset_index(drop=True)
        return df

    '''
    Generates a list of files containing stocktwits or twitter posts within the bounding dates
    '''

    def _get_file_list(self, ticker, is_stocktwits):
        date_format = "%Y-%m-%d"
        if is_stocktwits:
            raw_path = self.config.raw_stocktwits_data_path
        else:
            raw_path = self.config.raw_twitter_data_path
        file_list = []
        latest_date, earliest_date, lower_limit = self._get_bounding_dates(is_stocktwits)
        for date in os.listdir(raw_path):
            file_date = datetime.strptime(date, date_format)
            if (lower_limit <= file_date <= earliest_date) or file_date >= latest_date:
                for csv in os.listdir(raw_path + "/" + date):
                    if csv.split(".")[0] == ticker:
                        csv_path = raw_path + "/" + date + "/" + csv
                        file_list.append(csv_path)
        return file_list

    def _get_post_score(self, row, is_stocktwits):
        if is_stocktwits:
            return (row["User Likes"] / (row["User Posts"] + 1)) * row["Sentiment"]
        else:
            return row["likes"] * row["Sentiment"]

    def _create_row_data(self, sentiment_df, index, row):
        for i in range(-19, 29):
            sentiment_df.loc[index[0], index[1], i] = 0, 0, 0, 0, row["ID"]
        return sentiment_df

    '''
    Stores the sentiment data in the relevant files
    '''

    def store_sentiments(self, df, is_tagged=True, is_stocktwits=True):
        file_path = self.config.get_sentiment_file_path(is_stocktwits, is_tagged)
        base_path = self.config.get_sentiment_base_path(is_stocktwits)
        csv_columns = ["date", "day", "interval", "raw_score", "posts", "weighted_score", "last_id", "first_id"]
        if is_stocktwits:
            date_row = "Date"
        else:
            date_row = "date"

        t0 = time.time()
        if os.path.exists(file_path):
            sentiment_df = pd.read_csv(file_path, index_col=[0, 1, 2])
            latest_id, earliest_id = self._get_bounding_ids(sentiment_df)
            df = df[(df["ID"] > latest_id) | (df["ID"] < earliest_id)]
        else:
            os.makedirs(base_path, exist_ok=True)
            sentiment_df = pd.DataFrame(columns=csv_columns)
            sentiment_df = sentiment_df.set_index(["date", "day", "interval"])

        total_length = len(df.index)
        trading_days = self._get_market_dates()
        for i, row in df.iterrows():
            self._time_operation(t0, i, total_length, "store sentiments")
            day, interval, date = self._convert_date(row[date_row], is_stocktwits)
            if date in trading_days:
                if (date, day, interval) in sentiment_df.index:
                    sentiment_df = self.add_row_to_df((date, day, interval), sentiment_df, row, is_stocktwits)
                else:
                    sentiment_df = self._create_row_data(sentiment_df, (date, day), row)
                    sentiment_df = self.add_row_to_df((date, day, interval), sentiment_df, row, is_stocktwits)
        sentiment_df = sentiment_df.sort_values(["date", "interval"])
        sentiment_df.to_csv(self.config.get_sentiment_file_path(is_stocktwits=is_stocktwits, is_tagged=is_tagged))

    def add_row_to_df(self, index, sentiment_df, row, is_stocktwits):
        sentiment_df.loc[index, "raw_score"] += row["Sentiment"]
        sentiment_df.loc[index, "weighted_score"] += self._get_post_score(row, is_stocktwits)
        sentiment_df.loc[index, "posts"] += 1
        if sentiment_df.loc[index, "last_id"] < row["ID"]:
            sentiment_df.loc[index, "last_id"] = row["ID"]
        if sentiment_df.loc[index, "first_id"] > row["ID"]:
            sentiment_df.loc[index, "first_id"] = row["ID"]
        return sentiment_df

    def get_ticker_stocktwits_posts(self, ticker):
        data_frames = []
        for file in self._get_file_list(ticker, is_stocktwits=True):
            data_frames.append(pd.read_csv(filepath_or_buffer=file, names=columns))
        df = pd.concat(data_frames, ignore_index=True)
        df = self._parse_file(df, ticker, True)
        df_tagged = df[df.Sentiment.notnull()]
        self.store_sentiments(df_tagged, is_tagged=True, is_stocktwits=True)
        df_to_tag = df[df.Sentiment.isnull()]
        return df_to_tag

    def get_ticker_twitter_posts(self, ticker):
        data_frames = []
        for file in self._get_file_list(ticker, is_stocktwits=False):
            data_frames.append(pd.read_csv(filepath_or_buffer=file))
        df = pd.concat(data_frames, ignore_index=True)
        df = self._parse_file(df, ticker, False)
        return df
