import os
import sys
from datetime import datetime
from operator import truediv

import numpy as np
import pandas as pd
import torch
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

from src.DataProcessing import technical_indictors_calc
from src.StockPrediction.nnconfig import NNConfig
from src.helper_functions import get_logger


class PredictionData:
    def __init__(self, config):
        self.config: NNConfig = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._generate_sequences()
        self.create_test_train_val_sequence()
        self.logger = get_logger(LOG_NAME="prediction_data")

    '''
    Converts dates from standard time format to normalised values to be used in model
    '''

    def _convert_dates(self, date_list):
        converted_dates = []
        converted_times = []
        raw_dates = []
        for date in date_list:
            if self.config.daily_model:
                date_time = datetime.strptime(date, "%Y-%m-%d")
            else:
                date_time = datetime.strptime(date, "%Y-%m-%d %H:%M")
            if str(date_time.date()) not in raw_dates:
                raw_dates.append(str(date_time.date()))
            converted_dates.append(float(date_time.weekday() / 4))
            if date_time.minute == 30:
                time = (date_time.hour - 9) * 2
            else:
                time = (date_time.hour - 9) * 2 - 1
            converted_times.append(float(time / 12))
        return converted_dates, converted_times, raw_dates

    """
    Gets the final closing price of previous day, need for working out percentage change for initial value
    """

    def get_previous_close_price(self, last_date):
        date_format = "%Y-%m-%d"
        price_path = "data/Daily-Stock-Prices/"
        start_date = datetime.strptime(last_date.split(" ")[0], date_format)
        df = pd.read_csv(filepath_or_buffer=price_path + self.config.ticker + ".csv", header=None)
        for index, row in enumerate(df.iterrows()):
            date_str = row[1][6]
            date = datetime.strptime(date_str, date_format)
            if date == start_date:
                return df.iloc[index - 1][4]

    '''
    Gets the daily price history DOES NOT currently calculate technical indicators
    '''

    def get_daily_price_history(self):
        date_format = "%Y-%m-%d"
        price_path = "data/Daily-Stock-Prices/"

        start_date = datetime.strptime(self.config.start_date, date_format)
        end_date = datetime.strptime(self.config.end_date, date_format)

        if (self.config.ticker + ".csv") in os.listdir(price_path):
            df = pd.read_csv(filepath_or_buffer=price_path + self.config.ticker + ".csv", header=None)
            open_price_list = df.iloc[:, 0].tolist()
            high_list = df.iloc[:, 1].tolist()
            low_list = df.iloc[:, 2].tolist()
            close_price_list = df.iloc[:, 4].tolist()
            volume_list = df.iloc[:, 5].tolist()
            date_list = df.iloc[:, 6].tolist()

            start_slice_index = 0
            end_slice_index = len(date_list)
            for index, date in enumerate(date_list):
                date = datetime.strptime(date, date_format)
                if date <= start_date:
                    start_slice_index = index + 1
                elif date > end_date:
                    end_slice_index = index
            open_price_list = open_price_list[start_slice_index:end_slice_index]
            close_price_list = close_price_list[start_slice_index:end_slice_index]
            high_list = high_list[start_slice_index:end_slice_index]
            low_list = low_list[start_slice_index:end_slice_index]
            date_list = date_list[start_slice_index:end_slice_index]
            volume_list = volume_list[start_slice_index:end_slice_index]
            return open_price_list, close_price_list, date_list, volume_list, high_list, low_list
        else:
            self.logger.error("Daily stock price file for ticker %s does not exist", self.config.ticker)
            sys.exit(-1)

    '''
    Gets the intraday price history and calculates the required technical indicators
    '''

    def get_intraday_price_history(self):
        date_format = "%Y-%m-%d"
        price_path = "data/Hourly-Stock-Prices"

        start_date = datetime.strptime(self.config.start_date, date_format)
        end_date = datetime.strptime(self.config.end_date, date_format)

        open_price_list = []
        close_price_list = []
        volume_list = []
        date_list = []
        high_list = []
        low_list = []
        for date in os.listdir(price_path):
            if (self.config.ticker + ".csv") in os.listdir(price_path + "/" + date):
                file_date = datetime.strptime(date, date_format)
                if start_date <= file_date <= end_date:
                    df = pd.read_csv(filepath_or_buffer=price_path + "/" + date + "/" + self.config.ticker + ".csv",
                                     header=None)
                    df = df.dropna()
                    open_price_list.extend(df.iloc[:, 0].tolist())
                    high_list.extend(df.iloc[:, 1].tolist())
                    low_list.extend(df.iloc[:, 2].tolist())
                    close_price_list.extend(df.iloc[:, 4].tolist())
                    volume_list.extend(df.iloc[:, 5].tolist())
                    date_list.extend(df.iloc[:, 6].tolist())

        offset = self.config.tech_indicator_offset * self.config.intervals_per_day
        sma_list, u_bollinger_band, l_bollinger_band = technical_indictors_calc.get_indicators(close_price_list, offset)
        return open_price_list, close_price_list[offset:], date_list[offset:], volume_list[offset:], \
               high_list[offset:], low_list[offset:], sma_list, u_bollinger_band, l_bollinger_band

    """
    Transforms social media sentiment data stored bi-hourly into a daily format so it can be used in the daily
    stock prediction model
    """

    def extract_daily_data(self, list):
        intervals_per_day = 48
        first_day_intervals = 33
        data = []
        i, j = 0, 0
        for value in list:
            if j == 0:
                data.append(value)
                j += 1
            elif j == first_day_intervals - 1 and i == 0:
                data[i] += value
                data[i] = data[i] / first_day_intervals
                j = 0
                i += 1
            elif j == intervals_per_day - 1:
                data[i] += value
                data[i] = data[i] / intervals_per_day
                j = 0
                i += 1
            else:
                data[i] += value
                j += 1
        return data[:-1]  # Remove last index as contains incomplete data

    '''
    Reads in the sentiment data and normalises it
    '''

    def get_sentiments(self, dates, times, is_stocktwits):
        length = len(times)

        if is_stocktwits:
            df = pd.read_csv(self.config.stocktwits_sentiment_save_file_path, index_col=[0, 1, 2])
            if self.config.use_untagged:
                df_untagged = pd.read_csv(self.config.stocktwits_sentiment_save_file_path_untagged, index_col=[0, 1, 2])
                df["raw_score"] = df["raw_score"] + df_untagged["raw_score"]
                df["weighted_score"] = df["weighted_score"] + df_untagged["weighted_score"]
                df["posts"] = df["posts"] + df_untagged["posts"]
        else:
            df = pd.read_csv(self.config.twitter_sentiment_save_file_path, index_col=[0, 1, 2])

        df = df[df.index.get_level_values("date").isin(dates)]

        if not self.config.daily_model:
            df = df[(df.index.get_level_values("interval") >= ((min(times) * 12) - 1)) & (
                    df.index.get_level_values("interval") < (max(times) * 12))]

        raw_scores = df["raw_score"].tolist()
        weighted_scores = df["weighted_score"].tolist()
        posts = df["posts"].tolist()

        if self.config.daily_model:
            raw_scores = self.extract_daily_data(raw_scores)
            weighted_scores = self.extract_daily_data(weighted_scores)
            posts = self.extract_daily_data(posts)

        reshaped_raw_scores = Series(raw_scores).values.reshape(-1, 1)
        raw_scores_scaler = MinMaxScaler(feature_range=(0, 1))
        raw_scores_normalized = raw_scores_scaler.fit_transform(reshaped_raw_scores)

        reshaped_weighted_scores = Series(weighted_scores).values.reshape(-1, 1)
        weighted_scores_scaler = MinMaxScaler(feature_range=(0, 1))
        weighted_scores_normalized = weighted_scores_scaler.fit_transform(reshaped_weighted_scores)

        reshaped_posts = Series(posts).values.reshape(-1, 1)
        posts_scaler = MinMaxScaler(feature_range=(0, 1))
        posts_normalized = posts_scaler.fit_transform(reshaped_posts)

        weighted_avg_scores = list(map(truediv, weighted_scores, posts))
        raw_avg_scores = list(map(truediv, raw_scores, posts))

        reshaped_weighted_avg_scores = Series(weighted_avg_scores).values.reshape(-1, 1)
        weighted_avg_scores_scaler = MinMaxScaler(feature_range=(0, 1))
        weighted_avg_scores_normalized = weighted_avg_scores_scaler.fit_transform(reshaped_weighted_avg_scores)

        reshaped_raw_avg_scores = Series(raw_avg_scores).values.reshape(-1, 1)
        raw_avg_scores_scaler = MinMaxScaler(feature_range=(0, 1))
        raw_avg_scores_normalized = raw_avg_scores_scaler.fit_transform(reshaped_raw_avg_scores)

        assert (length == len(raw_scores_normalized))
        assert (length == len(posts_normalized))
        assert (length == len(weighted_scores_normalized))
        assert (length == len(weighted_avg_scores_normalized))
        assert (length == len(raw_avg_scores_normalized))

        return weighted_scores_normalized, raw_scores_normalized, weighted_avg_scores_normalized, raw_avg_scores_normalized, posts_normalized

    def _normalize_targets(self, prices, target_prices):
        reshaped_prices = Series(prices).values.reshape(-1, 1)
        reshaped_target_prices = Series(target_prices).values.reshape(-1, 1)
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        prices_normalized = self.price_scaler.fit_transform(reshaped_prices)
        target_prices_normalized = self.price_scaler.fit_transform(reshaped_target_prices)
        return prices_normalized, target_prices_normalized

    def _normalize_data(self, price_data, volumes):
        price_scaler = MinMaxScaler(feature_range=(0, 1))

        normalized_inputs = []
        for input in price_data:
            reshaped_input = Series(input).values.reshape(-1, 1)
            input_normalized = price_scaler.fit_transform(reshaped_input)
            normalized_inputs.append(input_normalized)

        # Normalize volumes
        reshaped_volumes = Series(volumes).values.reshape(-1, 1)
        volumes_scaler = MinMaxScaler(feature_range=(0, 1))
        volumes_normalized = volumes_scaler.fit_transform(reshaped_volumes)
        normalized_inputs.append(volumes_normalized)

        return normalized_inputs

    """
    Converts the raw stock price inputs into percentage change between each time interval 
    and across the prediction offset
    """

    def _convert_to_percentage_change(self, prices, dates):
        target_offset = self.config.prediction_offset + self.config.sequence_length
        self.price_targets = prices[self.config.sequence_length:]

        prices.insert(0, self.get_previous_close_price(dates[0]))
        converted_prices = []
        converted_targets = []

        for i in range(1, len(prices)):
            converted_prices.append((prices[i] - prices[i - 1]) / prices[i - 1])

        for i in range(target_offset + 1, len(prices)):
            converted_targets.append(
                (prices[i] - prices[i - self.config.prediction_offset]) / prices[i - self.config.prediction_offset])
        return converted_prices, converted_targets

    def _generate_sequences(self):
        if self.config.daily_model:
            open_price, close_price, dates, volumes, high, low = self.get_daily_price_history()
        else:
            open_price, close_price, dates, volumes, high, low, sma, ubb, lbb = self.get_intraday_price_history()

        price_data = [open_price, close_price, high, low, sma, ubb, lbb]
        norm_open, norm_close, norm_high, norm_low, norm_volumes, norm_sma, norm_ubb, norm_lbb = self._normalize_data(
            price_data, volumes)
        converted_dates, converted_times, raw_dates = self._convert_dates(dates)

        if self.config.percentage_change:
            if self.config.predict_moving_average:
                prices, price_targets = self._convert_to_percentage_change(sma, dates)
            else:
                prices, price_targets = self._convert_to_percentage_change(close_price, dates)
        else:
            prices = close_price
            price_targets = prices[self.config.sequence_length + self.config.prediction_offset:]
        normalized_prices, normalized_target_prices = self._normalize_targets(prices, price_targets)

        if self.config.is_using_stocktwits():
            stocktwits_weighted_scores, stocktwits_raw_scores, stocktwits_weighted_avg_scores, stocktwits_raw_avg_scores, stocktwits_posts = self.get_sentiments(
                raw_dates, converted_times, True)
        if self.config.is_using_twitter():
            twitter_weighted_scores, twitter_raw_scores, twitter_weighted_avg_scores, twitter_raw_avg_scores, twitter_posts = self.get_sentiments(
                raw_dates, converted_times, False)

        self.sequence_data = {"inputs": [[] for i in range(self.config.get_num_features())],
                              "targets": normalized_target_prices}

        if self.config.predict_future:
            max_index = len(prices) - self.config.prediction_offset
            for i in range(self.config.sequence_length):
                self.sequence_data["targets"] = np.append(self.sequence_data["targets"], [0.5])
            self.sequence_data["targets"] = Series(self.sequence_data["targets"]).values.reshape(-1, 1)
        else:
            max_index = len(prices) - self.config.sequence_length - self.config.prediction_offset

        for i in range(0, max_index):
            input_index = 0
            if self.config.get_features()["use_stock_price_change"]:
                self.sequence_data["inputs"][input_index].append(
                    normalized_prices[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_open"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_open[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_close"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_close[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_high"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_high[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_low"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_low[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_sma"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_sma[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_bollinger_upper"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_ubb[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_price_bollinger_lower"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_lbb[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stock_volumes"]:
                self.sequence_data["inputs"][input_index].append(
                    norm_volumes[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_day_of_week"]:
                self.sequence_data["inputs"][input_index].append(converted_dates[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_time_of_day"]:
                self.sequence_data["inputs"][input_index].append(converted_times[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_weighted_stocktwits_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    stocktwits_weighted_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_raw_stocktwits_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    stocktwits_raw_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_weighted_avg_stocktwits_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    stocktwits_weighted_avg_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_raw_avg_stocktwits_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    stocktwits_raw_avg_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_stocktwits_num_posts"]:
                self.sequence_data["inputs"][input_index].append(
                    stocktwits_posts[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_weighted_twitter_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    twitter_weighted_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_raw_twitter_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    twitter_raw_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_weighted_avg_twitter_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    twitter_weighted_avg_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_raw_avg_twitter_sentiment"]:
                self.sequence_data["inputs"][input_index].append(
                    twitter_raw_avg_scores[i:i + self.config.sequence_length])
                input_index += 1
            if self.config.get_features()["use_twitter_num_posts"]:
                self.sequence_data["inputs"][input_index].append(
                    twitter_posts[i:i + self.config.sequence_length])
                input_index += 1

        for index, feature in enumerate(self.sequence_data["inputs"]):
            self.sequence_data["inputs"][index] = np.array(feature).reshape(len(feature),
                                                                            self.config.sequence_length)

    def inverse_normalization(self, data, start_index, end_index):
        true_values = self.price_scaler.inverse_transform(Series(data).values.reshape(-1, 1)).flatten().tolist()
        # If predicting percentage changes convert percentages to actual prices
        if self.config.percentage_change:
            if end_index is None or end_index == 0:
                targets_slice = self.price_targets[start_index:]
            else:
                targets_slice = self.price_targets[start_index:end_index]
            # assert len(true_percentages) == len(targets_slice)
            true_predictions = []
            for index, value in enumerate(true_values):
                true_predictions.append((value + 1) * targets_slice[index])
            return true_predictions, true_values
        else:
            return true_values, None

    def get_data_length(self):
        return len(self.sequence_data)

    def create_test_train_val_sequence(self):
        train_index = int(len(self.sequence_data["targets"]) * 0.8)
        train_data = []
        for feature in self.sequence_data["inputs"]:
            train_data.append(feature[:train_index])
        train_data_inputs = np.dstack(train_data)
        train_data_targets = self.sequence_data["targets"][:train_index]

        # Creates full training data set
        self.full_training_data = {"inputs": torch.tensor(train_data_inputs, dtype=torch.double),
                                   "targets": torch.tensor(train_data_targets, dtype=torch.double)}

        # Creates rolling cross validation folds of the training data
        self.training_data, self.validation_data = self.get_cross_val_folds(
            torch.tensor(train_data_inputs, dtype=torch.double),
            torch.tensor(train_data_targets, dtype=torch.double),
            self.config.folds)

        test_data = []
        for feature in self.sequence_data["inputs"]:
            test_data.append(feature[train_index:])
        test_data_inputs = np.dstack(test_data)
        test_data_targets = self.sequence_data["targets"][train_index:]

        # Creates the test data set
        self.test_data = {"inputs": torch.tensor(test_data_inputs, dtype=torch.double),
                          "targets": torch.tensor(test_data_targets, dtype=torch.double)}

    def get_model_data(self, training, validation, testing, fold):
        if testing:
            if training:
                return self.full_training_data["inputs"], self.full_training_data["targets"]
            else:
                return self.test_data["inputs"], self.test_data["targets"]
        elif training:
            return self.training_data["inputs"][fold], self.training_data["targets"][fold]
        elif validation:
            return self.validation_data["inputs"][fold], self.validation_data["targets"][fold]

    def get_cross_val_folds(self, training_inputs, training_targets, n_folds):
        training_data = {"inputs": [], "targets": []}
        validation_data = {"inputs": [], "targets": []}
        total_len = len(training_inputs)
        fold_len = int(total_len / (n_folds + 1))
        for i in range(1, n_folds + 1):
            training_data["inputs"].append(training_inputs[:fold_len * i])
            training_data["targets"].append(training_targets[:fold_len * i])
            training_data["inputs"][i - 1] = torch.cat(
                [training_data["inputs"][i - 1], training_inputs[fold_len * (i + 1):]], dim=0)
            training_data["targets"][i - 1] = torch.cat(
                [training_data["targets"][i - 1], training_targets[fold_len * (i + 1):]], dim=0)
            validation_data["inputs"].append(training_inputs[fold_len * i:fold_len * (i + 1)])
            validation_data["targets"].append(training_targets[fold_len * i:fold_len * (i + 1)])
        return training_data, validation_data

    def get_base_targets(self):
        pred, _ = self.inverse_normalization(self.sequence_data["targets"].flatten(),
                                             start_index=self.config.prediction_offset,
                                             end_index=None)
        return pred

    def get_prices(self, is_targets, is_sma, prediction_length):
        if self.config.daily_model:
            outputs = self.get_daily_price_history()
        else:
            outputs = self.get_intraday_price_history()
        if is_sma:
            prices = outputs[6]  # Get the close price history
        else:
            prices = outputs[1]  # Get the close price history

        if is_targets:
            current_prices = prices[
                             -(prediction_length + self.config.prediction_offset):]
        else:
            current_prices = prices[-prediction_length:]

        return current_prices
