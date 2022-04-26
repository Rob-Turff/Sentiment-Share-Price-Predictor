import torch


class NNConfig:
    def __init__(self, daily_model=False):
        self.daily_model = daily_model

        if daily_model:
            # Data specific config
            self.sequence_length = 5
            self.prediction_offset = 0  # how far in the future to predict 0 = 1 day
            self.start_date = "2018-01-01"
            self.end_date = "2021-03-01"
            self.save_text = "daily"
        else:
            # Data specific config
            self.sequence_length = 13
            self.prediction_offset = 13  # how far in the future to predict 0 = 30 minutes 12 = 1 trading day
            self.start_date = "2020-10-02"
            self.end_date = "2021-04-19"
            self.save_text = "intraday"

        self.ticker = "TSLA"
        self.predict_future = False  # if true predict values which don't have data to validate with - currently doesn't work
        self.use_untagged = True
        self.intervals_per_day = 13
        self.tech_indicator_offset = 2  # How many days to use for technical indicator calculations
        self.plot_base_price = True

        # Model specific config
        self.use_CNN = True
        self.bidirectional = False
        self.hidden_layer_size = 150
        self.num_layers = 2
        self.split_sentiment = False
        self.model_hidden_outputs = 100

        # CNN config
        self.conv_kernel = 3
        self.conv_stride = 1
        self.conv_outputs = 25
        self.maxpool_kernel = 4
        self.maxpool_stride = 1

        # General nn config
        self.learning_rate = 0.00025
        self.weight_decay = 5e-6
        self.dropout = 0.1
        self.weight_clipping = 1.0
        self.num_epochs = 150
        self.batch_size = 8
        self.loss_function = torch.nn.MSELoss()  # Currently taking square root so actually RMSE error

        # Features to use
        self.intraday_features = {
            "use_stock_price_change": True,
            "use_stock_price_open": True,
            "use_stock_price_close": True,
            "use_stock_price_high": True,
            "use_stock_price_low": True,
            "use_stock_price_sma": True,
            "use_stock_price_bollinger_upper": True,
            "use_stock_price_bollinger_lower": True,
            "use_stock_volumes": True,
            "use_day_of_week": True,
            "use_time_of_day": True,
            "use_weighted_stocktwits_sentiment": True,
            "use_raw_stocktwits_sentiment": True,
            "use_weighted_avg_stocktwits_sentiment": True,
            "use_raw_avg_stocktwits_sentiment": True,
            "use_stocktwits_num_posts": True,
            "use_weighted_twitter_sentiment": True,
            "use_raw_twitter_sentiment": True,
            "use_weighted_avg_twitter_sentiment": True,
            "use_raw_avg_twitter_sentiment": True,
            "use_twitter_num_posts": True,
        }

        self.daily_features = {
            "use_stock_price_change": True,
            "use_stock_price_open": True,
            "use_stock_price_close": True,
            "use_stock_price_high": True,
            "use_stock_price_low": True,
            "use_stock_price_sma": True,
            "use_stock_price_bollinger_upper": True,
            "use_stock_price_bollinger_lower": True,
            "use_stock_volumes": True,
            "use_day_of_week": True,
            "use_time_of_day": False,  # Must be false for daily data
            "use_weighted_stocktwits_sentiment": True,
            "use_raw_stocktwits_sentiment": True,
            "use_weighted_avg_stocktwits_sentiment": True,
            "use_raw_avg_stocktwits_sentiment": True,
            "use_stocktwits_num_posts": True,
            "use_weighted_twitter_sentiment": True,
            "use_raw_twitter_sentiment": True,
            "use_weighted_avg_twitter_sentiment": True,
            "use_raw_avg_twitter_sentiment": True,
            "use_twitter_num_posts": True,
        }

        self.outputs = {
            "close_price": True
        }

        self.percentage_change = True
        self.predict_moving_average = True

        # Non-performance related settings
        self.load_model = False
        self.save_model = True
        self.folds = 4

        # File paths
        self.model_save_location = "data/models/stock_best_model_state_" + str(
            self.prediction_offset) + "-" + self.save_text + ".bin"
        self.stock_prediction_var_path = "var/stock_prediction_best_model" + str(
            self.prediction_offset) + "-" + self.save_text + ".txt"
        self.hyperparameter_save_location = "var/hyperparameters-" + self.ticker + "-" + self.save_text + ".csv"

        self.stocktwits_sentiment_save_path = "data/stocktwits-sentiment"
        self.stocktwits_sentiment_save_file_path = self.stocktwits_sentiment_save_path + "/" + self.ticker + ".csv"
        self.stocktwits_sentiment_save_file_path_untagged = self.stocktwits_sentiment_save_path + "/untagged-" + self.ticker + ".csv"
        self.twitter_sentiment_save_path = "data/twitter-sentiment"
        self.twitter_sentiment_save_file_path = self.twitter_sentiment_save_path + "/" + self.ticker + ".csv"

    def get_cnn_linear_size(self):
        l1_out = int((self.sequence_length - (self.conv_kernel - 1) - 1) / self.conv_stride + 1)
        l2_out = int((l1_out - (self.maxpool_kernel - 1) - 1) / self.maxpool_stride + 1)
        return l2_out * self.conv_outputs

    def get_features(self):
        if self.daily_model:
            return self.daily_features
        else:
            return self.intraday_features

    # Get the number to scale input size of linear layer to account for inputs from multiple networks
    def get_size_multiplier(self, ignore_split=False):
        multiplier = 1
        if self.bidirectional:
            multiplier *= 2
        if self.split_sentiment and not ignore_split:
            multiplier *= 2
        return multiplier

    def get_num_features(self, sentiment=False, ignore_split=True):
        if self.split_sentiment and not ignore_split:
            return len(self.get_feature_indexes(sentiment))
        else:
            num_features = 0
            for feature in self.get_features():
                if self.get_features()[feature]:
                    num_features += 1
        return num_features

    def get_feature_indexes(self, sentiment):
        if sentiment:
            features_to_use = [False, False, False, False, False, False, False, False, False, True, True, True, True,
                               True, True, True, True, True, True, True, True]
        else:
            features_to_use = [True, True, True, True, True, True, True, True, True, True, True, False, False, False,
                               False, False, False, False, False, False, False]
        indexes = []
        index = 0
        for i, feature in enumerate(self.get_features().values()):
            if feature and features_to_use[i]:
                indexes.append(index)
                index += 1
            elif feature:
                index += 1
        return indexes

    def get_num_outputs(self):
        num_outputs = 0
        for output in self.outputs:
            if self.outputs[output]:
                num_outputs += 1
        return num_outputs

    def set_parameters(self, param_dict):
        self.split_sentiment = param_dict["split_sentiment"]
        self.bidirectional = param_dict["bidirectional"]
        self.use_CNN = param_dict["use_cnn"]
        self.batch_size = param_dict["batch_size"]
        self.num_layers = param_dict["num_layers"]
        self.hidden_layer_size = param_dict["hidden_layer_size"]
        self.weight_decay = param_dict["weight_decay"]
        self.dropout = param_dict["dropout"]
        self.num_epochs = param_dict["num_epochs"]
        self.sequence_length = param_dict["sequence_length"]
        self.prediction_offset = param_dict["prediction_offset"]
        self.conv_kernel = param_dict["conv_kernel"]
        self.conv_stride = param_dict["conv_stride"]
        self.maxpool_kernel = param_dict["maxpool_kernel"]
        self.maxpool_stride = param_dict["maxpool_stride"]
        self.conv_outputs = param_dict["conv_outputs"]
        for index, feature in enumerate(self.get_features()):
            if self.daily_model:
                self.daily_features[feature] = param_dict["features"][index]
            else:
                self.intraday_features[feature] = param_dict["features"][index]

    def get_current_parameters(self):
        param_dict = {"split_sentiment": self.split_sentiment,
                      "bidirectional": self.bidirectional,
                      "use_cnn": self.use_CNN,
                      "num_layers": self.num_layers,
                      "hidden_layer_size": self.hidden_layer_size,
                      "weight_decay": self.weight_decay,
                      "dropout": self.dropout,
                      "batch_size": self.batch_size,
                      "num_epochs": self.num_epochs,
                      "sequence_length": self.sequence_length,
                      "prediction_offset": self.prediction_offset,
                      "conv_kernel": self.conv_kernel,
                      "conv_stride": self.conv_stride,
                      "maxpool_kernel": self.maxpool_kernel,
                      "maxpool_stride": self.maxpool_stride,
                      "conv_outputs": self.conv_outputs,
                      "features": list(self.get_features().values())}
        return param_dict

    '''
    Generates the parameter dictionaries, set use default to True to only run for the parameters set at top of class
    '''
    def generate_param_dict(self, use_default=True, run_from=None):
        param_dict_list = []
        if use_default:
            param_dict_list.append(self.get_current_parameters())
        else:
            bidirectional_options = [False]
            split_sentiment_options = [False]
            use_cnn_options = [True]
            num_layers_options = [2]
            batch_size_options = [8]
            hidden_layer_size_options = [50, 100, 150]
            weight_decay_options = [5e-6]
            dropout_options = [0.1]
            num_epochs_options = [50, 100, 150]
            sequence_length_options = [13]
            prediction_offset_options = [13]
            conv_kernel_options = [3]
            conv_stride_options = [1]
            maxpool_kernel_options = [4]
            maxpool_stride_options = [1]
            conv_outputs_options = [25]
            features_options = [
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True]]
            for split in split_sentiment_options:
                for use_cnn in use_cnn_options:
                    for direction in bidirectional_options:
                        for layer in num_layers_options:
                            for batch_size in batch_size_options:
                                for h_layer in hidden_layer_size_options:
                                    for weight_decay in weight_decay_options:
                                        for dropout in dropout_options:
                                            for epoch in num_epochs_options:
                                                for length in sequence_length_options:
                                                    for offset in prediction_offset_options:
                                                        for conv_kernel in conv_kernel_options:
                                                            for conv_stride in conv_stride_options:
                                                                for max_kernel in maxpool_kernel_options:
                                                                    for max_stride in maxpool_stride_options:
                                                                        for conv_outputs in conv_outputs_options:
                                                                            for feature in features_options:
                                                                                para_dict = {"split_sentiment": split,
                                                                                             "bidirectional": direction,
                                                                                             "use_cnn": use_cnn,
                                                                                             "num_layers": layer,
                                                                                             "hidden_layer_size": h_layer,
                                                                                             "weight_decay": weight_decay,
                                                                                             "dropout": dropout,
                                                                                             "batch_size": batch_size,
                                                                                             "num_epochs": epoch,
                                                                                             "sequence_length": length,
                                                                                             "prediction_offset": offset,
                                                                                             "conv_kernel": conv_kernel,
                                                                                             "conv_stride": conv_stride,
                                                                                             "maxpool_kernel": max_kernel,
                                                                                             "maxpool_stride": max_stride,
                                                                                             "conv_outputs": conv_outputs,
                                                                                             "features": feature}
                                                                                param_dict_list.append(para_dict)
        if run_from is not None and not use_default:
            return param_dict_list[run_from:]
        else:
            return param_dict_list

    def is_using_stocktwits(self):
        if self.get_features()["use_weighted_stocktwits_sentiment"]:
            return True
        elif self.get_features()["use_raw_stocktwits_sentiment"]:
            return True
        elif self.get_features()["use_weighted_avg_stocktwits_sentiment"]:
            return True
        elif self.get_features()["use_raw_avg_stocktwits_sentiment"]:
            return True
        else:
            return False

    def is_using_twitter(self):
        if self.get_features()["use_weighted_twitter_sentiment"]:
            return True
        elif self.get_features()["use_raw_twitter_sentiment"]:
            return True
        elif self.get_features()["use_weighted_avg_twitter_sentiment"]:
            return True
        elif self.get_features()["use_raw_avg_twitter_sentiment"]:
            return True
        else:
            return False
