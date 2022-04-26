class RobertaConfig:
    def __init__(self, daily_model=False):
        self.daily_model = daily_model

        # Model related parameters
        self.pre_trained_model_name = "roberta-base"
        self.epsilon = 1e-8
        self.max_post_len = 140

        # General nn config
        self.learning_rate = 2e-5
        self.batch_size = 32
        self.num_epochs = 5
        self.validation_split = 0.2

        # Data specific config
        self.ticker = "TSLA"
        self.num_labels = 2
        self.generate_daily_data = False

        # Data Locations
        self.raw_stocktwits_data_path = "data/StockTwits"
        self.training_data_path = "data/TrainingData/stocktwits_tagged_sentiment" + self.ticker + ".csv"
        self.testing_data_path = "data/TrainingData/testing_stocktwits_tagged_sentiment" + self.ticker + ".csv"
        self.stocktwits_sentiment_save_path = "data/stocktwits-sentiment"
        self.stocktwits_sentiment_save_file_path = self.stocktwits_sentiment_save_path + "/" + self.ticker + ".csv"
        self.stocktwits_sentiment_save_file_path_untagged = self.stocktwits_sentiment_save_path + "/untagged-" + self.ticker + ".csv"

        self.raw_twitter_data_path = "data/Twitter/scraped/filtered"
        self.twitter_sentiment_save_path = "data/twitter-sentiment"
        self.twitter_sentiment_save_file_path = self.twitter_sentiment_save_path + "/" + self.ticker + ".csv"
        self.twitter_id_var_path = "var/twitter-id.txt"

        self.price_path = "data/Daily-Stock-Prices/" + self.ticker + ".csv"

        # Non-performance related settings
        self.load_model = True
        self.model_save_location = "data/models/sentiment_best_model_state" + "-" + self.ticker + ".bin"
        # self.model_save_location = "data/models/sentiment_best_model_state.bin"
        # self.model_save_location = "data/models/sentiment_best_model_state" + "-mixed.bin"

    def get_price_path(self):
        return self.price_path

    def get_sentiment_file_path(self, is_stocktwits, is_tagged=True):
        if is_stocktwits:
            if is_tagged:
                return self.stocktwits_sentiment_save_file_path
            else:
                return self.stocktwits_sentiment_save_file_path_untagged
        else:
            return self.twitter_sentiment_save_file_path

    def get_sentiment_base_path(self, is_stocktwits):
        if is_stocktwits:
            return self.stocktwits_sentiment_save_path
        else:
            return self.twitter_sentiment_save_path

    def get_data_save_location(self):
        if self.load_model:
            return self.testing_data_path
        else:
            return self.training_data_path
