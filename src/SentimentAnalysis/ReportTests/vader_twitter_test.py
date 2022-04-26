from DataProcessing import pre_processing
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from SentimentAnalysis.roberta_config import RobertaConfig
from helper_functions import format_time, get_logger

def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer(lexicon_file="../../../../data/Lexicons/NTUSD_plus_msc_student.txt")
    # sid_obj = SentimentIntensityAnalyzer()
    return calc_score(sid_obj, sentence)

def calc_score(sent_obj, sentence):
    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sent_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.005:
        return 1
    else:
        return 0

logger = get_logger(LOG_NAME="vader-twitter-test")

twitter_df = pd.read_csv("data/Downloaded/6k-dataset/stock_data.csv")
posts = twitter_df.iloc[:, 0].tolist()

twitter_predictions = []
for index, post in enumerate(posts):
    if index % 500 == 0:
        logger.info("index - %d", index)
    post = pre_processing.twitter_cleanup(post)
    twitter_predictions.append(sentiment_scores(post))

twitter_df["Tagged Sentiment"] = twitter_predictions

twitter_df.loc[twitter_df["Sentiment"] == -1, "Sentiment"] = 0

y_true = twitter_df["Sentiment"]
y_pred = twitter_df["Tagged Sentiment"]
logger.info(sklearn.metrics.classification_report(y_true, y_pred))
cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
cf_plot = sklearn.metrics.ConfusionMatrixDisplay(cf_matrix).plot(values_format="d")
plt.show()