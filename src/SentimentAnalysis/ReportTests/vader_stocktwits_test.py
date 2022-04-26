from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from src.SentimentAnalysis.roberta_config import RobertaConfig
from src.helper_functions import format_time, get_logger


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer(lexicon_file="../../../../data/Lexicons/NTUSD_plus_msc_student.txt")
    return calc_score(sid_obj, sentence)

def calc_score(sent_obj, sentence):
    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sent_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0:
        return 1
    else:
        return 0

def start():
    logger = get_logger(LOG_NAME="vader-stocktwits-test")

    cf: RobertaConfig = RobertaConfig()

    training_data = pd.read_csv(cf.testing_data_path, names=["text", "sentiment"])
    posts = training_data.iloc[:, 0].tolist()
    y_true = training_data.iloc[:, 1].tolist()
    y_pred = []
    for index, post in enumerate(posts):
        if index % 100 == 0:
            print(str(index) + "/" + str(len(posts)))
        y_pred.append(sentiment_scores(post))

    logger.info("run for %s", cf.ticker)
    logger.info(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_plot = ConfusionMatrixDisplay(cf_matrix).plot(values_format="d")
    plt.show()


start()