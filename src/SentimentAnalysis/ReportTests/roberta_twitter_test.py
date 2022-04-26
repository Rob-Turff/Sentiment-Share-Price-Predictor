import os
import time
from pathlib import Path

import sklearn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import matplotlib.pyplot as plt

from src.SentimentAnalysis.roberta_config import RobertaConfig
from src.helper_functions import get_logger, format_time
from src.DataProcessing import pre_processing


def get_predictions(model, device, dataloader):
    predictions = []

    t0 = time.time()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 40 == 0 and not i == 0:
                elapsed = format_time(time.time() - t0)
                eta = format_time(((time.time() - t0) / i) * (len(dataloader) - i))
                logger.info("Batch %d of %d ----- Elapsed time: %s ETA: %s", i, len(dataloader), elapsed, eta)

            batch_input_ids = batch["input_ids"].to(device)
            batch_mask = batch["attention_mask"].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_mask)
            _, preds = torch.max(outputs[0].detach(), dim=1)
            predictions.extend(preds)

    predictions = torch.stack(predictions).cpu()

    return predictions


class PostAnalysisDataset(Dataset):
    def __init__(self, config: RobertaConfig, tokenizer, data):
        self.posts = data.iloc[:, 0].tolist()
        self.pre_process()
        self.tokenizer = tokenizer
        self.max_len = config.max_post_len

    def __len__(self):
        return len(self.posts)

    def pre_process(self):
        for index, post in enumerate(self.posts):
            self.posts[index] = pre_processing.twitter_cleanup(post)

    def __getitem__(self, index):
        post = self.posts[index]

        encoding = self.tokenizer.encode_plus(post, add_special_tokens=True,
                                              truncation=True,
                                              max_length=self.max_len,
                                              padding="max_length",
                                              return_attention_mask=True,
                                              return_tensors='pt')

        return {
            "post_text": post,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def load_twitter_posts():
    df = pd.read_csv("data/Downloaded/6k-dataset/stock_data.csv")
    return df

if __name__ == "__main__":
    logger = get_logger(LOG_NAME="roberta-twitter-test")
    cf: RobertaConfig = RobertaConfig()
    roberta_tokenizer = RobertaTokenizer.from_pretrained(cf.pre_trained_model_name)

    twitter_df = load_twitter_posts()

    twitter_dataset = PostAnalysisDataset(cf, roberta_tokenizer, twitter_df)
    twitter_nn_dataloader = DataLoader(twitter_dataset, batch_size=cf.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s" % cf.model_save_location)
    if not os.path.exists(cf.model_save_location):
        logger.error("No model found, exiting...")
        exit(-1)

    model_path = Path(cf.model_save_location)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=cf.num_labels)
    model.to(device)

    twitter_predictions = get_predictions(model, device, twitter_nn_dataloader)

    twitter_df["Tagged Sentiment"] = twitter_predictions

    twitter_df.loc[twitter_df["Sentiment"] == -1, "Sentiment"] = 0

    y_true = twitter_df["Sentiment"]
    y_pred = twitter_df["Tagged Sentiment"]
    logger.info(sklearn.metrics.classification_report(y_true, y_pred))
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    cf_plot = sklearn.metrics.ConfusionMatrixDisplay(cf_matrix).plot(values_format="d")
    plt.show()
