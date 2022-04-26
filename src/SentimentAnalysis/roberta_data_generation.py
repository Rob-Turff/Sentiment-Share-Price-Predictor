import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.SentimentAnalysis.roberta_config import RobertaConfig
from src.SentimentAnalysis.roberta_data import RobertaData
from src.helper_functions import get_logger, format_time


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
        self.tokenizer = tokenizer
        self.max_len = config.max_post_len

    def __len__(self):
        return len(self.posts)

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


if __name__ == "__main__":
    logger = get_logger(LOG_NAME="roberta_data_generation")
    cf: RobertaConfig = RobertaConfig()
    roberta_tokenizer = RobertaTokenizer.from_pretrained(cf.pre_trained_model_name)

    data_loader = RobertaData(cf)
    stocktwits_df = data_loader.get_ticker_stocktwits_posts(cf.ticker)
    twitter_df = data_loader.get_ticker_twitter_posts(cf.ticker)

    stocktwits_dataset = PostAnalysisDataset(cf, roberta_tokenizer, stocktwits_df)
    twitter_dataset = PostAnalysisDataset(cf, roberta_tokenizer, twitter_df)
    stocktwits_nn_dataloader = DataLoader(stocktwits_dataset, batch_size=cf.batch_size)
    twitter_nn_dataloader = DataLoader(twitter_dataset, batch_size=cf.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s" % cf.model_save_location)
    if not os.path.exists(cf.model_save_location):
        logger.error("No model found, exiting...")
        exit(-1)

    model_path = Path(cf.model_save_location)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=cf.num_labels)
    model.to(device)

    stocktwits_predictions = get_predictions(model, device, stocktwits_nn_dataloader)
    twitter_predictions = get_predictions(model, device, twitter_nn_dataloader)

    stocktwits_df["Sentiment"] = stocktwits_predictions
    twitter_df["Sentiment"] = twitter_predictions

    stocktwits_df.loc[stocktwits_df["Sentiment"] == 0, "Sentiment"] = -1
    twitter_df.loc[twitter_df["Sentiment"] == 0, "Sentiment"] = -1

    data_loader.store_sentiments(stocktwits_df, is_stocktwits=True, is_tagged=False)
    data_loader.store_sentiments(twitter_df, is_stocktwits=False)
